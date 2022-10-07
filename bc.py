import pickle

import numpy as np
import torch
import tqdm
from ray import air, tune

from agents import Agents
from env import SZ_BLUDGER, SZ_GLOBAL, SZ_SNAFFLE, SZ_WIZARD, FantasticBits
from tuning import IntervalHalvingSearch, log_halving_search
from utils import Logger, grad_norm


def stack_obs(dicts):
    size = len(dicts)
    ret = {
        "global": np.zeros((size, SZ_GLOBAL), dtype=np.float32),
    }
    for i in range(4):
        ret[f"wizard{i}"] = np.zeros((size, SZ_WIZARD), dtype=np.float32)
    for i in range(7):
        ret[f"snaffle{i}"] = np.full((size, SZ_SNAFFLE), np.nan, dtype=np.float32)
    for i in range(2):
        ret[f"bludger{i}"] = np.zeros((size, SZ_BLUDGER), dtype=np.float32)

    for i in range(len(dicts)):
        for k in dicts[i].keys():
            ret[k][i] = dicts[i][k]
    return ret


def stack_actions(dicts):
    size = len(dicts)
    ret = {
        "id": np.zeros((size, 2), dtype=np.int64),
        "target": np.zeros(
            (size, 2, 2),
            dtype=np.float32,
        ),
    }
    for i in range(len(dicts)):
        for k in dicts[i].keys():
            ret[k][i] = dicts[i][k]
    return ret


def generate_demonstrations(num_episodes=50):
    env = FantasticBits(bludgers_enabled=True, opponents_enabled=True)
    obs_buf = []
    act_buf = []
    for _ in tqdm.trange(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_buf.append(obs)

            actions = {
                "id": np.zeros(2, dtype=np.int64),
                "target": np.zeros((2, 2), dtype=np.float32),
            }
            for i in range(2):
                wizard = env.agents[i]
                if wizard.grab_cd == 2:
                    actions["id"][i] = 1
                    actions["target"][i] = (16000 - wizard.x, 3750 - wizard.y)
                else:
                    actions["id"][i] = 0
                    nearest_snaffle = min(
                        env.snaffles, key=lambda s: s.distance2(wizard)
                    )
                    actions["target"][i] = (
                        nearest_snaffle.x - wizard.x,
                        nearest_snaffle.y - wizard.y,
                    )

            act_buf.append(actions)

            obs, _, done = env.step(actions)

    ret_obs = stack_obs(obs_buf)
    ret_act = stack_actions(act_buf)

    with open("data/basic_demo.pickle", "wb") as f:
        pickle.dump((ret_obs, ret_act), f)


class BCTrainer:
    def __init__(
        self,
        agents,
        demo_filename,
        lr,
        minibatch_size,
        weight_decay,
        grad_clipping,
        seed=None,
    ):
        self.agents = agents
        self.optim = torch.optim.Adam(
            agents.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.minibatch_size = minibatch_size
        self.grad_clipping = grad_clipping

        with open(demo_filename, "rb") as f:
            demo_obs, demo_actions = pickle.load(f)
        self.rollout = {
            "obs": {k: torch.tensor(v) for k, v in demo_obs.items()},
            "act": {k: torch.tensor(v) for k, v in demo_actions.items()},
        }
        for i in range(2):
            self.rollout["act"]["target"] /= torch.norm(
                self.rollout["act"]["target"], dim=2, keepdim=True
            )

        self.demo_sz = demo_obs["global"].shape[0]

        self.rng = np.random.default_rng(seed=seed)
        self.logger = Logger()

    def train(self):
        idx = np.arange(self.demo_sz)
        self.rng.shuffle(idx)

        for i in range(idx.shape[0] // self.minibatch_size):
            batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]
            logp, distrs = self.agents.policy_forward(self.rollout, batch_idx)

            loss = -logp.mean()
            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            self.logger.log(loss=loss.item())

            norm = grad_norm(self.agents)
            self.logger.log(grad_norm=norm)
            if self.grad_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.agents.parameters(), self.grad_clipping
                )
                self.logger.log(
                    grad_clipped=(norm > self.grad_clipping),
                )

            self.optim.step()

        self.logger.step()

    def evaluate(self, num_episodes=50):
        temp_logger = Logger()
        eval_env = FantasticBits(
            bludgers_enabled=True, opponents_enabled=True, logger=temp_logger
        )
        for _ in range(num_episodes):
            obs = eval_env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    actions, _ = self.agents.step(obs)
                obs, _, done = eval_env.step(actions)
        temp_logger.step()
        self.logger.cumulative_data.update(
            {"eval_" + k: v for k, v in temp_logger.cumulative_data.items()}
        )


# TODO: fix behavioral cloning loss, use cosine angle? normalize to length 1 mean?
# TODO: tune behavioral cloning, analyze data scaling
# TODO: use behavioral cloning performance as a proxy for RL model capacity


def train(config):
    trainer = BCTrainer(
        Agents(
            num_layers=config["num_layers"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            action_parameterization=config["action_parameterization"],
            dispersion_scale=config["dispersion_scale"],
        ),
        "../../../data/basic_demo.pickle",
        lr=config["lr"],
        minibatch_size=config["minibatch_size"],
        weight_decay=config["weight_decay"],
        grad_clipping=None,
    )
    for i in range(101):
        trainer.train()
        if i % 20 == 0:
            trainer.evaluate()
            tune.report(**{k: v[-1] for k, v in trainer.logger.cumulative_data.items()})


def main():
    for action_parameterization in ("euclidean", "normed_euclidean", "von_mises"):
        search_alg = IntervalHalvingSearch(
            search_space={
                "action_parameterization": action_parameterization,
                "lr": log_halving_search(1e-4, 1e-3, 1e-2),
                "dispersion_scale": log_halving_search(1e-1, 1e0, 1e1),
                "minibatch_size": 128,
                "weight_decay": 1e-5,
                "num_layers": 1,
                "d_model": 32,
                "nhead": 2,
            },
            depth=2,
            metric="eval_goals_scored_mean",
            mode="max",
        )

        tuner = tune.Tuner(
            train,
            tune_config=tune.TuneConfig(
                num_samples=-1,
                search_alg=search_alg,
                max_concurrent_trials=8,
            ),
            run_config=air.RunConfig(
                name="action_parameterization",
                local_dir="ray_results/",
            ),
        )
        tuner.fit()

        print(f"results for {action_parameterization=}")
        print("best score:", search_alg.best_score)
        print("best config:", search_alg.best_config)


def main2():
    trainer = BCTrainer(
        Agents(action_parameterization="von_mises"),
        demo_filename="data/basic_demo.pickle",
        lr=1e-3,
        minibatch_size=128,
        weight_decay=1e-5,
        grad_clipping=10.0,
    )
    trainer.train()
    trainer.evaluate()
    print({k: v[-1] for k, v in trainer.logger.cumulative_data.items()})


if __name__ == "__main__":
    # generate_demonstrations(100)
    # main()
    main2()
