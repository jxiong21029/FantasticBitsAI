import pickle

import numpy as np
import torch
import torch.nn as nn
import tqdm

from env import SZ_BLUDGER, SZ_GLOBAL, SZ_SNAFFLE, SZ_WIZARD, FantasticBits
from trainer import Trainer
from utils import Logger, component_grad_norms


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


class BCTrainer(Trainer):
    def __init__(
        self,
        agents,
        demo_filename,
        lr,
        minibatch_size,
        weight_decay,
        grad_clipping,
        env_kwargs=None,
        seed=None,
    ):
        super().__init__(env_kwargs=env_kwargs, seed=seed)
        self._agents = agents
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

    @property
    def agents(self):
        return self._agents

    def train(self):
        self.train()

        idx = np.arange(self.demo_sz)
        self.rng.shuffle(idx)

        for i in range(idx.shape[0] // self.minibatch_size):
            batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]
            logp, _ = self.agents.policy_forward(self.rollout, batch_idx)

            loss = -logp.mean()
            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            self.logger.log(loss=loss.item())

            norms = component_grad_norms(
                self.agents, exclude=("value_encoder", "value_head")
            )
            self.logger.log(**{"grad_norm_" + k: v for k, v in norms.items()})
            if self.grad_clipping is not None:
                nn.utils.clip_grad_norm_(self.agents.parameters(), self.grad_clipping)
                self.logger.log(
                    grad_clipped=(norms["total"] > self.grad_clipping),
                )

            self.optim.step()

        self.logger.step()


def main():
    from architectures import VonMisesAgents

    trainer = BCTrainer(
        VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128),
        demo_filename="data/basic_demo.pickle",
        lr=10**-2.5,
        minibatch_size=512,
        weight_decay=1e-5,
        grad_clipping=10.0,
    )
    trainer.evaluate()
    for i in tqdm.trange(100):
        trainer.train()
        if i % 20 == 19:
            trainer.evaluate()
            trainer.logger.generate_plots(dirname="plotgen_bc/")
    torch.save(trainer.agents.state_dict(), "bc_agents.pth")
    for _ in range(5):
        trainer.evaluate_with_render()


if __name__ == "__main__":
    main()
