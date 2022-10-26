import copy
import os
import random

import torch
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.sample import Domain

from experiments.distillation.redistill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from utils import PROJECT_DIR

SMOKE_TEST = False


def train(config):
    agents = ReDistillAgents(
        num_layers=2, d_model=64, nhead=2, dim_feedforward=128, share_parameters=True
    )

    trainer = JointReDistillTrainer(
        agents,
        demo_filename=os.path.join(PROJECT_DIR, "data/basic_demo.pickle"),
        beta_bc=config["beta_bc"],
        ppo_config=PPOConfig(
            lr=config["lr"],
            gae_lambda=0.95,
            weight_decay=1e-4,
            rollout_steps=128 if SMOKE_TEST else 4096,
            minibatch_size=64 if SMOKE_TEST else 512,
            epochs=3,
            entropy_reg=config["entropy_reg"],
            ppo_clip_coeff=config["ppo_clip_coeff"],
            value_loss_wt=config["beta_vf"],
            grad_clipping=10.0,
            env_kwargs={
                "reward_shaping_snaffle_goal_dist": True,
                "reward_own_goal": 2,
                "reward_teammate_goal": 2,
                "reward_opponent_goal": -2,
            },
        ),
    )

    if checkpoint := session.get_checkpoint():  # if resuming from a checkpoint
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_data = torch.load(os.path.join(ckpt_dir, "state.pt"))

            step = ckpt_data["step"]
            agents.load_state_dict(ckpt_data["agents_state_dict"])
            trainer.optim.load_state_dict(ckpt_data["optim_state_dict"])
    else:
        step = 0
        agents.load_state_dict(
            torch.load(os.path.join(PROJECT_DIR, "data/pretrained.pth"))
        )

    while True:
        trainer.run()
        for p in agents.parameters():
            assert not p.isnan().any()

        if step % 50 == 49 or (SMOKE_TEST and step % 3 == 2):
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(
                {
                    "step": step + 1,
                    "agents_state_dict": trainer.agents.state_dict(),
                    "optim_state_dict": trainer.optim.state_dict(),
                },
                "checkpoint/state.pt",
            )
            new_checkpoint = air.Checkpoint.from_directory("checkpoint")

            trainer.vectorized_evaluate(10 if SMOKE_TEST else 200)
            trainer.logger.air_report(checkpoint=new_checkpoint)
        step += 1


def main():
    param_space = {
        "lr": tune.loguniform(10**-4, 10**-3),
        "beta_bc": tune.loguniform(10**-2.5, 1),
        "beta_vf": tune.loguniform(1e-1, 1e1),
        "entropy_reg": tune.loguniform(10**-6, 10**-4),
        "ppo_clip_coeff": tune.uniform(0.02, 0.2),
    }

    resample_prob = 0.25

    def custom_get_new_config(_trial, trial_to_clone):
        config = trial_to_clone.config
        new_config = copy.deepcopy(config)
        for k, space in param_space.items():
            if isinstance(space, Domain):
                if random.random() < resample_prob:
                    new_config[k] = space.sample()
                elif r := random.random() < 1 / 3:
                    new_config[k] = config[k] * 0.8
                elif r > 2 / 3:
                    new_config[k] = config[k] * 1.25
            else:
                assert isinstance(space, list)
                if random.random() < resample_prob:
                    new_config[k] = random.choice(space)
                r = random.random()
                idx = space.index(config[k])
                if r < 1 / 3 and idx - 1 >= 0:
                    new_config[k] = space[idx - 1]
                elif r > 2 / 3 and idx + 1 < len(space):
                    new_config[k] = space[idx + 1]
        return new_config

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_goals_scored_mean",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations=param_space,
    )
    pbt._get_new_config = custom_get_new_config

    tuner = tune.Tuner(
        train,
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=-1,
            max_concurrent_trials=12,
        ),
        run_config=air.RunConfig(
            name="limit_test",
            local_dir="ray_results",
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
