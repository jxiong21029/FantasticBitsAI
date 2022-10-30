import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch

from experiments.distillation.redistill import ReDistillAgents
from ppo import PPOConfig, PPOTrainer
from utils import PROJECT_DIR

matplotlib.use("Tkagg")
SMOKE_TEST = False


@ray.remote
def run_trial(task_id):
    agents = ReDistillAgents(
        num_layers=2, d_model=64, nhead=2, dim_feedforward=128, share_parameters=True
    )
    agents.load_state_dict(torch.load(os.path.join(PROJECT_DIR, "data/pretrained.pth")))

    trainer = PPOTrainer(
        agents,
        config=PPOConfig(
            lr=2e-4,
            gae_lambda=0.95,
            weight_decay=1e-4,
            rollout_steps=128 if SMOKE_TEST else 4096,
            minibatch_size=128 if SMOKE_TEST else 512,
            ppo_clip_coeff=0.15,
            grad_clipping=10.0,
            entropy_reg=10**-5.5,
            value_loss_wt=2,
            env_kwargs={
                "reward_shaping_snaffle_goal_dist": True,
                "reward_own_goal": 2,
                "reward_teammate_goal": 2,
                "reward_opponent_goal": -2,
            },
        ),
    )

    means = []
    stds = []
    trainer.vectorized_evaluate(10 if SMOKE_TEST else 200)
    means.append(trainer.logger.cumulative_data["eval_goals_scored_mean"][-1])
    stds.append(trainer.logger.cumulative_data["eval_goals_scored_std"][-1])
    for i in range(2 if SMOKE_TEST else 30):
        if SMOKE_TEST:
            print(f"{task_id=} starting iteration {i}")
        elif i % 10 == 0:
            print(f"{task_id=} starting iteration {i + 1}/30")
        trainer.run()
        trainer.vectorized_evaluate(10 if SMOKE_TEST else 200)
        means.append(trainer.logger.cumulative_data["eval_goals_scored_mean"][-1])
        stds.append(trainer.logger.cumulative_data["eval_goals_scored_std"][-1])
    return means, stds


def main():
    tasks = [run_trial.remote(i) for i in range(2 if SMOKE_TEST else 12)]
    results = ray.get(tasks)

    fig, ax = plt.subplots()
    fig.suptitle("PPO w/ pretrained pi+vf")
    ax.set_xlabel("iterations (4096 timesteps / iter)")
    ax.set_ylabel("avg goals scored (higher is better)")

    for means, stds in results:
        means, stds = np.array(means), np.array(stds)
        ax.plot(means)
        # (line,) = ax.plot(means)
        # ax.fill_between(
        #     np.arange(len(means)),
        #     means - stds,
        #     means + stds,
        #     alpha=0.2,
        #     color=line.get_color(),
        # )
    fig.savefig("does_straight_ppo_collapse.png")


if __name__ == "__main__":
    main()
