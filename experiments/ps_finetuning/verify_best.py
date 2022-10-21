import os

import numpy as np
import ray

from experiments.distillation.repr_distill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from utils import Logger


SMOKE_TEST = False

dirs = os.getcwd().split("/")
root_dir = os.path.join(*dirs[: 1 + dirs.index("FantasticBits")])


@ray.remote(num_cpus=1)
def run_trial():
    agents = ReDistillAgents(
        num_layers=2,
        d_model=64,
        nhead=2,
        dim_feedforward=128,
        share_parameters=True,
    )

    trainer = JointReDistillTrainer(
        agents,
        demo_filename=os.path.join("/", root_dir, "data/basic_demo.pickle"),
        beta_bc=0.3,
        ppo_config=PPOConfig(
            lr=10**-3.5,
            gamma=0.99,
            gae_lambda=0.97,
            weight_decay=1e-5,
            rollout_steps=4096,
            minibatch_size=512,
            epochs=3,
            ppo_clip_coeff=0.1,
            grad_clipping=None,
            entropy_reg=1e-5,
            value_loss_wt=0.1,
            env_kwargs={
                "reward_shaping_snaffle_goal_dist": True,
                "reward_own_goal": 2,
                "reward_teammate_goal": 2,
                "reward_opponent_goal": -2,
            },
        ),
    )

    phase1_logger = Logger()
    trainer.pretrain_policy(
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1 if SMOKE_TEST else 50,
        logger=phase1_logger,
    )
    trainer.vectorized_evaluate(5 if SMOKE_TEST else 200)

    p1g = trainer.logger.cumulative_data["eval_goals_scored_mean"][-1]

    phase2_logger = Logger()
    trainer.pretrain_value(
        lr=1e-4,
        weight_decay=1e-3,
        beta_kl=1,
        epochs=1 if SMOKE_TEST else 50,
        logger=phase2_logger,
    )
    trainer.vectorized_evaluate(5 if SMOKE_TEST else 200)

    p2g = trainer.logger.cumulative_data["eval_goals_scored_mean"][-1]

    return p1g, p2g


def iqm_ci(dataset):
    results = []
    for _ in range(1000):
        data = np.random.choice(dataset, size=len(dataset))
        lower_bound = np.percentile(data, 25, method="lower")
        upper_bound = np.percentile(data, 75, method="higher")

        results.append(np.mean(data[(lower_bound <= data) & (data <= upper_bound)]))

    return (
        np.nanpercentile(results, 2.5, axis=0),
        np.nanmean(results, axis=0),
        np.nanpercentile(results, 97.5, axis=0),
    )


def main():
    ray.init(num_cpus=8)

    futures = ray.get([run_trial.remote() for _ in range(20)])

    scores_1 = np.zeros(20)
    scores_2 = np.zeros(20)
    for i in range(20):
        s1, s2 = futures[i]
        scores_1[i] = s1
        scores_2[i] = s2

    print("BC pretraining:", iqm_ci(scores_1))
    print("BC+VF pretraining:", iqm_ci(scores_2))
    print("Improvement:", iqm_ci(scores_2 - scores_1))


if __name__ == "__main__":
    main()
