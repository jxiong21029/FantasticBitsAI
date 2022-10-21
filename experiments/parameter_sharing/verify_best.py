import os
import random
from collections import defaultdict

import numpy as np
import tqdm

from experiments.distillation.repr_distill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from utils import Logger

dirs = os.getcwd().split("/")
root_dir = os.path.join(*dirs[: 1 + dirs.index("FantasticBits")])

res = defaultdict(list)
for i in tqdm.trange(20):
    print("starting iteration", i)
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
        epochs=50,
        logger=phase1_logger,
    )
    trainer.vectorized_evaluate(200)

    p1c = phase1_logger.estimate_convergence("pretrain_loss_bc_mean")
    print("phase 1 convergence:", p1c)
    p1g = trainer.logger.cumulative_data["eval_goals_scored_mean"][-1]
    print("phase 1 goals:", p1g)

    phase2_logger = Logger()
    trainer.pretrain_value(
        lr=1e-4,
        weight_decay=1e-3,
        beta_kl=10,
        epochs=50,
        logger=phase2_logger,
    )
    trainer.vectorized_evaluate(200)

    p2c = phase2_logger.estimate_convergence("pretrain_loss_total_mean")
    print(
        "phase 2 convergence:",
        p2c,
    )
    p2g = trainer.logger.cumulative_data["eval_goals_scored_mean"][-1]
    print("phase 2 goals:", p2g)

    res["phase1_score"].append(p1g)
    res["phase2_score"].append(p2g)


def iqm_ci(dataset):
    results = []
    for _ in range(1000):
        data = np.array(random.choices(dataset, k=len(dataset)))
        lower_bound = np.percentile(data, 25, method="lower")
        upper_bound = np.percentile(data, 75, method="higher")

        results.append(np.mean(data[(lower_bound <= data) & (data <= upper_bound)]))

    return (
        np.nanpercentile(results, 2.5, axis=0),
        np.nanmean(results, axis=0),
        np.nanpercentile(results, 97.5, axis=0),
    )


for k, v in res:
    print(k, iqm_ci(v))
