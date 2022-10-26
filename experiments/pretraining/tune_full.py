import os
from collections import defaultdict

import numpy as np
from ray import air, tune

from experiments.distillation.redistill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from tuning import IntervalHalvingSearch, log_halving_search
from utils import Logger


def run_trial(config):
    dirs = os.getcwd().split("/")
    root_dir = os.path.join(*dirs[: 1 + dirs.index("FantasticBits")])

    res = defaultdict(list)
    for i in range(5):
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
            lr=config["lr_bc"],
            weight_decay=config["l2_bc"],
            epochs=50,
            logger=phase1_logger,
        )
        trainer.vectorized_evaluate(200)
        res["jc_bc"].append(phase1_logger.estimate_convergence("pretrain_loss_bc_mean"))
        res["score_bc"].append(
            trainer.logger.cumulative_data["eval_goals_scored_mean"][-1]
        )

        if config["train_vf"]:
            phase2_logger = Logger()
            trainer.pretrain_value(
                lr=config["lr_vf"],
                weight_decay=config["l2_vf"],
                beta_kl=config["kl_vf"],
                epochs=50,
                logger=phase2_logger,
            )
            trainer.vectorized_evaluate(200)
            res["jc_vf"].append(
                phase2_logger.estimate_convergence("pretrain_loss_total_mean")
            )
            res["score_vf"].append(
                trainer.logger.cumulative_data["eval_goals_scored_mean"][-1]
            )

    ret = {}
    for k in res:
        ret[k] = np.mean(sorted(res[k])[1:4])
    tune.report(**ret)


def main():
    bc_searcher = IntervalHalvingSearch(
        search_space={
            "lr_bc": log_halving_search(1e-4, 1e-3, 1e-2),
            "l2_bc": log_halving_search(1e-4, 1e-3, 1e-2),
            "train_vf": False,
        },
        depth=1,
        metric="score_bc",
        mode="max",
    )
    bc_tuner = tune.Tuner(
        run_trial,
        tune_config=tune.TuneConfig(
            search_alg=bc_searcher,
            num_samples=-1,
            max_concurrent_trials=8,
        ),
        run_config=air.RunConfig(
            name="vf_finetuning_part1",
            local_dir="ray_results",
            verbose=0,
        ),
    )
    bc_tuner.fit()

    vf_searcher = IntervalHalvingSearch(
        search_space={
            "lr_bc": bc_searcher.best_config[1]["lr_bc"],
            "l2_bc": bc_searcher.best_config[1]["l2_bc"],
            "train_vf": True,
            "lr_vf": log_halving_search(1e-4, 1e-3, 1e-2),
            "l2_vf": log_halving_search(1e-4, 1e-3, 1e-2),
            "kl_vf": log_halving_search(1e-1, 1e0, 1e1),
        },
        depth=1,
        metric="score_vf",
        mode="max",
    )
    vf_tuner = tune.Tuner(
        run_trial,
        tune_config=tune.TuneConfig(
            search_alg=vf_searcher,
            num_samples=-1,
            max_concurrent_trials=8,
        ),
        run_config=air.RunConfig(
            name="vf_finetuning_part2",
            local_dir="ray_results",
            verbose=0,
        ),
    )
    vf_tuner.fit()


if __name__ == "__main__":
    main()
