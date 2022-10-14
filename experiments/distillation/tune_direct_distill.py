import torch
from direct_distill import DirectDistillationTrainer
from ray import air, tune

from architectures import VonMisesAgents
from tuning import (
    IndependentGroupsSearch,
    grid_search,
    log_halving_search,
    q_log_halving_search,
)


def train(config):
    results = []
    for run in range(3):
        trainer = DirectDistillationTrainer(
            agents=VonMisesAgents(
                num_layers=2,
                d_model=64,
                nhead=2,
                dim_feedforward=128,
            ),
            ckpt_filename="../../../../bc_agents.pth",
            lr=config["lr"],
            gae_lambda=1 - config["1-gae_lambda"],
            minibatch_size=config["minibatch_size"],
            weight_decay=config["weight_decay"],
            epochs=config["epochs"],
            env_kwargs={
                "reward_shaping_snaffle_goal_dist": True,
                "reward_own_goal": 3.0,
                "reward_teammate_goal": 0.0,
            },
        )
        start_beta = config["start_beta"]
        end_beta = config["end_beta"]
        for i in range(201):
            trainer.beta_kl = start_beta + (end_beta - start_beta) * i / 200
            trainer.train()
            if i % 20 == 0:
                trainer.evaluate()
                trainer.logger.tune_report()

        results.append({k: v[-1] for k, v in trainer.logger.cumulative_data.items()})
        torch.save(
            trainer.agents.state_dict(),
            f"{config}_{run}.ckpt",
        )

    tune.report(
        **{
            "mo3_" + k: (results[0][k] + results[1][k] + results[2][k]) / 3
            for k in results[0].keys()
        }
    )


def main():
    import os

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    search_alg = IndependentGroupsSearch(
        search_space={
            "lr": log_halving_search(1e-4, 1e-3, 1e-2),
            "1-gae_lambda": log_halving_search(0.025, 0.1, 0.4),
            "minibatch_size": q_log_halving_search(256, 512, 1024),
            "epochs": grid_search(1, 2, 3),
            "weight_decay": log_halving_search(1e-5, 1e-4, 1e-3),
            "start_beta": 1.0,
            "end_beta": grid_search(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        },
        depth=1,
        defaults={
            "lr": 1e-3,
            "1-gae_lambda": 0.1,
            "minibatch_size": 256,
            "epochs": 2,
            "weight_decay": 1e-4,
            "start_beta": 1.0,
            "end_beta": 1.0,
        },
        groups=(
            ("1-gae_lambda",),
            ("lr", "minibatch_size"),
            ("weight_decay",),
            ("epochs",),
            ("end_beta",),
        ),
        metric="mo3_eval_goals_scored_mean",
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
            name="direct_distill",
            local_dir="../ray_results/",
        ),
    )
    tuner.fit()

    print("best score:", search_alg.best_score)
    print("best config:", search_alg.best_config)


# TODO parameterize dropout, add flip augmentation
if __name__ == "__main__":
    main()
