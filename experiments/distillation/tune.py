import torch
from distill_modeling import (
    RepresentationDistillationAgents,
    RepresentationDistillationTrainer,
)
from ray import air, tune

from tuning import IndependentGroupsSearch, log_halving_search, q_log_halving_search


def train(config):
    results = []
    for run in range(3):
        trainer = RepresentationDistillationTrainer(
            agents=RepresentationDistillationAgents(
                num_layers=2,
                d_model=64,
                nhead=2,
                dim_feedforward=128,
            ),
            demo_filename="../../../../data/basic_demo.pickle",
            lr=config["lr"],
            minibatch_size=config["minibatch_size"],
            weight_decay=config["weight_decay"],
            epochs=3,
            env_kwargs={
                "reward_shaping_snaffle_goal_dist": True,
                "reward_own_goal": 3.0,
            },
        )
        for i in range(101):
            trainer.train()
            if i % 20 == 0:
                trainer.evaluate()
                trainer.logger.tune_report()
                results.append(
                    {k: v[-1] for k, v in trainer.logger.cumulative_data.items()}
                )

        torch.save(
            trainer.agents.state_dict(),
            f"checkpoints/{config}_{run}.ckpt",
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
            "minibatch_size": q_log_halving_search(32, 128, 512),
            "weight_decay": log_halving_search(1e-5, 1e-4, 1e-3),
        },
        depth=1,
        defaults={
            "lr": 1e-4,
            "minibatch_size": 128,
            "weight_decay": 1e-5,
        },
        groups=(("lr", "minibatch_size"), ("weight_decay",)),
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
            name="repr_distill",
            local_dir="../ray_results/",
        ),
    )
    tuner.fit()

    print("best score:", search_alg.best_score)
    print("best config:", search_alg.best_config)


if __name__ == "__main__":
    main()
