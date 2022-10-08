from ray import air, tune
from von_mises_agents import VonMisesAgents

from architectures import Agents
from behavioral_cloning import BCTrainer
from tuning import IntervalHalvingSearch, log_halving_search


def train(config):
    results = []
    for _ in range(3):
        if config["action_parameterization"].endswith("normed_euclidean"):
            trainer = BCTrainer(
                Agents(
                    num_layers=config["num_layers"],
                    d_model=config["d_model"],
                    nhead=config["nhead"],
                    norm_action_mean=(
                        config["action_parameterization"] == "normed_euclidean"
                    ),
                ),
                demo_filename="../../../../data/basic_demo.pickle",
                lr=config["lr"],
                minibatch_size=config["minibatch_size"],
                weight_decay=config["weight_decay"],
                grad_clipping=None,
            )
        else:
            trainer = BCTrainer(
                VonMisesAgents(
                    num_layers=config["num_layers"],
                    d_model=config["d_model"],
                    nhead=config["nhead"],
                ),
                demo_filename="../../../../data/basic_demo.pickle",
                lr=config["lr"],
                minibatch_size=config["minibatch_size"],
                weight_decay=config["weight_decay"],
                grad_clipping=None,
            )
        for i in range(101):
            trainer.train()
            # if i % 20 == 0:
            #     trainer.evaluate()
            #     tune.report(
            #         **{k: v[-1] for k, v in trainer.logger.cumulative_data.items()}
            #     )
        trainer.evaluate()
        results.append({k: v[-1] for k, v in trainer.logger.cumulative_data.items()})
    tune.report(
        **{
            "mo3_" + k: (results[0][k] + results[1][k] + results[2][k]) / 3
            for k in results[0].keys()
        }
    )


def main():
    import os

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    for action_parameterization in ("von_mises", "euclidean", "normed_euclidean"):
        search_alg = IntervalHalvingSearch(
            search_space={
                "action_parameterization": action_parameterization,
                "lr": log_halving_search(10**-5, 10**-4, 10**-3),
                "minibatch_size": 256,
                "weight_decay": log_halving_search(1e-6, 1e-5, 1e-4),
                "num_layers": 2,
                "d_model": 64,
                "nhead": 2,
            },
            depth=1,
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
                name=f"action_param4_{action_parameterization}",
                local_dir="../ray_results/",
            ),
        )
        tuner.fit()

        print(f"results for {action_parameterization=}")
        print("best score:", search_alg.best_score)
        print("best config:", search_alg.best_config)


if __name__ == "__main__":
    main()
