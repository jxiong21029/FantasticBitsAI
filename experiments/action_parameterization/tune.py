from ray import air, tune
from von_mises_agents import VonMisesAgents

from architectures import Agents
from behavioral_cloning import BCTrainer
from tuning import IntervalHalvingSearch, log_halving_search


def train(config):
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
        if i % 20 == 0:
            trainer.evaluate()
            tune.report(**{k: v[-1] for k, v in trainer.logger.cumulative_data.items()})


def main():
    for action_parameterization in ("euclidean", "normed_euclidean", "von_mises"):
        search_alg = IntervalHalvingSearch(
            search_space={
                "action_parameterization": action_parameterization,
                "lr": log_halving_search(1e-4, 1e-3, 1e-2),
                "minibatch_size": 128,
                "weight_decay": 1e-5,
                "num_layers": 1,
                "d_model": 32,
                "nhead": 2,
            },
            depth=1,
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
                name=f"action_param2_{action_parameterization}",
                local_dir="../ray_results/",
            ),
        )
        tuner.fit()

        print(f"results for {action_parameterization=}")
        print("best score:", search_alg.best_score)
        print("best config:", search_alg.best_config)


if __name__ == "__main__":
    main()
