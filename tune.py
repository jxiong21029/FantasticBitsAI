from ray import air, tune

from agents import Agents
from env import FantasticBits
from ppo import Trainer


# TODO: implement evaluation function which gets a reliable performance estimate
# TODO: write a tuner class which automatically does binary log search, extracts
#  best mean policy out of three runs at each stage, and checkpoints the best agent
def train(config):
    trainer = Trainer(
        Agents(),
        FantasticBits,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        env_kwargs={"shape_snaffle_dist": True},
    )

    for _ in range(100):
        trainer.train()
        tune.report(**{k: v[-1] for k, v in trainer.logger.cumulative_data.items()})


if __name__ == "__main__":
    tuner = tune.Tuner(
        train,
        param_space={
            "lr": tune.grid_search([1e-4, 1e-3, 1e-2]),
            "weight_decay": tune.grid_search([1e-5, 1e-4, 1e-3]),
        },
        tune_config=tune.TuneConfig(
            num_samples=3,
            metric="goals_scored_mean",
            mode="max",
        ),
        run_config=air.RunConfig(
            name="tune_lr_l2",
            local_dir="ray_results/",
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result()
    print(best_result.config, best_result.metrics)
