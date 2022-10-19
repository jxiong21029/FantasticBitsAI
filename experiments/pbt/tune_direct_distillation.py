import copy
import os
import random

import torch
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.sample import Domain

from architectures import VonMisesAgents
from experiments.distillation.direct_distill import DirectDistillationTrainer
from ppo import PPOConfig


def train(config):
    agents = VonMisesAgents(num_layers=2, d_model=64, nhead=2, dim_feedforward=128)
    trainer = DirectDistillationTrainer(
        agents,
        ckpt_filename="../../../../bc_agents.pth",
        beta_kl=config["beta_kl"],
        ppo_config=PPOConfig(
            lr=config["lr"],
            gae_lambda=0.975,
            minibatch_size=config["minibatch_size"],
            weight_decay=config["weight_decay"],
            epochs=config["epochs"],
            env_kwargs={
                "reward_shaping_snaffle_goal_dist": True,
                "reward_own_goal": 3.0,
                "reward_teammate_goal": 0.0,
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

    while True:
        trainer.train_epoch()
        if step % 25 == 0:
            trainer.evaluate()

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
            trainer.logger.air_report(checkpoint=new_checkpoint)
        step += 1


def main():
    param_space = {
        "lr": tune.loguniform(10**-4, 10**-2.5),
        "minibatch_size": [256, 512, 1024],
        "weight_decay": tune.loguniform(10**-6, 10**-3),
        "epochs": [1, 2, 3],
        "beta_kl": tune.loguniform(10**-2.5, 1),
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
        perturbation_interval=2,
        hyperparam_mutations=param_space,
    )
    pbt._get_new_config = custom_get_new_config

    tuner = tune.Tuner(
        train,
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=-1,
            max_concurrent_trials=8,
            time_budget_s=3600 * 12,
        ),
        run_config=air.RunConfig(
            name="direct_distill_pbt",
            local_dir="../ray_results",
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
