from ray import air, tune

from experiments.distillation.redistill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from tuning import IntervalHalvingSearch, log_halving_search


def train(config):
    agents = ReDistillAgents(
        num_layers=2, d_model=64, nhead=2, dim_feedforward=128, share_parameters=True
    )

    trainer = JointReDistillTrainer(
        agents,
        demo_filename="../../../../data/basic_demo.pickle",
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

    trainer.pretrain_policy(
        lr=10**-2.5, weight_decay=10**-2.5, epochs=50, logger=trainer.logger
    )

    trainer.pretrain_value(
        lr=config["lr_vf"],
        weight_decay=config["l2_vf"],
        beta_kl=config["beta_kl"],
        epochs=50,
        logger=trainer.logger,
    )
    trainer.vectorized_evaluate(200)

    trainer.logger.tune_report()


def main():
    searcher = IntervalHalvingSearch(
        search_space={
            "lr_vf": log_halving_search(1e-4, 1e-3, 1e-2),
            "l2_vf": log_halving_search(1e-5, 1e-4, 1e-3),
            "beta_kl": log_halving_search(1e-2, 1e0, 1e2),
        },
        depth=1,
        metric="eval_goals_scored_mean",
        mode="max",
    )
    tuner = tune.Tuner(
        train,
        tune_config=tune.TuneConfig(
            search_alg=searcher,
            num_samples=-1,
            max_concurrent_trials=8,
        ),
        run_config=air.RunConfig(
            name="tune_pretrain_vf",
            local_dir="../ray_results",
        ),
    )
    tuner.fit()

    print(searcher.best_config)
    print(searcher.best_score)


if __name__ == "__main__":
    main()
