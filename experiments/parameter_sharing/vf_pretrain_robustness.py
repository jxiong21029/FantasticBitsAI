from ray import air, tune

from experiments.distillation.repr_distill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from tuning import IntervalHalvingSearch, log_halving_search
from utils import Logger


def train(config):
    agents = ReDistillAgents(
        num_layers=2, d_model=64, nhead=2, dim_feedforward=128, share_parameters=True
    )

    trainer = JointReDistillTrainer(
        agents,
        demo_filename="../../data/basic_demo.pickle",
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
        verbose=True,
    )
    trainer.vectorized_evaluate(200)
    phase1_logger.tune_report()


def main():
    searcher = IntervalHalvingSearch(
        search_space={
            "lr_bc": log_halving_search(1e-4, 1e-3, 1e-2),
            "l2_bc": log_halving_search(1e-5, 1e-4, 1e-3),
        },
        depth=1,
        metric="eval_goals_scored_mean",
        mode="max",
    )
    tuner = tune.Tuner(
        train,
        tune_config=tune.TuneConfig(
            search_alg=searcher,
            max_concurrent_trials=8,
        ),
        run_config=air.RunConfig(
            name="tune_pretrain",
            local_dir="../ray_results",
        ),
    )
    tuner.fit()

    print(searcher.best_config)
    print(searcher.best_score)


if __name__ == "__main__":
    main()
