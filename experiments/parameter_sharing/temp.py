from experiments.distillation.repr_distill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from utils import Logger

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
    lr=10**-3, weight_decay=1e-5, epochs=50, logger=phase1_logger, verbose=True
)
phase1_logger.generate_plots("temp_pi")

trainer.vectorized_evaluate(num_episodes=200)
print({k: v[-1] for k, v in trainer.logger.cumulative_data.items()})

phase2_logger = Logger()
trainer.pretrain_value(
    lr=10**-3,
    weight_decay=1e-5,
    beta_kl=1.0,
    epochs=50,
    logger=phase2_logger,
    verbose=True,
)
phase2_logger.generate_plots("temp_vf")
trainer.vectorized_evaluate(num_episodes=200)
print({k: v[-1] for k, v in trainer.logger.cumulative_data.items()})
