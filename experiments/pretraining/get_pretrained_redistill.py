import torch

from experiments.distillation.redistill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig
from utils import Logger

agents = ReDistillAgents(
    num_layers=2, d_model=64, nhead=2, dim_feedforward=128, share_parameters=True
)

trainer = JointReDistillTrainer(
    agents,
    demo_filename="../../data/basic_demo.pickle",
    ppo_config=PPOConfig(
        env_kwargs={
            "reward_shaping_snaffle_goal_dist": True,
            "reward_own_goal": 2,
            "reward_teammate_goal": 2,
            "reward_opponent_goal": -2,
        },
    ),
)

pi_logger = Logger()
trainer.pretrain_policy(
    lr=10**-3, weight_decay=10**-4, logger=pi_logger, verbose=True
)
print("BC convergence:", pi_logger.estimate_convergence("pretrain_loss_bc_mean"))

vf_logger = Logger()
trainer.pretrain_value(
    lr=10**-3, weight_decay=10**-3, beta_kl=1, logger=vf_logger, verbose=True
)
print("VF convergence:", vf_logger.estimate_convergence("pretrain_loss_vf_mean"))
print("VF+KL convergence:", vf_logger.estimate_convergence("pretrain_loss_total_mean"))

torch.save(
    trainer.agents.state_dict(),
    "../../data/pretrained.pth",
)
