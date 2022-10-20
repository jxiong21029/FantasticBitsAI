import torch

from experiments.distillation.repr_distill import JointReDistillTrainer, ReDistillAgents
from ppo import PPOConfig

agents = ReDistillAgents(
    num_layers=2, d_model=64, nhead=2, dim_feedforward=128, share_parameters=True
)

trainer = JointReDistillTrainer(
    agents,
    demo_filename="../../data/basic_demo.pickle",
    beta_bc=0.3,
    ppo_config=PPOConfig(
        env_kwargs={
            "reward_shaping_snaffle_goal_dist": True,
            "reward_own_goal": 2,
            "reward_teammate_goal": 2,
            "reward_opponent_goal": -2,
        },
    ),
)

trainer.pretrain_policy(lr=10**-2.5, weight_decay=10**-2.5, verbose=True)
# TODO: replace placeholders
trainer.pretrain_value(lr=-1, weight_decay=-1, beta_kl=-1, verbose=True)

torch.save(
    {
        "agents": trainer.agents.state_dict(),
        "optim": trainer.optim.state_dict(),
    },
    "../../data/pretrained_agents.pth",
)
