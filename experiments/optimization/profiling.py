import time

import torch

from architectures import VonMisesAgents
from ppo import PPOTrainer
from utils import profileit

# for rollout_device, train_device in (
#     (torch.device("cpu"), torch.device("cpu")),
#     (torch.device("cpu"), torch.device("cuda:0")),
#     # (torch.device("cuda:0"), torch.device("cuda:0")),
# ):
#     print(f"{rollout_device=}, {train_device=}")
#     trainer = PPOTrainer(
#         VonMisesAgents(
#             num_layers=2,
#             d_model=64,
#             dim_feedforward=128
#         ),
#         rollout_steps=4096,
#         minibatch_size=512,
#         rollout_device=rollout_device,
#         train_device=train_device,
#     )
#     start = time.perf_counter()
#     for _ in range(2):
#         trainer.train()
#     print(f"time taken (train, 2 iters): {time.perf_counter() - start}")
#
#     start = time.perf_counter()
#     trainer.evaluate()
#     print(f"time taken (eval, 50 ep): {time.perf_counter() - start}")


@profileit
def train():
    trainer = PPOTrainer(
        VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128),
        rollout_steps=4096,
        minibatch_size=512,
        env_kwargs={"reward_shaping_snaffle_goal_dist": True},
    )
    trainer.train()
    trainer.train()


train()
