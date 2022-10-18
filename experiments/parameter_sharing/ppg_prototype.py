import torch

from ppo import PPOTrainer


class ReplayBuffer:
    def __init__(
        self, size, gamma=0.99, lam=0.95, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.device = device


class PPGTrainer(PPOTrainer):
    pass
