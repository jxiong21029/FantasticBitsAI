from abc import ABC, abstractmethod

import numpy as np
import torch

from env import FantasticBits
from utils import Logger


class Trainer(ABC):
    def __init__(self, env_kwargs=None, seed=None):
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.logger = Logger()
        self.rng = np.random.default_rng(seed=seed)

    @property
    @abstractmethod
    def agents(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def custom_loss(self, loss, rollout):
        return loss

    def evaluate(self, num_episodes=50):
        self.agents.eval()

        temp_logger = Logger()
        eval_env = FantasticBits(**self.env_kwargs, logger=temp_logger)
        for _ in range(num_episodes):
            obs = eval_env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    actions, _ = self.agents.step(obs)
                obs, _, done = eval_env.step(actions)
        temp_logger.step()
        self.logger.cumulative_data.update(
            {"eval_" + k: v for k, v in temp_logger.cumulative_data.items()}
        )

        self.agents.train()

    def evaluate_with_render(self):
        import time

        self.agents.eval()

        done = False
        eval_env = FantasticBits(**self.env_kwargs, render=True, logger=self.logger)
        obs = eval_env.reset()
        tot_rew = np.zeros(2)
        while not done:
            with torch.no_grad():
                actions, _ = self.agents.step(obs)
            obs, rew, done = eval_env.step(actions)
            tot_rew += rew
            time.sleep(0.1)
        print(f"total reward: {tot_rew.tolist()}")
        print(f"final score: {eval_env.score[0]} - {eval_env.score[1]}")

        self.agents.train()
