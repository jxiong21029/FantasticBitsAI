import copy

import numpy as np
import torch

from env import FantasticBits
from ppo import PPOTrainer


class VectorizedPPOTrainer(PPOTrainer):
    def __init__(
        self,
        agents,
        env_kwargs=None,
        lr=1e-3,
        weight_decay=1e-4,
        rollout_steps=4096,
        gamma=0.99,
        gae_lambda=0.95,
        minibatch_size=64,
        epochs=3,
        ppo_clip_coeff=0.2,
        grad_clipping=10.0,
        entropy_reg=1e-5,
        vectorized_envs=1,
        rollout_device=torch.device("cpu"),
        train_device=torch.device("cpu"),
        seed=None,
    ):
        super().__init__(
            agents=agents,
            env_kwargs=env_kwargs,
            lr=lr,
            weight_decay=weight_decay,
            rollout_steps=rollout_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            minibatch_size=minibatch_size,
            epochs=epochs,
            ppo_clip_coeff=ppo_clip_coeff,
            grad_clipping=grad_clipping,
            entropy_reg=entropy_reg,
            rollout_device=rollout_device,
            train_device=train_device,
            seed=seed,
        )
        self.env = [copy.deepcopy(self.env) for _ in range(vectorized_envs)]
        self.env_obs = [env.reset() for env in self.env]

    def collect_rollout(self):
        with torch.no_grad():
            self.agents.to(self.rollout_device)

            actions, logps = None, None
            for t in range(self.buf.max_size):
                idx = t % len(self.env)
                if idx == 0:
                    actions, logps = self.agents.vectorized_step(self.env_obs)
                next_obs, reward, done = self.env[idx].step(actions[idx])
                self.buf.store(self.env_obs, action, reward, logp)
                self.env_obs = next_obs

                epoch_ended = t == self.buf.max_size - 1
                if epoch_ended or done:
                    if done:
                        value = (0, 0)
                    else:
                        value = self.agents.predict_value(self.env_obs, None)

                    self.buf.val_buf[
                        self.buf.path_start_idx : self.buf.ptr
                    ] = self.agents.value_forward(
                        {
                            "obs": {
                                k: torch.tensor(
                                    v[self.buf.path_start_idx : self.buf.ptr]
                                )
                                for k, v in self.buf.obs_buf.items()
                            }
                        },
                        np.arange(self.buf.ptr - self.buf.path_start_idx),
                    ).numpy()
                    self.buf.finish_path(value)
                    self.env_obs = self.env.reset()

                    self.logger.log(rollout_ep_reward=curr_ep_reward)
                    curr_ep_reward = 0

        self.rollout = self.buf.get()
