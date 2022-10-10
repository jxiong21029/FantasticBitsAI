import pickle

import torch
import torch.nn as nn

from ppo import PPOTrainer

from ..action_parameterization.von_mises_agents import VonMisesAgents


class RepresentationDistillationAgents(VonMisesAgents):
    def __init__(
        self,
        num_layers=1,
        d_model=32,
        nhead=2,
        dim_feedforward=64,
    ):
        super().__init__(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.move_head_2 = nn.Linear(d_model, 3)
        self.throw_head_2 = nn.Linear(d_model, 3)


class RepresentationDistillationTrainer(PPOTrainer):
    def __init__(
        self,
        agents,
        demo_filename,
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
            seed=seed,
        )

        with open(demo_filename, "rb") as f:
            demo_obs, demo_actions = pickle.load(f)
        self.rollout = {
            "obs": {k: torch.tensor(v) for k, v in demo_obs.items()},
            "act": {k: torch.tensor(v) for k, v in demo_actions.items()},
        }
        for i in range(2):
            self.rollout["act"]["target"] /= torch.norm(
                self.rollout["act"]["target"], dim=2, keepdim=True
            )
        self.demo_sz = demo_obs["global"].shape[0]

    def custom_loss(self, loss, rollout):
        pass
