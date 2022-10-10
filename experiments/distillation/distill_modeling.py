import copy
import pickle

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from experiments.action_parameterization.von_mises_agents import VonMisesAgents
from ppo import PPOTrainer
from utils import Logger, grad_norm


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

    def policy_forward_2(self, rollout, batch_idx):
        z = self.policy_encoder(rollout["obs"], batch_idx)  # S x B x 32
        distrs = []
        ret = torch.zeros((batch_idx.shape[0], 2))
        for i in range(2):
            embed = z[i + 1]
            logits = torch.zeros((len(batch_idx), 3))

            throw_turns = rollout["obs"][f"wizard{i}"][batch_idx, 5] == 1
            logits[throw_turns] = self.throw_head_2(embed[throw_turns])
            logits[~throw_turns] = self.move_head_2(embed[~throw_turns])

            actions_taken = rollout["act"]["target"][batch_idx, i]

            x = logits[:, 0]
            y = logits[:, 1]
            concentration = F.softplus(logits[:, 2]) + 1e-3
            distrs.append(distributions.VonMises(torch.atan2(y, x), concentration))

            angles = torch.atan2(actions_taken[:, 1], actions_taken[:, 0])
            ret[:, i] = distrs[i].log_prob(angles)

        return ret, distrs


class RepresentationDistillationTrainer(PPOTrainer):
    def __init__(
        self,
        agents: RepresentationDistillationAgents,
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
        beta_kl=1.0,
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
        self.beta_kl = beta_kl

        with open(demo_filename, "rb") as f:
            demo_obs, demo_actions = pickle.load(f)
        self.demo = {
            "obs": {k: torch.tensor(v) for k, v in demo_obs.items()},
            "act": {k: torch.tensor(v) for k, v in demo_actions.items()},
        }
        for i in range(2):
            self.demo["act"]["target"] /= torch.norm(
                self.demo["act"]["target"], dim=2, keepdim=True
            )
        self.demo_sz = demo_obs["global"].shape[0]
        self.demo_idx = np.arange(self.demo_sz)
        self.rng.shuffle(self.demo_idx)
        self.ptr = 0

        self.phase2_logger = Logger()
        self.phase2_optim = torch.optim.Adam(
            self.agents.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self):
        super().train()

        frozen_agents = copy.deepcopy(self.agents)

        idx = np.arange(self.buf.max_size)
        self.rng.shuffle(idx)

        for i in range(idx.shape[0] // self.minibatch_size):
            batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]

            _, distrs = self.agents.policy_forward(self.rollout, batch_idx)
            with torch.no_grad():
                _, frozen_distrs = frozen_agents.policy_forward(self.rollout, batch_idx)

            kl_loss = sum(
                distributions.kl_divergence(frozen_distrs[j], distrs[j]).mean()
                for j in range(2)
            )

            if self.ptr + self.minibatch_size > self.demo_sz:
                self.rng.shuffle(self.demo_idx)
                self.ptr = 0
            demo_idx = self.demo_idx[self.ptr : self.ptr + self.minibatch_size]
            self.ptr += self.minibatch_size

            bc_logp, _ = self.agents.policy_forward_2(self.demo, demo_idx)
            bc_loss = -bc_logp.mean()

            total_loss = bc_loss + self.beta_kl * kl_loss

            self.phase2_optim.zero_grad()
            total_loss.backward()
            with torch.no_grad():
                norm = grad_norm(self.agents)
            self.phase2_optim.step()

            self.phase2_logger.log(
                kl_loss=kl_loss.item(),
                bc_loss=bc_loss.item(),
                total_distill_loss=total_loss.item(),
                distill_grad_norm=norm,
            )

        self.phase2_logger.step()
        self.logger.update_from(self.phase2_logger)


def main():
    import tqdm

    trainer = RepresentationDistillationTrainer(
        agents=RepresentationDistillationAgents(
            num_layers=2,
            d_model=64,
            nhead=2,
            dim_feedforward=128,
        ),
        demo_filename="../../data/basic_demo.pickle",
        lr=10**-3.5,
        weight_decay=10**-4.5,
        epochs=2,
        env_kwargs={
            "reward_shaping_snaffle_goal_dist": True,
            "reward_own_goal": 3.0,
        },
    )
    for _ in tqdm.trange(20):
        trainer.train()
    trainer.evaluate_with_render()


if __name__ == "__main__":
    main()