import copy
import pickle

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from architectures import VonMisesAgents
from ppo import PPOTrainer
from utils import Logger, component_grad_norms


class ReDistillAgents(VonMisesAgents):
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

        with torch.no_grad():
            self.move_head_2.weight *= 0.01
            self.throw_head_2.weight *= 0.01

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
            distrs.append(
                distributions.VonMises(
                    torch.atan2(y, x), concentration, validate_args=False
                )
            )

            angles = torch.atan2(actions_taken[:, 1], actions_taken[:, 0])
            ret[:, i] = distrs[i].log_prob(angles)

        return ret, distrs


class PhasicReDistillTrainer(PPOTrainer):
    def __init__(
        self,
        agents: ReDistillAgents,
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

        self.phase2_optim = torch.optim.Adam(
            self.agents.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self):
        super().train()
        phase2_logger = Logger()

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

            self.phase2_optim.zero_grad(set_to_none=True)

            total_loss.backward()
            norms = component_grad_norms(
                self.agents, exclude=("value_encoder", "value_head")
            )

            self.phase2_optim.step()

            phase2_logger.log(
                kl_loss=kl_loss.item(),
                bc_loss=bc_loss.item(),
                total_distill_loss=total_loss.item(),
                **{"distill_grad_norm_" + k: v for k, v in norms.items()},
            )

        phase2_logger.step()
        self.logger.update_from(phase2_logger)


class JointReDistillTrainer(PPOTrainer):
    def __init__(
        self,
        agents: ReDistillAgents,
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
        beta_bc=1.0,
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
        self.beta_bc = beta_bc

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

    def train(self):
        self.collect_rollout()
        self.train()

        idx = np.arange(self.buf.max_size)
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for i in range(idx.shape[0] // self.minibatch_size):
                batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]

                logp, distrs = self.agents.policy_forward(self.rollout, batch_idx)
                ppo_loss = self.ppo_loss(batch_idx, logp, distrs)

                if self.ptr + self.minibatch_size > self.demo_sz:
                    self.rng.shuffle(self.demo_idx)
                    self.ptr = 0
                demo_idx = self.demo_idx[self.ptr : self.ptr + self.minibatch_size]
                self.ptr += self.minibatch_size

                bc_logp, _ = self.agents.policy_forward_2(self.demo, demo_idx)
                bc_loss = -bc_logp.mean()

                total_loss = ppo_loss + self.beta_bc * bc_loss
                self.optim.zero_grad(set_to_none=True)
                total_loss.backward()

                norms = component_grad_norms(self.agents)
                self.logger.log(
                    bc_loss=bc_loss.item(),
                    total_loss=total_loss.item(),
                    **{"grad_norm_" + k: v for k, v in norms.items()},
                )
                if self.grad_clipping is not None:
                    nn.utils.clip_grad_norm_(
                        self.agents.parameters(), self.grad_clipping
                    )
                    self.logger.log(
                        grad_clipped=(norms["total"] > self.grad_clipping)
                        if self.grad_clipping is not None
                        else False,
                    )
                self.optim.step()

        self.logger.step()


def main():
    import tqdm

    trainer = JointReDistillTrainer(
        agents=ReDistillAgents(
            num_layers=2,
            d_model=64,
            nhead=2,
            dim_feedforward=128,
        ),
        env_kwargs={
            "reward_shaping_snaffle_goal_dist": True,
            "reward_own_goal": 3.0,
        },
        demo_filename="../../data/basic_demo.pickle",
        lr=10**-3,
        gae_lambda=0.5,
        minibatch_size=256,
        weight_decay=10**-3.5,
        epochs=2,
        beta_bc=1e-1,
    )
    for i in tqdm.trange(201):
        trainer.train()
        if i % 20 == 0:
            trainer.evaluate()
            trainer.logger.generate_plots()


if __name__ == "__main__":
    main()
