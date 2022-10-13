import copy

import numpy as np
import torch
from torch import distributions

from architectures import VonMisesAgents
from ppo import PPOTrainer
from utils import component_grad_norms


class DirectDistillationTrainer(PPOTrainer):
    def __init__(
        self,
        agents: VonMisesAgents,
        ckpt_filename: str,
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

        self.frozen_agents = copy.deepcopy(agents)
        self.frozen_agents.load_state_dict(torch.load(ckpt_filename))

    def train(self):
        self.collect_rollout()

        idx = np.arange(self.buf.max_size)

        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for i in range(idx.shape[0] // self.minibatch_size):
                batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]

                logp, pi = self.agents.policy_forward(self.rollout, batch_idx)
                total_loss = self.ppo_loss(batch_idx, logp, pi)

                with torch.no_grad():
                    _, frozen_pi = self.frozen_agents.policy_forward(
                        self.rollout, batch_idx
                    )
                total_loss += self.beta_kl * sum(
                    distributions.kl_divergence(frozen_pi[i], pi[i]).mean()
                    for i in range(2)
                )

                self.optim.zero_grad(set_to_none=True)
                total_loss.backward()

                norms = component_grad_norms(self.agents)

                self.logger.log(**{"grad_norm_" + k: v for k, v in norms.items()})
                if self.grad_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
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

    trainer = DirectDistillationTrainer(
        agents=VonMisesAgents(
            num_layers=2,
            d_model=64,
            nhead=2,
            dim_feedforward=128,
        ),
        ckpt_filename="../../bc_agents.pth",
        lr=10**-3,
        minibatch_size=256,
        weight_decay=10**-3.5,
        epochs=2,
        env_kwargs={
            "reward_shaping_snaffle_goal_dist": True,
            "reward_own_goal": 3.0,
        },
    )
    for i in tqdm.trange(201):
        trainer.train()
        if i % 20 == 0:
            trainer.evaluate()
            trainer.logger.generate_plots("plotgen_direct")


if __name__ == "__main__":
    main()
