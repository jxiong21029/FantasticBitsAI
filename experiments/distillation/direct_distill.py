import copy

import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from architectures import VonMisesAgents
from ppo import PPOConfig, PPOTrainer
from utils import component_grad_norms


class DirectDistillationTrainer(PPOTrainer):
    def __init__(
        self,
        agents,
        ppo_config: PPOConfig,
        ckpt_filename,
        beta_kl=1.0,
    ):
        super().__init__(agents, ppo_config)
        self.beta_kl = beta_kl

        self.frozen_agents = copy.deepcopy(self.agents)
        self.frozen_agents.load_state_dict(torch.load(ckpt_filename))

    def run(self):
        self.collect_rollout()
        self.agents.train()

        idx = np.arange(self.rollout_steps)

        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for i in range(idx.shape[0] // self.minibatch_size):
                batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]

                logp, pi = self.agents.policy_forward(self.rollout, batch_idx)
                loss_ppo = self.ppo_loss(batch_idx, logp, pi)

                with torch.no_grad():
                    _, frozen_pi = self.frozen_agents.policy_forward(
                        self.rollout, batch_idx
                    )
                loss_kl = self.beta_kl * sum(
                    distributions.kl_divergence(frozen_pi[i], pi[i]).mean()
                    for i in range(2)
                )
                self.logger.log(loss_kl=loss_kl.item())

                total_loss = loss_ppo + loss_kl
                self.optim.zero_grad(set_to_none=True)
                total_loss.backward()

                norms = component_grad_norms(self.agents)

                self.logger.log(**{"grad_norm_" + k: v for k, v in norms.items()})
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

    trainer = DirectDistillationTrainer(
        agents=VonMisesAgents(
            num_layers=2,
            d_model=64,
            nhead=2,
            dim_feedforward=128,
            dropout=0,
        ),
        ckpt_filename="../../bc_agents.pth",
        beta_kl=0.1,
        ppo_config=PPOConfig(
            lr=10**-3.5,
            minibatch_size=512,
            weight_decay=10**-3,
            gae_lambda=0.975,
            epochs=2,
            env_kwargs={
                "reward_shaping_snaffle_goal_dist": True,
                "reward_own_goal": 3.0,
            },
        ),
    )
    for i in tqdm.trange(501):
        trainer.run()
        if i % 20 == 0:
            trainer.evaluate()
            trainer.logger.generate_plots("plotgen_direct")
    torch.save(trainer.agents.state_dict(), "direct_distill_agents.pth")


if __name__ == "__main__":
    main()
