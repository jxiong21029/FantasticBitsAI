import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from env import FantasticBits


class Rollout:
    def __init__(
        self,
        
    ):


def deep_idx(obj, idx, copy=False):
    # copy may be useful if memory needs to be freed
    if isinstance(obj, dict):
        return {k: deep_idx(v, idx, copy) for k, v in obj.items()}
    else:
        try:
            if copy:
                return obj[idx].copy()
            return obj[idx]
        except TypeError:
            warnings.warn(f"cant index object of type {type(obj)}: {obj}")
            return obj


class Agents(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.global_prep = nn.Linear(4, d_model)
        self.entity_prep = nn.Linear(9, d_model)
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=nhead, dim_feedforward=128, dropout=0
            ),
            num_layers=2,
        )
        self.move_head = nn.Linear(d_model, 4)
        self.throw_head = nn.Linear(d_model, 4)

        self._std_offset = torch.log(torch.exp(torch.tensor(0.5)) - 1)

    def rollout_forward(self, obs):
        xg = torch.stack(
            [
                torch.tensor(obs["wizard_0"]["global"]),  # 4,
                torch.tensor(obs["wizard_1"]["global"]),  # 4,
            ]
        )  # 2 x 4
        xg = self.global_prep(xg).unsqueeze(0)  # 1 x 2 x 32

        xe0 = obs["wizard_0"]["entities"]
        xe0 = torch.stack([torch.tensor(entry) for entry in xe0])  # S x 9
        xe1 = obs["wizard_1"]["entities"]
        xe1 = torch.stack([torch.tensor(entry) for entry in xe1])  # S x 9
        xe = torch.stack([xe0, xe1], dim=1)  # S x 2 x 9
        xe = self.entity_prep(xe)  # S x 2 x 32

        xc = torch.cat([xg, xe], dim=0)  # (S+1) x 2 x 32
        z = self.backbone(xc)[0]  # 2 x 32

        ret = {}, 0
        for i in range(2):
            if obs[f"wizard_{i}"]["global"][3] == 1:  # holding snaffle
                logits = self.throw_head(z[i])
            else:
                logits = self.move_head(z[i])
            mu = logits[:2]
            sigma = F.softplus(logits[2:] + self._std_offset.to(device=logits.device))
            distr = distributions.Normal(mu, sigma)

            transforms = [
                distributions.SigmoidTransform(),
                distributions.AffineTransform(-1, 1),
            ]
            distr = distributions.TransformedDistribution(
                distr, transforms, validate_args=False
            )

            action = distr.sample()
            logp = distr.log_prob(action)
            if obs[f"wizard_{i}"]["global"][3] == 1:  # holding snaffle
                ret[0][f"wizard_{i}"] = {"throw": action}
            else:
                ret[0][f"wizard_{i}"] = {"move": action}
            ret[1] += logp
        return ret

    def train_forward(self, rollout, batch_idx) -> dict[str, torch.tensor]:
        for i in range(2):
            obs = rollout["obs"][f"wizard_{i}"]
            # "global": N x 3
            # "mask": N x T
            # "entities": T x N x 9, T = max(seq len)

            xg = self.global_prep(obs["global"][batch_idx])

    def predict_values(self, rollout, batch_idx=None):
        pass


class MARLTrainer:
    def __init__(
        self,
        gamma=0.99,
        gae_lambda=0.9,
        rollout_steps=4096,
        num_epochs=3,
        minibatch_size=64,
        recompute_advantages=True,
        gradient_clipping=5.0,
        device=torch.device("cpu"),
    ):
        self.env = FantasticBits()
        self.env_obs = self.env.reset()
        self.env_terminal = False

        self.agent_ids = self.env.observation_spaces.keys()
        self.agent_group = Agents()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.recompute_advantages = recompute_advantages
        self.gradient_clipping = gradient_clipping

        self.device = device

        self.metrics = {}
        self.iteration = 0

    def collect_rollout(self) -> dict:
        rollout_entries = []
        num_steps = 0

        while num_steps < self.rollout_steps or not self.env_terminal:
            num_steps += 1
            if self.env_terminal:
                self.env_obs = self.env.reset()
                self.env_terminal = False

            with torch.no_grad():
                actions, action_logp = self.agent_group.rollout_forward(self.env_obs)

            next_obs, env_rewards, self.env_terminal, _ = self.env.step(actions)

            rollout_entries.append(
                {
                    "obs": self.env_obs,
                    "actions": actions,
                    "action_logp": action_logp,
                    "rewards": env_rewards,
                    "terminal": self.env_terminal,
                }
            )
            self.env_obs = next_obs

        rollout = stack_dicts(rollout_entries)

        max_len = max(
            len(agent_obs["entities"]) for agent_obs in rollout["obs"].values()
        )
        rollout["obs"] = {
            {
                "global": agent_obs["global"],
                "masks": torch.stack(
                    [
                        torch.cat([
                            torch.ones(len(entry), dtype=torch.uint8),
                            torch.zeros(max_len - len(entry), dtype=torch.uint8),
                        ])
                        for entry in agent_obs["entities"]
                    ],
                    dim=0,
                ),
                "entities": pad_sequence(agent_obs["entities"]),
            }
            for agent_id, agent_obs in rollout["obs"].items()
        }

        rollout["num_steps"] = num_steps

        # Avoids training on terminal states, but those states can still be used for
        # value bootstrapping
        # TODO: The empirical impacts of this change are untested.
        for i, step in enumerate(rollout["segment_ends"]):
            if rollout["terminal"][i]:
                rollout["segment_ends"][i] = step - 1

        if not self.recompute_advantages:
            self.compute_advantages_and_returns(rollout)

        self.metrics["num_steps"] = num_steps

        return rollout

    def compute_advantages_and_returns(self, rollout):
        with torch.no_grad():
            value_preds = {
                agent_id: value_estimate.cpu().numpy()
                for agent_id, value_estimate in self.agent_group.predict_values(
                    rollout["obs"], rollout["actions"]
                ).items()
            }

        advantages = {
            agent_id: np.zeros(rollout["num_steps"]) for agent_id in self.agent_ids
        }
        for agent_id in self.agent_ids:
            last_gae_lambda = 0
            for step in reversed(range(rollout["num_steps"])):
                if step == rollout["num_steps"] - 1:
                    next_value = 0
                else:
                    next_value = value_preds[agent_id][step + 1]

                non_terminal = 1 - rollout["terminal"][step]

                delta = (
                    rollout["rewards"][agent_id][step]
                    + self.gamma * next_value * non_terminal
                    - value_preds[agent_id][step]
                )
                last_gae_lambda = delta + (
                    self.gamma * self.gae_lambda * non_terminal * last_gae_lambda
                )
                advantages[agent_id][step] = last_gae_lambda

        rollout["advantages"] = advantages
        rollout["value_targets"] = {
            agent_id: advantages[agent_id] + value_preds[agent_id]
            for agent_id in self.agent_ids
        }

        explained_variances = {
            agent_id: 1
            - np.var(rollout["value_targets"][agent_id] - value_preds[agent_id])
            / np.var(rollout["value_targets"][agent_id])
            for agent_id in self.agent_ids
        }
        self.metrics["explained_variance"] = explained_variances

    def train(self):
        rollout = self.collect_rollout()

        idxs = np.arange(rollout["num_steps"])

        policy_losses = defaultdict(list)
        critic_losses = defaultdict(list)
        clip_fractions = defaultdict(list)

        num_grad_clipped = 0
        num_backwards = 0
        grad_norms = []
        for epoch in range(self.num_epochs):
            np.random.shuffle(idxs)

            if self.recompute_advantages:
                self.compute_advantages_and_returns(rollout)

            for i in range(len(idxs) // self.minibatch_size):
                idx = idxs[i * self.minibatch_size : (i + 1) * self.minibatch_size]
                action_logps = self.agent_group.train_forward(rollout, idx)

                tot_loss = torch.tensor(0.0, device=self.device)
                for agent_id in self.agent_ids:
                    adv = rollout["advantages"][agent_id][idx]
                    advantages = torch.as_tensor(
                        (adv - adv.mean()) / (adv.std() + 1e-8),
                        device=self.device,
                        dtype=torch.float32,
                    )

                    action_logps_old = torch.as_tensor(
                        rollout["action_logp"][agent_id][idx],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    ratio = torch.exp(action_logps[agent_id] - action_logps_old)

                    with torch.no_grad():
                        clip_fraction = torch.mean(
                            torch.gt(torch.abs(ratio - 1), 0.2).float()
                        ).item()
                        clip_fractions[agent_id].append(clip_fraction)

                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 0.8, 1.2)

                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    policy_losses[agent_id].append(policy_loss.item())

                    tot_loss += policy_loss

                values = self.agent_group.predict_values(
                    deep_idx(rollout["obs"], idx),  # TODO not batch first
                    deep_idx(rollout["actions"], idx),
                )

                for agent_id in self.agent_ids:
                    value_loss = F.mse_loss(
                        values[agent_id],
                        torch.as_tensor(
                            rollout["value_targets"][agent_id][idx],
                            device=self.device,
                            dtype=torch.float32,
                        ),
                    )

                    critic_losses[agent_id].append(value_loss.item())

                    tot_loss += value_loss

                tot_loss = self.agent_group.custom_loss(
                    loss=tot_loss,
                    rollout=rollout,
                    batch_idx=idx,
                    iteration=self.iteration,
                )
                for optim in self.agent_group.optims:
                    optim.zero_grad(set_to_none=True)
                tot_loss.backward()

                if self.gradient_clipping is not None:
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.agent_group.parameters(), self.gradient_clipping
                    )
                    if norm > self.gradient_clipping:
                        num_grad_clipped += 1
                    grad_norms.append(norm.item())
                num_backwards += 1

                for optim in self.agent_group.optims:
                    optim.step()

        if hasattr(self.agent_group, "metrics"):
            self.metrics.update(self.agent_group.metrics)

        self.metrics["policy_loss"] = {
            agent_id: np.array(policy_losses[agent_id]).mean()
            for agent_id in self.agent_ids
        }
        self.metrics["critic_loss"] = {
            agent_id: np.array(critic_losses[agent_id]).mean()
            for agent_id in self.agent_ids
        }
        self.metrics["ppo_clip_fraction"] = {
            agent_id: np.array(clip_fractions[agent_id]).mean()
            for agent_id in self.agent_ids
        }

        if self.gradient_clipping is not None:
            self.metrics["grad_clip_fraction"] = num_grad_clipped / num_backwards
            self.metrics["grad_norm_mean"] = np.array(grad_norms).mean()
            self.metrics["grad_norm_std"] = np.array(grad_norms).std()

        self.iteration += 1

    def evaluate(self, num_episodes=100):
        self.agent_group.reset()

        entropies = defaultdict(list)
        wins = 0
        all_rewards = defaultdict(list)
        episode_lens = []

        for _ in range(num_episodes):
            curr_episode_len = 0

            obs = self.env.reset()
            self.agent_group.reset()
            done = False

            while not done:
                curr_episode_len += 1
                with torch.no_grad():
                    actions, action_logps = self.agent_group.rollout_forward(
                        {
                            agent_id: squash(batchify(agent_obs), device=self.device)
                            for agent_id, agent_obs in obs.items()
                        }
                    )
                    for agent_id in action_logps.keys():
                        entropies[agent_id].append(
                            -(
                                torch.exp(action_logps[agent_id])
                                * action_logps[agent_id]
                            )
                            .sum()
                            .item()
                        )

                obs, rewards, done, _ = self.env.step(actions)
                for agent_id in rewards.keys():
                    all_rewards[agent_id].append(rewards[agent_id])

            episode_lens.append(curr_episode_len)
            if (
                hasattr(self.env, "metrics")
                and "won" in self.env.metrics.keys()
                and self.env.metrics["won"]
            ):
                wins += 1

        self.metrics.update(self.reward_signal.metrics())
        self.metrics["entropy"] = {
            agent_id: np.array(entropies[agent_id]).mean()
            for agent_id in self.agent_ids
        }

        if hasattr(self.env, "metrics") and "won" in self.env.metrics.keys():
            self.metrics["winrate"] = wins / num_episodes

        self.metrics["mean_episode_reward"] = {
            agent_id: sum(all_rewards[agent_id]) / num_episodes
            for agent_id in self.agent_ids
        }
        self.metrics["mean_episode_len"] = np.array(episode_lens).mean()


def main():
    import logging

    import tqdm
    from agent_groups import RecurrentMIMAgents2
    from envs.fire_commander import FireCommanderEnv
    from reward_signals import LocalActionGAILReward

    logging.basicConfig(filename="training.log", filemode="w", level=logging.DEBUG)

    trainer = MARLTrainer(
        FireCommanderEnv,
        RecurrentMIMAgents2(
            policy_lr=1e-3,
            critic_lr=2e-3,
            mim_coeff=0.1,
        ),
        LocalActionGAILReward(
            demo_filename="data/fire_commander_demos.dat",
            lr=1e-5,
            demo_limit=50000,
        ).normalized(),
        gae_lambda=0.5,
    )

    # for iteration in tqdm.trange(500):
    for iteration in tqdm.trange(5):
        if False and iteration % 50 == 1:
            logging.info(f"Starting iteration {iteration:03}")
            trainer.evaluate()
            trainer.train()
            for k, v in trainer.metrics.items():
                if k == "explained_variance":
                    logging.debug(
                        f"metric {k}: "
                        f"{{{', '.join(f'{vk}: {vv:.2%}' for vk, vv in v.items())}}}"
                    )
                elif isinstance(v, dict):
                    logging.debug(
                        f"metric {k}: "
                        f"{{{', '.join(f'{vk}: {vv:.4f}' for vk, vv in v.items())}}}"
                    )
                else:
                    logging.debug(f"metric {k}: {v}")

            if iteration % 50 == 0:
                filename = f"{trainer.agent_group.__class__.__name__}{iteration:03}"
                torch.save(
                    trainer.agent_group.state_dict(),
                    f"checkpoints/{filename}.pth",
                )
        else:
            trainer.train()


if __name__ == "__main__":
    main()
