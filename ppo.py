import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import tqdm

from architectures import VonMisesAgents
from env import SZ_BLUDGER, SZ_GLOBAL, SZ_SNAFFLE, SZ_WIZARD, FantasticBits
from trainer import Trainer
from utils import RunningMoments, discount_cumsum, grad_norm


# adapted from SpinningUp PPO
class Buffer:
    """
    A buffer for storing trajectories experienced by the agents interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self, size, gamma=0.99, lam=0.95, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.device = device

        self.obs_buf = {
            "global": np.zeros((size, SZ_GLOBAL), dtype=np.float32),
        }
        for i in range(4):
            self.obs_buf[f"wizard{i}"] = np.zeros((size, SZ_WIZARD), dtype=np.float32)
        for i in range(7):
            self.obs_buf[f"snaffle{i}"] = np.full(
                (size, SZ_SNAFFLE), np.nan, dtype=np.float32
            )
        for i in range(2):
            self.obs_buf[f"bludger{i}"] = np.zeros((size, SZ_BLUDGER), dtype=np.float32)

        self.act_buf = {
            "id": np.zeros((size, 2), dtype=np.int64),
            "target": np.zeros((size, 2, 2), dtype=np.float32),
        }
        self.adv_buf = np.zeros((size, 2), dtype=np.float32)
        self.rew_buf = np.zeros((size, 2), dtype=np.float32)
        self.ret_buf = np.zeros((size, 2), dtype=np.float32)
        self.val_buf = np.zeros((size, 2), dtype=np.float32)
        self.logp_buf = np.zeros((size, 2), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.discounted_rew_tot = np.zeros(2, dtype=np.float32)
        self.rew_stats = [RunningMoments(), RunningMoments()]

    def store(self, obs, act, rew, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        if self.ptr == 0:
            for i in range(7):
                self.obs_buf[f"snaffle{i}"][:] = np.nan

        for k, v in obs.items():
            self.obs_buf[k][self.ptr] = v

        for k, v in act.items():
            self.act_buf[k][self.ptr] = v

        self.discounted_rew_tot = self.gamma * self.discounted_rew_tot + rew
        for i in range(2):
            self.rew_stats[i].push(self.discounted_rew_tot[i])
            self.rew_buf[self.ptr, i] = rew[i] / (self.rew_stats[i].std() + 1e-8)

        self.logp_buf[self.ptr] = logp

        self.ptr += 1

    def finish_path(self, last_vals):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(
            self.rew_buf[path_slice], np.reshape(last_vals, (1, 2)), axis=0
        )
        vals = np.append(
            self.val_buf[path_slice], np.reshape(last_vals, (1, 2)), axis=0
        )

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all the data from the buffer, with
        advantages appropriately normalized (shifted to have mean zero and std one).
        Also, resets some pointers in the buffer.
        """
        self.ptr = 0
        self.path_start_idx = 0

        explained_var = 1 - np.var(self.ret_buf - self.val_buf) / np.var(self.ret_buf)
        if self.logger is not None:
            self.logger.log(
                value_target_mean=np.mean(self.ret_buf),
                value_target_std=np.std(self.ret_buf),
                explained_variance=explained_var,
            )

        return {
            "obs": {
                k: torch.as_tensor(v, device=self.device)
                for k, v in self.obs_buf.items()
            },
            "act": {
                k: torch.as_tensor(v, device=self.device)
                for k, v in self.act_buf.items()
            },
            "ret": torch.as_tensor(self.ret_buf, device=self.device),
            "adv": torch.as_tensor(self.adv_buf, device=self.device),
            "logp": torch.as_tensor(self.logp_buf, device=self.device),
        }


class PPOTrainer(Trainer):
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
        rollout_device=torch.device("cpu"),
        train_device=torch.device("cpu"),
        seed=None,
    ):
        super().__init__(env_kwargs=env_kwargs, seed=seed)
        self.rollout_device = rollout_device
        self.train_device = train_device
        self._agents = agents
        self.optim = torch.optim.Adam(
            agents.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.buf = Buffer(
            size=rollout_steps,
            gamma=gamma,
            lam=gae_lambda,
            logger=self.logger,
            device=train_device,
        )
        self.minibatch_size = minibatch_size
        self.epochs = epochs

        self.ppo_clip_coeff = ppo_clip_coeff
        self.grad_clipping = grad_clipping
        self.entropy_reg = entropy_reg

        if env_kwargs is None:
            env_kwargs = {}
        env_kwargs["reward_gamma"] = gamma
        self.env = FantasticBits(**env_kwargs, logger=self.logger)
        self.env_kwargs = env_kwargs
        self.env_obs = self.env.reset()

        self.rollout = None

    @property
    def agents(self):
        return self._agents

    def collect_rollout(self):
        with torch.no_grad():
            self.agents.to(self.rollout_device)
            curr_ep_reward = 0

            for t in range(self.buf.max_size):
                action, logp = self.agents.step(self.env_obs)
                next_obs, reward, done = self.env.step(action)
                curr_ep_reward += reward.sum()
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

    def ppo_loss(self, batch_idx, logp, distrs):
        logp_old = self.rollout["logp"][batch_idx]
        ratio = torch.exp(logp - logp_old)

        adv = self.rollout["adv"][batch_idx]
        norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        clip_adv = (
            torch.clamp(ratio, 1 - self.ppo_clip_coeff, 1 + self.ppo_clip_coeff)
            * norm_adv
        )
        loss_pi = -(torch.min(ratio * norm_adv, clip_adv)).mean()

        v_pred = self.agents.value_forward(self.rollout, batch_idx)
        loss_v = ((v_pred - self.rollout["ret"][batch_idx]) ** 2).mean()

        total_loss = loss_pi + loss_v

        total_entropy = 0
        if self.entropy_reg > 0:
            for d in distrs:
                if isinstance(d, distributions.Normal):
                    mean_sq_norm = d.mean[:, 0] ** 2 + d.mean[:, 1] ** 2
                    scaled_entropy = (
                        torch.sum(
                            torch.log(
                                2
                                * torch.pi
                                * (d.variance / mean_sq_norm.reshape(-1, 1))
                            )
                        )
                        / batch_idx.shape[0]
                        / 2
                    )
                    total_loss += -self.entropy_reg * scaled_entropy
                    total_entropy += scaled_entropy.item() + 0.5
                else:
                    total_loss += (-self.entropy_reg * (ent := d.entropy())).mean()
                    total_entropy += ent.mean().item()

        self.logger.log(
            loss_pi=loss_pi.item(),
            loss_v=loss_v.item(),
            loss_ppo=total_loss.item(),
            ppo_clip_ratio=torch.gt(torch.abs(ratio - 1), self.ppo_clip_coeff)
            .float()
            .mean()
            .item(),
            entropy=total_entropy,
        )
        return total_loss

    def train(self):
        self.collect_rollout()
        self.agents.to(self.train_device)

        idx = np.arange(self.buf.max_size)

        # TODO architecture design: e.g. add relu, norm after preprocessors, restructure
        #  obs/action embeddings
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for i in range(idx.shape[0] // self.minibatch_size):
                batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]

                logp, distrs = self.agents.policy_forward(self.rollout, batch_idx)
                total_loss = self.ppo_loss(batch_idx, logp, distrs)

                self.optim.zero_grad(set_to_none=True)
                total_loss.backward()

                norm = grad_norm(self.agents)

                self.logger.log(
                    grad_norm=norm,
                    grad_norm_policy_encoder=grad_norm(self.agents.policy_encoder),
                    grad_norm_value_encoder=grad_norm(self.agents.value_encoder),
                    grad_norm_move_head=grad_norm(self.agents.move_head),
                    grad_norm_throw_head=grad_norm(self.agents.throw_head),
                    grad_norm_value_head=grad_norm(self.agents.value_head),
                )
                if self.grad_clipping is not None:
                    nn.utils.clip_grad_norm_(
                        self.agents.parameters(), self.grad_clipping
                    )
                    self.logger.log(
                        grad_clipped=(norm > self.grad_clipping)
                        if self.grad_clipping is not None
                        else False,
                    )
                self.optim.step()

        self.logger.step()

    def evaluate(self, num_episodes=50):
        self.agents.to(self.rollout_device)
        super().evaluate(num_episodes=50)

    def evaluate_with_render(self):
        self.agents.to(self.rollout_device)
        super().evaluate_with_render()


def main():
    trainer = PPOTrainer(
        VonMisesAgents(),
        rollout_steps=4096,
        lr=10**-3.5,
        weight_decay=1e-4,
        minibatch_size=512,
        gae_lambda=0.975,
        epochs=3,
        env_kwargs={
            "reward_shaping_snaffle_goal_dist": True,
            "reward_own_goal": 3,
        },
    )
    for i in tqdm.trange(501):
        trainer.train()
        if i % 20 == 0:
            trainer.evaluate()
            trainer.logger.generate_plots()


if __name__ == "__main__":
    main()
