import math

import numpy as np
import scipy.signal
import torch
import tqdm

from agents import Agents
from env import SZ_BLUDGER, SZ_GLOBAL, SZ_SNAFFLE, SZ_WIZARD, FantasticBits


# adapted from SpinningUp PPO
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class RunningMoments:
    """
    Tracks running mean and variance
    Adapted from github.com/MadryLab/implementation-matters, which took it from
    github.com/joschu/modular_rl. Math in johndcook.com/blog/standard_deviation
    """

    def __init__(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def push(self, x):
        assert isinstance(x, float) or isinstance(x, int)
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            old_m = self.m
            self.m = old_m + (x - old_m) / self.n
            self.s = self.s + (x - old_m) * (x - self.m)

    def mean(self):
        return self.m

    def std(self):
        if self.n > 1:
            return math.sqrt(self.s / (self.n - 1))
        else:
            return self.m


class Buffer:
    """
    A buffer for storing trajectories experienced by the agents interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95):
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

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store

        for k, v in obs.items():
            self.obs_buf[k][self.ptr] = v

        for k in self.act_buf.keys():
            self.act_buf[k][self.ptr] = act[k]

        self.discounted_rew_tot = 0.99 * self.discounted_rew_tot + rew
        for i in range(2):
            self.rew_stats[i].push(self.discounted_rew_tot[i])
            self.rew_buf[self.ptr, i] = rew[i] / (self.rew_stats[i].std() + 1e-8)

        self.val_buf[self.ptr, :] = val
        self.logp_buf[self.ptr, :] = logp

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
        assert np.isclose(
            discount_cumsum(rews[:, 0], self.gamma)[:-1], self.ret_buf[path_slice][:, 0]
        ).all()
        assert np.isclose(
            discount_cumsum(rews[:, 1], self.gamma)[:-1], self.ret_buf[path_slice][:, 1]
        ).all()

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all the data from the buffer, with
        advantages appropriately normalized (shifted to have mean zero and std one).
        Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        print(
            f"value targets: mean {np.mean(self.ret_buf)}, std {np.std(self.ret_buf)}"
        )

        explained_var = 1 - np.var(self.ret_buf - self.val_buf) / np.var(self.ret_buf)
        print(f"explained variance: {explained_var:.1%}")

        return {
            "obs": {k: torch.as_tensor(v) for k, v in self.obs_buf.items()},
            "act": {k: torch.as_tensor(v) for k, v in self.act_buf.items()},
            "ret": torch.as_tensor(self.ret_buf),
            "adv": torch.as_tensor(self.adv_buf),
            "logp": torch.as_tensor(self.logp_buf),
        }


class Trainer:
    def __init__(
        self,
        agents,
        env_fn,
        env_kwargs=None,
        lr=1e-3,
        rollout_steps=4096,
        gamma=0.99,
        gae_lambda=0.95,
        minibatch_size=64,
        wake_phases=1,
        sleep_phases=2,
        seed=None,
    ):
        # agent.step(single obs) -> action, value, logp
        # agent.predict_values(batch obs) -> values (2xB)
        # agent.logp(batch obs) -> logps

        self.rng = np.random.default_rng(seed=seed)

        self.agents = agents
        self.optim = torch.optim.Adam(agents.parameters(), lr=lr)
        self.buf = Buffer(size=rollout_steps, gamma=gamma, lam=gae_lambda)
        self.minibatch_size = minibatch_size
        self.wake_phases = wake_phases
        self.sleep_phases = sleep_phases

        if env_kwargs is None:
            env_kwargs = {}
        self.env = env_fn(**env_kwargs)
        self.env_fn = env_fn
        self.env_kwargs = env_kwargs
        self.env_obs = self.env.reset()

    def collect_rollout(self):
        tot_r = 0
        for t in tqdm.trange(self.buf.max_size):
            action, logp = self.agents.step(self.env_obs)
            value = self.agents.predict_value(self.env_obs, None)
            next_obs, reward, done = self.env.step(action)
            tot_r += reward.sum()
            self.buf.store(self.env_obs, action, reward, value, logp)
            self.env_obs = next_obs

            epoch_ended = t == self.buf.max_size - 1
            if epoch_ended or done:
                if done:
                    value = (0, 0)
                else:
                    value = self.agents.predict_value(self.env_obs, None)

                self.buf.finish_path(value)
                self.env_obs = self.env.reset()

        print(f"mean reward per 200: {tot_r / self.buf.max_size:.3f}")
        return self.buf.get()

    def train(self):
        rollout = self.collect_rollout()

        idx = np.arange(self.buf.max_size)

        losses_pi = []
        losses_v = []
        grad_clipped = []
        grad_norms = []
        clip_ratios = []
        # TODO code review and debugger run
        # TODO probe environments
        # TODO clip ratio, entropy
        for _ in range(self.wake_phases):
            self.rng.shuffle(idx)
            for i in range(idx.shape[0] // self.minibatch_size):
                batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]

                logp_old = rollout["logp"][batch_idx]
                logp = self.agents.logp(rollout, batch_idx)
                ratio = torch.exp(logp - logp_old)

                clip_ratios.append(
                    torch.gt(torch.abs(ratio - 1), 0.2).float().mean().item()
                )

                adv = rollout["adv"][batch_idx]
                norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                clip_adv = torch.clamp(ratio, 0.8, 1.2) * norm_adv
                loss_pi = -(torch.min(ratio * norm_adv, clip_adv)).mean()
                losses_pi.append(loss_pi.item())

                v_pred = self.agents.value_forward(rollout, batch_idx)
                loss_v = ((v_pred - rollout["ret"][batch_idx]) ** 2).mean()
                losses_v.append(loss_v.item())

                self.optim.zero_grad()
                (loss_pi + loss_v).backward()
                norm = torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 10.0)
                if norm > 10.0:
                    grad_clipped.append(True)
                else:
                    grad_clipped.append(False)
                grad_norms.append(norm.item())
                self.optim.step()

        print(
            f"policy loss mean: {np.mean(losses_pi):.2f}, std: {np.std(losses_pi):.2f}"
        )
        print(f"value loss mean: {np.mean(losses_v):.2f}, std: {np.std(losses_v):.2f}")
        print(f"ppo clip proportion: {np.mean(clip_ratios):.1%}")
        print(f"grad clip proportion: {np.mean(grad_clipped):.1%}")
        print(
            f"grad norm mean: {np.mean(grad_norms):.2f}, std: {np.std(grad_norms):.2f}"
        )

        for _ in range(self.sleep_phases):
            pass

    # def evaluate(self):
    #     done = False
    #     eval_env = copy.deepcopy(self.env)
    #     obs = eval_env.reset()
    #     while not done:
    #         with torch.no_grad():
    #             actions = self.agents.step(obs)

    def evaluate_with_render(self):
        import pygame
        import sys
        import time

        done = False
        eval_env = self.env_fn(**self.env_kwargs, render=True)
        obs = eval_env.reset()
        tot_rew = np.zeros(2)
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    pygame.quit()
                    pygame.display.quit()
                    sys.exit()

            with torch.no_grad():
                actions, _ = self.agents.step(obs)
            obs, rew, done = eval_env.step(actions)
            tot_rew += rew
            time.sleep(0.1)
        print(f"total reward: {tot_rew.tolist()}")


def main():
    trainer = Trainer(
        Agents(),
        FantasticBits,
        rollout_steps=4096,
        lr=3e-4,
        wake_phases=3,
        env_kwargs={"shape_snaffling": True},
    )
    for _ in range(250):
        trainer.train()


if __name__ == "__main__":
    main()
