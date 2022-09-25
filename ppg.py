import numpy as np
import scipy.signal
import torch

from env import FantasticBits

MAX_ENTITIES = 13
ENTITY_BITS = 9


# adapted from SpinningUp PPO
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[:, ::-1], axis=1)[:, ::-1]


class Buffer:
    """
    A buffer for storing trajectories experienced by the agents interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95):
        self.obs_buf = {
            "global_0": np.zeros((size, 3), dtype=np.float32),
            "entities_0": np.zeros((size, MAX_ENTITIES, ENTITY_BITS), dtype=np.float32),
            "global_1": np.zeros((size, 3), dtype=np.float32),
            "entities_1": np.zeros((size, MAX_ENTITIES, ENTITY_BITS), dtype=np.float32),
        }
        self.act_buf = {
            "id_0": np.zeros(size, dtype=int),
            "target_0": np.zeros((size, 2), dtype=np.float32),
            "id_1": np.zeros(size, dtype=int),
            "target_1": np.zeros((size, 2), dtype=np.float32),
        }
        self.adv_buf = np.zeros((2, size), dtype=np.float32)
        self.rew_buf = np.zeros((2, size), dtype=np.float32)
        self.ret_buf = np.zeros((2, size), dtype=np.float32)
        self.val_buf = np.zeros((2, size), dtype=np.float32)
        self.logp_buf = np.zeros((2, size), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        for k, v in obs.items():
            self.obs_buf[k][self.ptr] = v
        for k, v in act.items():
            self.act_buf[k][self.ptr] = v

        self.rew_buf[0, self.ptr] = rew["wizard_0"]
        self.rew_buf[1, self.ptr] = rew["wizard_1"]

        self.val_buf[0, self.ptr] = val["wizard_0"]
        self.val_buf[0, self.ptr] = val["wizard_1"]

        self.logp_buf[0, self.ptr] = logp["wizard_0"]
        self.logp_buf[1, self.ptr] = logp["wizard_1"]

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
            self.rew_buf[path_slice], np.reshape(last_vals, (2, 1)), axis=1
        )
        vals = np.append(
            self.val_buf[path_slice], np.reshape(last_vals, (2, 1)), axis=1
        )

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:, :-1] + self.gamma * vals[:, 1:] - vals[:, :-1]
        self.adv_buf[:, path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[:, path_slice] = discount_cumsum(rews, self.gamma)[:, :-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all the data from the buffer, with
        advantages appropriately normalized (shifted to have mean zero and std one).
        Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()

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

        self.env = FantasticBits()
        self.env_obs = self.env.reset()

    def collect_rollout(self):
        for t in range(self.buf.max_size):
            action, value, logp = self.agents.step()
            next_obs, reward, done = self.env.step(action)
            self.buf.store(self.env_obs, action, reward, value, logp)
            self.env_obs = next_obs

            epoch_ended = t == self.buf.max_size - 1
            if epoch_ended or done:
                if epoch_ended:
                    _, value, _ = self.agents.step(self.env_obs)
                else:
                    value = (0, 0)

                self.buf.finish_path(value)
                self.env_obs = self.env.reset()

        return self.buf.get()

    def train(self):
        rollout = self.collect_rollout()

        idx = np.arange(self.buf.max_size)
        for _ in range(self.wake_phases):
            self.rng.shuffle(idx)
            for i in range(idx.shape[0] // self.minibatch_size):
                batch_idx = idx[i * self.minibatch_size : (i + 1) * self.minibatch_size]

                logp_old = rollout["logp"][batch_idx]
                logp = self.agents.logp(rollout["obs"], batch_idx)
                ratio = torch.exp(logp - logp_old)
                adv = rollout["adv"][:, batch_idx]
                clip_adv = torch.clamp(ratio, 0.8, 1.2) * adv
                loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

                v_pred = self.agents.predict_values(rollout, batch_idx)
                loss_v = ((v_pred - rollout["ret"][:, batch_idx])**2).mean()

                self.optim.zero_grad()
                (loss_pi + loss_v).backward()
                self.optim.step()

        for _ in range(self.sleep_phases):
            pass