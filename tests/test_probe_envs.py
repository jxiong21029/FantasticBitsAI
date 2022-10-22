import numpy as np
import torch

from architectures import VonMisesAgents
from env import FantasticBits
from ppo import PPOConfig, PPOTrainer


class ProbeEnv1(FantasticBits):
    def __init__(self):
        self.rng = np.random.default_rng()
        super().__init__()

    def reset(self):
        return FantasticBits().reset()

    def step(self, _actions):
        return (
            FantasticBits().reset(),
            np.full(2, fill_value=self.rng.normal(loc=6, scale=2)),
            True,
        )


def test_probe_1():
    agents = VonMisesAgents()
    trainer = PPOTrainer(
        agents,
        config=PPOConfig(
            rollout_steps=256,
            minibatch_size=256,
            epochs=2,
        ),
    )
    trainer.envs = [ProbeEnv1() for _ in range(len(trainer.envs))]
    trainer.env_obs = [env.reset() for env in trainer.envs]
    assert trainer.bufs[0].max_size == 256 / 8

    trainer.train_epoch()
    assert np.isclose(trainer.reward_stats.mean(), 6, rtol=0.1).all()
    assert np.isclose(trainer.reward_stats.std(), 2, rtol=0.1).all()

    assert trainer.rollout["ret"].shape == (256, 2)
    assert np.isclose(trainer.rollout["ret"].mean(), 3, rtol=0.1).all()


class ProbeEnv3(FantasticBits):
    def __init__(self):
        super().__init__()
        self.tt = None

    def reset(self):
        self.tt = 0
        return FantasticBits().reset()

    def step(self, _actions):
        self.tt += 1
        if self.tt == 1:
            return FantasticBits().reset(), np.zeros(2), False
        else:
            return FantasticBits().reset(), np.ones(2), True


def test_probe_3():
    agents = VonMisesAgents()
    trainer = PPOTrainer(
        agents,
        config=PPOConfig(
            rollout_steps=256,
            minibatch_size=256,
            epochs=2,
        ),
    )
    trainer.envs = [ProbeEnv3() for _ in range(len(trainer.envs))]
    trainer.env_obs = [env.reset() for env in trainer.envs]

    trainer.train_epoch()
    assert np.isclose(trainer.reward_stats.std(), 0.5, rtol=0.1)
    assert np.isclose(trainer.rollout["ret"].mean(), 2, rtol=0.1)


class ProbeEnv4(FantasticBits):
    def __init__(self):
        super().__init__()

    def reset(self):
        return FantasticBits().reset()

    def step(self, actions):
        rew = np.zeros(2)
        assert actions["target"].shape == (2, 2)
        if actions["target"][0, 1] > 0:
            rew[0] = 1
        if actions["target"][1, 1] > 0:
            rew[1] = 1
        return FantasticBits().reset(), rew, True


def test_probe_4():
    agents = VonMisesAgents(flip_augment=False)
    trainer = PPOTrainer(
        agents,
        config=PPOConfig(
            lr=10**-3.5,
            rollout_steps=256,
            minibatch_size=256,
            entropy_reg=0,
            epochs=2,
        ),
    )
    trainer.envs = [ProbeEnv4() for _ in range(len(trainer.envs))]
    trainer.env_obs = [env.reset() for env in trainer.envs]

    results = np.zeros(15)
    for i in range(15):
        trainer.train_epoch()
        results[i] = (
            (trainer.rollout["act"]["target"][:, :, 1] > 0).to(torch.float32).mean()
        )

    assert results[-5:].mean() > results[:5].mean()
    assert results[-1] > 0.55


class ProbeEnv6(FantasticBits):
    def __init__(self):
        super().__init__()

    def step(self, actions):
        rew = np.zeros(2)

        assert actions["target"].shape == (2, 2)

        # rewards agents for moving towards central horizontal line @ y=3750
        # reward dependends on both obs and action
        for i in range(2):
            a = self.agents[i].y >= 3750
            b = actions["target"][i, 1] > 0
            rew[i] = 1 if a ^ b else 0

        return FantasticBits().reset(), rew, True


def test_probe_6():
    agents = VonMisesAgents()
    trainer = PPOTrainer(
        agents,
        config=PPOConfig(
            lr=10**-3.5,
            rollout_steps=256,
            minibatch_size=256,
            entropy_reg=0,
            epochs=2,
        ),
    )
    trainer.envs = [ProbeEnv6() for _ in range(len(trainer.envs))]
    trainer.env_obs = [env.reset() for env in trainer.envs]

    results = np.zeros(25)
    for i in range(25):
        trainer.train_epoch()
        results[i] = trainer.reward_stats.mean()
    assert results[-10:].mean() > results[:10].mean()
