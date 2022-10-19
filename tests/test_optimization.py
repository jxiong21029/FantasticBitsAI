import numpy as np
import pytest
import torch
from torch.distributions import kl_divergence

from architectures import VonMisesAgents
from ppo import PPOConfig, PPOTrainer


@pytest.fixture(scope="module")
def sample_rollout():
    rng = np.random.default_rng(42 * 0xBEEF)
    torch.manual_seed(rng.integers(2**32))
    trainer = PPOTrainer(VonMisesAgents(), PPOConfig(rollout_steps=128))
    trainer.collect_rollout()
    return trainer.rollout


def test_kl_encourages_similar_policies(sample_rollout):
    rng = np.random.default_rng(0xF00DBEEF)

    for _ in range(10):
        torch.manual_seed(rng.integers(2**32))

        agents1 = VonMisesAgents()
        agents2 = VonMisesAgents()

        _, distrs1 = agents1.policy_forward(sample_rollout, np.arange(64))
        with torch.no_grad():
            _, distrs2 = agents2.policy_forward(sample_rollout, np.arange(64))

        assert distrs1[0].mean.shape == distrs1[1].mean.shape == (64,)
        assert not torch.equal(distrs1[0].mean, distrs1[1].mean)

        optim = torch.optim.Adam(agents1.parameters(), lr=1e-6)
        optim.zero_grad(set_to_none=True)
        loss = (kl_divergence(distrs2[0], distrs1[0])).mean() + (
            kl_divergence(distrs2[1], distrs1[1])
        ).mean()
        old_kl = loss.item()
        loss.backward()
        optim.step()

        with torch.no_grad():
            _, distrs3 = agents1.policy_forward(sample_rollout, np.arange(64))
        with torch.no_grad():
            _, distrs4 = agents2.policy_forward(sample_rollout, np.arange(64))

        assert torch.equal(distrs2[0].mean, distrs4[0].mean)
        assert torch.equal(distrs2[1].mean, distrs4[1].mean)
        assert torch.equal(distrs2[0].variance, distrs4[0].variance)
        assert torch.equal(distrs2[1].variance, distrs4[1].variance)
        assert not torch.equal(distrs1[0].mean, distrs2[0].mean)
        assert not torch.equal(distrs1[1].mean, distrs2[1].mean)
        assert not torch.equal(distrs1[0].variance, distrs2[0].variance)
        assert not torch.equal(distrs1[1].variance, distrs2[1].variance)
        new_kl = (
            (kl_divergence(distrs4[0], distrs3[0])).mean()
            + (kl_divergence(distrs4[1], distrs3[1])).mean()
        ).item()
        assert new_kl < old_kl
