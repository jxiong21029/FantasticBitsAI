import copy

import numpy as np
import torch

from architectures import GaussianAgents, VonMisesAgents
from ppo import PPOConfig, PPOTrainer


def test_permutation_invariance_gaussian():
    agents = GaussianAgents(flip_augment=False)
    trainer = PPOTrainer(agents, PPOConfig(rollout_steps=128))
    trainer.collect_rollout()

    rollout_cpy = copy.deepcopy(trainer.rollout)
    with torch.no_grad():
        logp1, distrs1 = agents.policy_forward(trainer.rollout, np.arange(128))
        logp2, distrs2 = agents.policy_forward(rollout_cpy, np.arange(128))

        assert torch.isclose(logp1, logp2).all()
        for i in range(2):
            assert torch.isclose(distrs1[i].mean, distrs2[i].mean).all()
            assert torch.isclose(distrs1[i].scale, distrs2[i].scale).all()

        rollout_cpy["obs"]["snaffle0"] = trainer.rollout["obs"]["snaffle1"]
        rollout_cpy["obs"]["snaffle1"] = trainer.rollout["obs"]["snaffle0"]
        rollout_cpy["obs"]["wizard2"] = trainer.rollout["obs"]["wizard3"]
        rollout_cpy["obs"]["wizard3"] = trainer.rollout["obs"]["wizard2"]
        rollout_cpy["obs"]["bludger0"] = trainer.rollout["obs"]["bludger1"]
        rollout_cpy["obs"]["bludger1"] = trainer.rollout["obs"]["bludger0"]

        logp3, distrs3 = agents.policy_forward(rollout_cpy, np.arange(128))

        assert torch.isclose(logp1, logp3).all(), f"{logp1[:5]} {logp3[:5]}"
        for i in range(2):
            assert torch.isclose(distrs1[i].mean, distrs3[i].mean).all()
            assert torch.isclose(distrs1[i].scale, distrs3[i].scale).all()


def test_permutation_invariance_von_mises():
    agents = VonMisesAgents(flip_augment=False)
    trainer = PPOTrainer(agents, PPOConfig(rollout_steps=128))
    trainer.collect_rollout()

    rollout_cpy = copy.deepcopy(trainer.rollout)
    with torch.no_grad():
        logp1, distrs1 = agents.policy_forward(trainer.rollout, np.arange(128))
        logp2, distrs2 = agents.policy_forward(rollout_cpy, np.arange(128))

        assert torch.isclose(logp1, logp2).all()
        for i in range(2):
            assert torch.isclose(distrs1[i].mean, distrs2[i].mean).all()
            assert torch.isclose(
                distrs1[i].concentration, distrs2[i].concentration
            ).all()

        rollout_cpy["obs"]["snaffle0"] = trainer.rollout["obs"]["snaffle1"]
        rollout_cpy["obs"]["snaffle1"] = trainer.rollout["obs"]["snaffle0"]
        rollout_cpy["obs"]["wizard2"] = trainer.rollout["obs"]["wizard3"]
        rollout_cpy["obs"]["wizard3"] = trainer.rollout["obs"]["wizard2"]
        rollout_cpy["obs"]["bludger0"] = trainer.rollout["obs"]["bludger1"]
        rollout_cpy["obs"]["bludger1"] = trainer.rollout["obs"]["bludger0"]

        logp3, distrs3 = agents.policy_forward(rollout_cpy, np.arange(128))

        assert torch.isclose(logp1, logp3).all(), f"{logp1[:5]} {logp3[:5]}"
        for i in range(2):
            assert torch.isclose(distrs1[i].mean, distrs3[i].mean).all()
            assert torch.isclose(
                distrs1[i].concentration, distrs3[i].concentration
            ).all()
