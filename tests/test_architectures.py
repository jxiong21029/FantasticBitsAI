import copy

import numpy as np
import torch

from architectures import GaussianAgents, VonMisesAgents
from env import FantasticBits
from ppo import PPOConfig, PPOTrainer


def probably_dont_share_memory(arr1, arr2):
    if not np.may_share_memory(arr1, arr2):
        return True
    try:
        return not np.shares_memory(arr1, arr2, max_work=1000)
    except np.TooHardError:
        return True


def test_objects_permutation_invariance_gaussian():
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


def test_objects_permutation_invariance_von_mises():
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


def test_agents_permutation_symmetry():
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

        rollout_cpy["obs"]["wizard0"] = trainer.rollout["obs"]["wizard1"]
        rollout_cpy["obs"]["wizard1"] = trainer.rollout["obs"]["wizard0"]
        rollout_cpy["act"]["id"][:, 0] = trainer.rollout["act"]["id"][:, 1]
        rollout_cpy["act"]["id"][:, 1] = trainer.rollout["act"]["id"][:, 0]
        rollout_cpy["act"]["target"][:, 0] = trainer.rollout["act"]["target"][:, 1]
        rollout_cpy["act"]["target"][:, 1] = trainer.rollout["act"]["target"][:, 0]

        assert probably_dont_share_memory(
            rollout_cpy["obs"]["wizard0"], rollout_cpy["obs"]["wizard1"]
        )

        logp3, distrs3 = agents.policy_forward(rollout_cpy, np.arange(128))

        assert torch.isclose(logp1[:, 0], logp3[:, 1]).all(), f"{logp1[:5]} {logp3[:5]}"
        assert torch.isclose(logp1[:, 1], logp3[:, 0]).all(), f"{logp1[:5]} {logp3[:5]}"
        for i in range(2):
            assert torch.isclose(distrs1[i].mean, distrs3[1 - i].mean).all()
            assert torch.isclose(
                distrs1[i].concentration, distrs3[1 - i].concentration
            ).all()


def test_key_ordering_invariance():
    rng = np.random.default_rng(0xBEEF)

    agents = VonMisesAgents()
    env = FantasticBits()

    obs = env.reset()

    for _ in range(5):
        keys = list(obs.keys())
        rng.shuffle(keys)

        obs2 = {}
        for k in keys:
            obs2[k] = copy.deepcopy(obs[k])

        for _ in range(10):
            seed = rng.integers(2**31)

            torch.random.manual_seed(seed)
            actions1, logps1 = agents.step(obs)

            torch.random.manual_seed(seed)
            actions2, logps2 = agents.step(obs2)

            for k in actions1.keys():
                assert np.array_equal(actions1[k], actions2[k])
            assert np.array_equal(logps1, logps2)


def test_key_ordering_invariance_vectorized():
    rng = np.random.default_rng(0xBEEF)

    agents = VonMisesAgents()
    env = FantasticBits()

    obses = [env.reset() for _ in range(8)]

    for _ in range(5):
        obses2 = [{} for _ in range(8)]
        for i in range(len(obses)):
            keys = list(obses[i].keys())
            rng.shuffle(keys)
            for k in keys:
                obses2[i][k] = copy.deepcopy(obses[i][k])

        for _ in range(10):
            seed = rng.integers(2**31)

            torch.random.manual_seed(seed)
            actions1, logps1 = agents.vectorized_step(obses)

            torch.random.manual_seed(seed)
            actions2, logps2 = agents.vectorized_step(obses2)

            for a1, l1, a2, l2 in zip(actions1, logps1, actions2, logps2):
                for k in a1.keys():
                    assert np.isclose(a1[k], a2[k]).all()
                assert np.isclose(l1, l2).all()
