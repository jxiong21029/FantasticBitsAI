import copy

import numpy as np
import torch

from architectures import VonMisesAgents
from experiments.optimization.vectorized_ppo import VectorizedPPOTrainer
from ppo import PPOTrainer


def test_one_env_matches():
    agents = VonMisesAgents()
    trainer1 = PPOTrainer(
        agents, rollout_steps=256, env_kwargs={"seed": 0xF00D}, seed=0xFFFF
    )

    trainer2 = VectorizedPPOTrainer(
        copy.deepcopy(agents),
        rollout_steps=256,
        env_kwargs={"seed": 0xF00D},
        seed=0xFFFF,
    )

    torch.manual_seed(0xBEEF)
    np.random.seed(0xBEEEF)
    trainer1.env_obs = trainer1.env.reset()
    trainer1.env.rng = np.random.default_rng(0xBEE)
    trainer1.collect_rollout()
    a = trainer1.env.rng.integers(2**31)

    torch.manual_seed(0xBEEF)
    np.random.seed(0xBEEEF)
    trainer2.env[0].rng = np.random.default_rng(0xBEE)
    trainer2.collect_rollout()
    b = trainer2.env[0].rng.integers(2**31)

    assert a == b

    assert "obs" in trainer1.rollout.keys()
    assert "ret" in trainer2.rollout.keys()
    for k in trainer1.rollout.keys():
        if isinstance(trainer1.rollout[k], dict):
            for k2 in trainer1.rollout[k].keys():
                if not torch.isclose(
                    trainer1.rollout[k][k2],
                    trainer2.rollout[k][k2],
                    equal_nan=True,
                ).all():
                    print(
                        torch.argwhere(
                            ~torch.isclose(
                                trainer1.rollout[k][k2],
                                trainer2.rollout[k][k2],
                                equal_nan=True,
                            )
                        )
                    )
                    raise AssertionError
        else:
            if not torch.isclose(
                trainer1.rollout[k], trainer2.rollout[k], equal_nan=True
            ).all():
                print(
                    torch.argwhere(
                        ~torch.isclose(
                            trainer1.rollout[k], trainer2.rollout[k], equal_nan=True
                        )
                    )
                )
                raise AssertionError
