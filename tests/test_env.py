import numpy as np
import pytest

from env import FantasticBits

env = FantasticBits()
env.reset()

for _ in range(50):
    env.step(
        {
            "wizard_0": {"move": np.array([1, -1])},
            "wizard_1": {"move": np.array([0, 1])},
        }
    )
