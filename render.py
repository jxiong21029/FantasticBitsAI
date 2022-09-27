import sys
import time

import numpy as np
import pygame

from engine import POLES
from env import Bludger, FantasticBits, Snaffle, Wizard

env = FantasticBits(render=True)
env.reset()

for _ in range(200):
    env.step(
        {
            "id": np.array([0, 0], dtype=np.int64),
            "target": np.random.randn(2, 2),
        }
    )

