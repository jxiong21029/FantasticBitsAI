import time

import numpy as np

from env import FantasticBits

for i in range(3):
    rng = np.random.default_rng(0xF00DBEEF + i)
    env = FantasticBits(seed=rng.integers(2**31))
    start = time.perf_counter()
    c = 0
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            _, _, done = env.step(
                {
                    "id": np.zeros(
                        2,
                    ),
                    "target": rng.normal(size=(2, 2)),
                }
            )
            c += 1
    print(f"timesteps: {c}")
    print(f"time taken: {time.perf_counter() - start}")
