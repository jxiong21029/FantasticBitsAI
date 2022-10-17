import time

import tqdm

from architectures import GaussianAgents
from env import FantasticBits

envs = [FantasticBits() for _ in range(2**11)]
agents = GaussianAgents()

for batch_size in (2**i for i in tqdm.trange(11)):
    total_steps = 0
    start = time.perf_counter()

    if batch_size == 1:
        obs = envs[0].reset()

        while time.perf_counter() - start < 10:
            action, _ = agents.step(obs)
            obs, _, done = envs[0].step(action)
            if done:
                envs[0].reset()
            total_steps += batch_size
    else:
        obses = []
        for env in envs[:batch_size]:
            obses.append(env.reset())
        assert len(obses) == batch_size

        while time.perf_counter() - start < 10:
            actions, _ = agents.vectorized_step(obses)

            for i, action in enumerate(actions):
                obses[i], _, done = envs[i].step(action)
                if done:
                    obses[i] = envs[i].reset()

            total_steps += batch_size

    print(batch_size, total_steps / (time.perf_counter() - start))
