import time

import numpy as np
import torch
import tqdm

# from architectures import VonMisesAgents
from env import FantasticBits
from experiments.distillation.repr_distill import ReDistillAgents

path = "bc_agents.pth"

# agents = VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128)
agents = ReDistillAgents(num_layers=2, d_model=64, dim_feedforward=128)
agents.load_state_dict(torch.load(path)["agents_state_dict"])
agents.eval()

goals = 0
wins = 0
for _ in tqdm.trange(1000):
    eval_env = FantasticBits(render=True)
    obs = eval_env.reset()
    while True:
        actions = {
            "id": np.zeros(2, dtype=np.int64),
            "target": np.zeros((2, 2), dtype=np.float32),
        }
        with torch.no_grad():
            z = agents.policy_encoder(obs)
            for i in range(2):
                embed = z[i + 1]
                if obs[f"wizard{i}"][5] == 1:  # throw available
                    actions["id"][i] = 1
                    logits = agents.throw_head(embed)
                else:
                    actions["id"][i] = 0
                    logits = agents.move_head(embed)
                actions["target"][i] = logits[:2]
        obs, rew, done = eval_env.step(actions)
        if done:
            break
        time.sleep(0.1)

    goals += eval_env.score[0]
    if eval_env.score[0] > eval_env.score[1]:
        wins += 1

print(f"goals: {goals}")
print(f"wins: {wins}")
