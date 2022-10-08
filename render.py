import time

import numpy as np
import torch

from env import FantasticBits

# from architectures import Agents
from experiments.action_parameterization.von_mises_agents import VonMisesAgents

path = "experiments/action_parameterization/checkpoints/von_mises_0.ckpt"

agents = VonMisesAgents(num_layers=2, d_model=64)
agents.load_state_dict(torch.load(path))
agents.eval()

done = False
eval_env = FantasticBits(render=True)
obs = eval_env.reset()
tot_rew = np.zeros(2)
while not done:
    with torch.no_grad():
        actions, _ = agents.step(obs)
    obs, rew, done = eval_env.step(actions)
    tot_rew += rew
    time.sleep(0.1)
print(f"total reward: {tot_rew.tolist()}")
print(f"final score: {eval_env.score[0]} - {eval_env.score[1]}")
