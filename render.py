import time

import numpy as np
import torch

from architectures import Agents
from env import FantasticBits

# path = "ray_results/full_tune_2/train_2c8b2b5e_4_entropy_reg=0.0000,epochs=3,gae_lambda=0.9500,gamma=0.9900,lr=0.0100,minibatch_size=512,rollout_steps=4096,weight_2022-10-01_23-56-20/agents_2.pth"
path = "bc_agents.pth"

agents = Agents()
agents.load_state_dict(torch.load(path))
agents.eval()

done = False
eval_env = FantasticBits(bludgers_enabled=False, opponents_enabled=False, render=True)
obs = eval_env.reset()
tot_rew = np.zeros(2)
while not done:
    with torch.no_grad():
        actions, _ = agents.step(obs)
    obs, rew, done = eval_env.step(actions)
    tot_rew += rew
    time.sleep(0.1)
print(f"total reward: {tot_rew.tolist()}")
