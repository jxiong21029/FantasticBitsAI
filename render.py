import time

import numpy as np
import torch

from env import FantasticBits

# from architectures import VonMisesAgents
from experiments.distillation.repr_distill import ReDistillAgents

path = "experiments/ray_results/repr_distill_backup/train_41faa86e_28_epochs=2,lr=0.0010,minibatch_size=1024,weight_decay=0.0000_2022-10-13_08-12-07/{'lr': 0.001, 'minibatch_size': 1024, 'epochs': 2, 'weight_decay': 1e-05}_1.ckpt"

# agents = VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128)
agents = ReDistillAgents(num_layers=2, d_model=64, dim_feedforward=128)
agents.load_state_dict(torch.load(path))
agents.eval()

done = False
eval_env = FantasticBits(render=True, opponents_enabled=False, bludgers_enabled=False)
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
