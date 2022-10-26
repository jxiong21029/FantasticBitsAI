import time

import numpy as np
import ray
import torch

from env import FantasticBits
from experiments.distillation.redistill import ReDistillAgents

path = (
    "ray_results/limit_test/train_ffbab_00004_4_2022-10-22_18-15-14/checkpoint/state.pt"
)


@ray.remote(num_cpus=1)
def eval_deterministic(num_episodes):
    agents = ReDistillAgents(
        num_layers=2, d_model=64, dim_feedforward=128, share_parameters=True
    )
    agents.load_state_dict(torch.load(path)["agents_state_dict"])
    agents.eval()

    goals = 0
    wins = 0
    for _ in range(num_episodes):
        eval_env = FantasticBits()
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

        goals += eval_env.score[0]
        if eval_env.score[0] > eval_env.score[1]:
            wins += 1

    return goals, wins


def eval_with_render(num_episodes=5, deterministic=True):
    agents = ReDistillAgents(
        num_layers=2, d_model=64, dim_feedforward=128, share_parameters=True
    )
    ckpt = torch.load(path)
    print("step:", ckpt["step"])
    agents.load_state_dict(ckpt["agents_state_dict"])
    agents.eval()

    goals = 0
    wins = 0
    for _ in range(num_episodes):
        eval_env = FantasticBits(render=True)
        obs = eval_env.reset()

        while True:
            if deterministic:
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
            else:
                actions, _ = agents.step(obs)
            obs, rew, done = eval_env.step(actions)
            if done:
                break
            time.sleep(0.1)

        goals += eval_env.score[0]
        if eval_env.score[0] > eval_env.score[1]:
            wins += 1

    print("goals / ep:", goals / num_episodes)
    print(f"winrate: {wins / num_episodes:.1%}")


if __name__ == "__main__":
    eval_with_render(deterministic=False)
