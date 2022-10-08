import math
import pickle

import numpy as np
import tqdm

from behavioral_cloning import stack_actions, stack_obs
from env import FantasticBits


def medium_heuristic(
    env: FantasticBits,
    target_leading=1.0,
    smart_shots=True,
    snaffle_pairing=True,
    angle_noise=0.0,
    seed=None,
):
    rng = np.random.default_rng(seed=seed)

    actions = {
        "id": np.zeros(2, dtype=np.int64),
        "target": np.zeros((2, 2), dtype=np.float32),
    }

    targets = [None, None]
    for i, wizard in enumerate(env.agents):
        if wizard.grab_cd == 2:
            actions["id"][i] = 1
            tx = 16000 - wizard.x - wizard.vx
            ty = (
                3750
                - wizard.y
                - wizard.vy
                + (wizard.y / 7500 * 1000 if smart_shots else 0)
            )
            angle = np.arctan2(ty, tx) + rng.normal() * angle_noise
            actions["target"][i] = (float(np.cos(angle)), float(np.sin(angle)))
        else:
            actions["id"][i] = 0
            nearest_snaffle = min(env.snaffles, key=lambda s: s.distance2(wizard))
            targets[i] = nearest_snaffle

    if (
        snaffle_pairing
        and targets[0] is targets[1]
        and targets[0] is not None
        and len(env.snaffles) > 1
    ):
        if env.agents[0].distance2(targets[0]) < env.agents[1].distance2(targets[0]):
            targets[1] = min(
                (s for s in env.snaffles if s is not targets[0]),
                key=lambda s: s.distance2(env.agents[1]),
            )
        else:
            targets[0] = min(
                (s for s in env.snaffles if s is not targets[1]),
                key=lambda s: s.distance2(env.agents[0]),
            )

    for i, wizard in enumerate(env.agents):
        if (target := targets[i]) is not None:
            vx, vy = None, None
            if min(opp.distance2(target) for opp in env.opponents) < min(
                wiz.distance2(target) for wiz in env.agents
            ):
                for opp in env.opponents:
                    if target.distance(opp) < 700:
                        vx, vy = -opp.x, 3750 - opp.y
                        norm = math.sqrt(vx * vx + vy * vy)
                        vx = vx / norm * 1000
                        vy = vy / norm * 1000
                        break
            if vx is None:
                vx, vy = target.vx, target.vy

            tx = target.x + target_leading * vx - wizard.x
            ty = (target.y + target_leading * vy - wizard.y,)
            angle = np.arctan2(ty, tx) + rng.normal() * angle_noise
            actions["target"][i] = (float(np.cos(angle)), float(np.sin(angle)))

    return actions


def generate_medium_demos_with_returns(num_episodes=100, env_kwargs=None, seed=None):
    if env_kwargs is None:
        env_kwargs = {}

    rng = np.random.default_rng(seed=seed)

    env = FantasticBits(**env_kwargs)
    obs_buf = []
    act_buf = []
    rtg_buf = []
    wins = 0

    for _ in tqdm.trange(num_episodes):
        target_leading = rng.choice([0, 1, 2 * rng.random()])
        smart_shots = rng.choice([True, False], p=[0.8, 0.2])
        snaffle_pairing = rng.choice([True, False], p=[0.8, 0.2])
        angle_noise = rng.choice([0.0, 0.25 * rng.random()], p=[0.8, 0.2])

        obs = env.reset()
        ep_rewards = []
        done = False
        while not done:
            obs_buf.append(obs)
            actions = medium_heuristic(
                env,
                target_leading=target_leading,
                smart_shots=smart_shots,
                snaffle_pairing=snaffle_pairing,
                angle_noise=angle_noise,
                seed=rng.integers(2**31),
            )
            act_buf.append(actions)
            obs, rew, done = env.step(actions)

            ep_rewards.append(rew)
        if env.score[0] > env.score[1]:
            wins += 1

        rtgs = np.zeros((len(ep_rewards), 2), dtype=np.float32)
        curr = np.zeros(2, dtype=np.float32)
        for i, rew in enumerate(reversed(ep_rewards)):
            curr += rew
            rtgs[len(rtgs) - i - 1] = curr
        rtg_buf.append(rtgs)

    ret_rtg = np.concatenate(rtg_buf, axis=0)
    ret_obs = stack_obs(obs_buf)
    ret_act = stack_actions(act_buf)

    with open("../../data/medium_demo_noisy.pickle", "wb") as f:
        pickle.dump((ret_rtg, ret_obs, ret_act), f)
    print(f"generated demos, winrate: {wins / num_episodes:.3f}")


if __name__ == "__main__":
    generate_medium_demos_with_returns(num_episodes=1000)
