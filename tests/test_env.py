import numpy as np
import pytest

from env import FantasticBits, DIST_NORM
from utils import vertical_flip_augment


@pytest.fixture
def null_action():
    return {
        "id": np.zeros(2, dtype=np.int64),
        "target": np.zeros((2, 2), dtype=np.float32),
    }


def test_upper_left_collisions(null_action):
    flag1 = False
    flag2 = False
    for starting_x in range(10):
        for starting_y in range(10):
            for starting_vx in (-1, -2):
                for starting_vy in (-1, -2):
                    env = FantasticBits()
                    env.reset()

                    nearest = min(env.snaffles, key=lambda s: s.x**2 + s.y**2)
                    nearest.x = starting_x + nearest.rad
                    nearest.y = starting_y + nearest.rad
                    nearest.vx = starting_vx
                    nearest.vy = starting_vy

                    for _ in range(3):
                        env.step(null_action)

                    if starting_vx + starting_x < 0:
                        flag1 = True
                        assert nearest.vx >= 0
                    if starting_vy + starting_y < 0:
                        flag2 = True
                        assert nearest.vy >= 0

    assert flag1 and flag2


def test_weird_goal_angle_collisions(null_action):
    rng = np.random.default_rng(seed=2**16 - 42)
    for _ in range(100):
        x = rng.integers(0, 500)
        y = rng.integers(2000, 5500)
        vx = rng.integers(-500, 0)
        vy = rng.integers(-5000, 5000)

        env = FantasticBits()
        env.reset()
        del env.snaffles[1:]
        env.snaffles[0].x = x
        env.snaffles[0].y = y
        env.snaffles[0].vx = vx
        env.snaffles[0].vy = vy
        env.agents[0].x += 1000
        env.agents[1].x += 1000

        for _ in range(10):
            env.step(null_action)
            if len(env.snaffles) == 0:
                break
            assert 0 <= env.snaffles[0].x <= 16000
            assert 0 <= env.snaffles[0].y <= 7500

    for _ in range(100):
        x = rng.integers(0, 10)
        y = rng.integers(2150, 5350)
        vx = rng.integers(-3, 0)
        vy = rng.integers(-5000, 5000)

        env = FantasticBits()
        env.reset()
        del env.snaffles[1:]
        env.snaffles[0].x = x
        env.snaffles[0].y = y
        env.snaffles[0].vx = vx
        env.snaffles[0].vy = vy
        env.agents[0].x += 1000
        env.agents[1].x += 1000

        for i in range(10):
            env.step(null_action)
            if len(env.snaffles) == 0:
                break
            assert 0 <= env.snaffles[0].x <= 16000
            assert 0 <= env.snaffles[0].y <= 7500


def test_dist_reward(null_action):
    env = FantasticBits(reward_shaping_snaffle_goal_dist=True)
    env.reset()

    env.snaffles = [snaffle := env.snaffles[0]]
    snaffle.x = 2000
    snaffle.vx = 500
    snaffle.y = 3750
    env.bludgers.clear()

    _, rewards, _ = env.step(null_action)
    assert rewards[0] == rewards[1] == 0.036


def test_flip_augment(null_action):
    rng = np.random.default_rng()

    env = FantasticBits(seed=rng.integers(2**31))
    obs = env.reset()
    double_flipped = vertical_flip_augment(vertical_flip_augment(obs))
    for k in obs.keys():
        assert np.array_equal(obs[k], double_flipped[k])

    for entity in env.agents + env.snaffles + env.opponents + env.bludgers:
        entity.vx = rng.integers(-500, 501)
        entity.vy = rng.integers(-500, 501)

    obs, _, _ = env.step(null_action)
    double_flipped = vertical_flip_augment(vertical_flip_augment(obs))
    for k in obs.keys():
        assert np.array_equal(obs[k], double_flipped[k])
        if k != "global":
            assert -3750 / DIST_NORM <= obs[k][1] <= 3750 / DIST_NORM
