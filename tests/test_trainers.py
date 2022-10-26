import numpy as np
from test_architectures import probably_dont_share_memory

from architectures import GaussianAgents, VonMisesAgents
from env import FantasticBits, FantasticBitsConfig
from ppo import PPOConfig, PPOTrainer


def test_step():
    env = FantasticBits()
    agents = GaussianAgents()
    obs = env.reset()
    env.step(agents.step(obs)[0])


def test_short_train():
    trainer = PPOTrainer(
        GaussianAgents(), PPOConfig(rollout_steps=128, minibatch_size=64, epochs=2)
    )
    trainer.run()
    trainer.evaluate(num_episodes=2)


def test_vectorized_evaluate():
    trainer = PPOTrainer(
        VonMisesAgents(), PPOConfig(rollout_steps=128, minibatch_size=64)
    )
    trainer.vectorized_evaluate(num_episodes=10)


def test_shared_reward_symmetry():
    rng = np.random.default_rng(0x42)

    for _ in range(10):
        rew_goal = rng.normal()
        trainer = PPOTrainer(
            VonMisesAgents(),
            PPOConfig(
                env_kwargs=FantasticBitsConfig(
                    reward_win=rng.normal(),
                    reward_loss=rng.normal(),
                    reward_own_goal=rew_goal,
                    reward_teammate_goal=rew_goal,
                    reward_opponent_goal=rng.normal(),
                    reward_shaping_snaffle_goal_dist=False,
                ),
                rollout_steps=128,
                minibatch_size=64,
            ),
        )

        trainer.collect_rollout()
        for buf in trainer.bufs:
            assert np.array_equal(buf.rew_buf[:, 0], buf.rew_buf[:, 1])
            assert probably_dont_share_memory(buf.rew_buf[:, 0], buf.rew_buf[:, 1])
