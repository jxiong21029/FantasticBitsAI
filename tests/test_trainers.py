from architectures import GaussianAgents
from env import FantasticBits
from ppo import PPOTrainer


def test_step():
    env = FantasticBits()
    agents = GaussianAgents()
    obs = env.reset()
    env.step(agents.step(obs)[0])


def test_short_train():
    trainer = PPOTrainer(
        GaussianAgents(), rollout_steps=128, epochs=2, seed=2**15 - 1
    )
    trainer.train()
    trainer.evaluate(num_episodes=2)
