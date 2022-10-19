from architectures import GaussianAgents, VonMisesAgents
from env import FantasticBits
from ppo import PPOConfig, PPOTrainer


def test_step():
    env = FantasticBits()
    agents = GaussianAgents()
    obs = env.reset()
    env.step(agents.step(obs)[0])


def test_short_train():
    trainer = PPOTrainer(GaussianAgents(), PPOConfig(rollout_steps=128, epochs=2))
    trainer.train_epoch()
    trainer.evaluate(num_episodes=2)


def test_vectorized_evaluate():
    trainer = PPOTrainer(VonMisesAgents(), PPOConfig(rollout_steps=128))
    trainer.vectorized_evaluate(num_episodes=10)
