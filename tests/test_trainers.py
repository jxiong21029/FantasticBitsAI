from architectures import GaussianAgents, VonMisesAgents
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
    trainer.train_epoch()
    trainer.evaluate(num_episodes=2)


def test_vectorized_evaluate():
    trainer = PPOTrainer(VonMisesAgents(), rollout_steps=128)
    trainer.vectorized_evaluate(num_episodes=10)
