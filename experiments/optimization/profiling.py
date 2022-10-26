from architectures import VonMisesAgents
from ppo import PPOTrainer
from utils import profileit


@profileit
def train():
    trainer = PPOTrainer(VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128))
    trainer.run()
    trainer.run()


@profileit
def evaluate():
    trainer = PPOTrainer(VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128))
    trainer.vectorized_evaluate(num_episodes=100)


train()
