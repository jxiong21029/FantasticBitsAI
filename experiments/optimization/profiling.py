from architectures import GaussianAgents, VonMisesAgents
from ppo import PPOTrainer
from utils import profileit


@profileit
def train():
    trainer = PPOTrainer(
        VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128),
        rollout_steps=4096,
        minibatch_size=512,
        env_kwargs={"reward_shaping_snaffle_goal_dist": True},
    )
    trainer.train_epoch()
    trainer.train_epoch()


@profileit
def evaluate():
    trainer = PPOTrainer(
        VonMisesAgents(num_layers=2, d_model=64, dim_feedforward=128),
        rollout_steps=4096,
        minibatch_size=512,
        env_kwargs={"reward_shaping_snaffle_goal_dist": True},
    )
    trainer.vectorized_evaluate(num_episodes=100)


train()
