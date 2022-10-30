import os

import torch
import tqdm

from architectures import VonMisesAgents
from behavioral_cloning import BCTrainer
from ppo import PPOTrainer
from utils import PROJECT_DIR, Logger

SMOKE_TEST = False

agents = VonMisesAgents(
    num_layers=2, d_model=64, nhead=2, dim_feedforward=128, share_parameters=False
)
bc_trainer = BCTrainer(
    agents,
    demo_filename=os.path.join(PROJECT_DIR, "data/basic_demo.pickle"),
    lr=1e-3,
    minibatch_size=512,
    weight_decay=1e-4,
)
for _ in tqdm.trange(2 if SMOKE_TEST else 50):
    bc_trainer.run()
print(f"BC convergence: {bc_trainer.logger.estimate_convergence('loss_mean')}")

vf_trainer = PPOTrainer(agents)
vf_logger = Logger()
vf_trainer.pretrain_value(
    lr=1e-3,
    weight_decay=1e-3,
    epochs=2 if SMOKE_TEST else 25,
    logger=vf_logger,
    verbose=True,
)
print(f"VF convergence: {vf_logger.estimate_convergence('pretrain_loss_vf_mean')}")

vf_trainer.vectorized_evaluate(num_episodes=5 if SMOKE_TEST else 250)
print({k: v[-1] for k, v in vf_trainer.logger.cumulative_data.items()})

torch.save(agents.state_dict(), os.path.join(PROJECT_DIR, "data/pretrained_nops.pth"))
