import numpy as np
import torch
import tqdm

from architectures import GaussianAgents, VonMisesAgents
from behavioral_cloning import BCTrainer


def train(config):
    results = []
    for run in range(5):
        if config["action_parameterization"].endswith("normed_euclidean"):
            trainer = BCTrainer(
                GaussianAgents(
                    num_layers=2,
                    d_model=64,
                    nhead=2,
                    norm_action_mean=(
                        config["action_parameterization"] == "normed_euclidean"
                    ),
                ),
                demo_filename="../../data/basic_demo.pickle",
                lr=config["lr"],
                minibatch_size=256,
                weight_decay=config["weight_decay"],
                grad_clipping=None,
            )
        else:
            trainer = BCTrainer(
                VonMisesAgents(num_layers=2, d_model=64, nhead=2),
                demo_filename="../../data/basic_demo.pickle",
                lr=config["lr"],
                minibatch_size=256,
                weight_decay=config["weight_decay"],
                grad_clipping=None,
            )

        for _ in tqdm.trange(100):
            trainer.train_epoch()
        trainer.evaluate()
        torch.save(
            trainer.agents.state_dict(),
            f"checkpoints/{config['action_parameterization']}_{run}.ckpt",
        )
        results.append(trainer.logger.cumulative_data["eval_goals_scored_mean"][-1])
    iqm = np.mean(sorted(results)[1:4])
    mean = np.mean(results)
    median = np.median(results)
    std = np.std(results)
    print(
        f"{config['action_parameterization']} results:\n"
        f"  > Using config: {config}\n"
        f"  - raw results: {results}\n"
        f"  - IQM: {iqm:.3f}\n"
        f"  - Mean: {mean:.3f}\n"
        f"  - Median: {median:.3f}\n"
        f"  - STD: {std:.3f}\n"
    )


def main():
    best_configs = {
        "von_mises": (
            {"lr": 1e-3, "weight_decay": 10**-4.5},
            {"lr": 1e-3, "weight_decay": 10**-4},
            {"lr": 10**-3.5, "weight_decay": 10**-4.5},
        ),
        "euclidean": (
            {"lr": 1e-3, "weight_decay": 10**-4.5},
            {"lr": 10**-3.5, "weight_decay": 10**-3.5},
            {"lr": 1e-3, "weight_decay": 1e-4},
        ),
        "normed_euclidean": (
            {"lr": 10**-3.5, "weight_decay": 1e-5},
            {"lr": 1e-3, "weight_decay": 1e-5},
            {"lr": 1e-3, "weight_decay": 10**-4.5},
        ),
    }
    for k, v in best_configs.items():
        for cfg in v:
            train(cfg | {"action_parameterization": k})


if __name__ == "__main__":
    main()
