import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from architectures import Encoder
from experiments.action_parameterization.von_mises_upgrades import upgrade

upgrade()


class VonMisesAgents(nn.Module):
    def __init__(self, num_layers=1, d_model=32, nhead=2, dim_feedforward=64):
        super().__init__()

        self.policy_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.value_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

        self.move_head = nn.Linear(d_model, 3)
        self.throw_head = nn.Linear(d_model, 3)
        self.value_head = nn.Linear(d_model, 1)
        # self.aux_value_head = nn.Linear(d_model, 1)

        with torch.no_grad():
            self.move_head.weight *= 0.01
            self.throw_head.weight *= 0.01

        self._std_offset = torch.log(torch.exp(torch.tensor(0.5)) - 1)

    def step(self, obs: dict[str, np.array]):
        actions = {
            "id": np.zeros(2, dtype=np.int64),
            "target": np.zeros((2, 2), dtype=np.float32),
        }
        logps = np.zeros(2, dtype=np.float32)
        with torch.no_grad():
            z = self.policy_encoder(obs)  # S x 32
            for i in range(2):
                # index 0 is global embedding, 1 is agent 0, 2 is agent 1
                embed = z[i + 1]

                if obs[f"wizard{i}"][5] == 1:  # throw available
                    actions["id"][i] = 1
                    logits = self.throw_head(embed)
                else:
                    actions["id"][i] = 0
                    logits = self.move_head(embed)

                x = logits[0]
                y = logits[1]
                angle = torch.atan2(y, x)
                concentration = F.softplus(logits[2]) + 1e-3
                distr = distributions.VonMises(angle, concentration)

                action = distr.sample()
                logp = distr.log_prob(action)

                actions["target"][i] = (
                    torch.cos(action).item(),
                    torch.sin(action).item(),
                )
                logps[i] = logp.sum().item()

        return actions, logps

    def predict_value(self, obs, _actions):
        # We could use OTHER agent's action in the future, but for now we only use the
        # agent's own observation
        values = np.zeros(2, dtype=np.float32)
        with torch.no_grad():
            z = self.value_encoder(obs)  # S x 32

        for i in range(2):
            embed = z[i + 1]  # index 0 is global embedding, 1 is agent 0, 2 is agent 1
            values[i] = self.value_head(embed).item()
        return values

    def policy_forward(self, rollout, batch_idx):
        z = self.policy_encoder(rollout["obs"], batch_idx)  # S x B x 32
        distrs = []
        ret = torch.zeros((batch_idx.shape[0], 2))
        for i in range(2):
            embed = z[i + 1]
            logits = torch.zeros((len(batch_idx), 3))

            throw_turns = rollout["obs"][f"wizard{i}"][batch_idx, 5] == 1
            logits[throw_turns] = self.throw_head(embed[throw_turns])
            logits[~throw_turns] = self.move_head(embed[~throw_turns])

            actions_taken = rollout["act"]["target"][batch_idx, i]

            x = logits[:, 0]
            y = logits[:, 1]
            concentration = F.softplus(logits[:, 2]) + 1e-3
            distrs.append(distributions.VonMises(torch.atan2(y, x), concentration))

            angles = torch.atan2(actions_taken[:, 1], actions_taken[:, 0])
            ret[:, i] = distrs[i].log_prob(angles)

        return ret, distrs

    def value_forward(self, rollout, batch_idx):
        z = self.value_encoder(rollout["obs"], batch_idx)  # S x B x 32
        ret = torch.zeros((batch_idx.shape[0], 2))
        for i in range(2):
            embed = z[i + 1]  # B x 32
            ret[:, i] = self.value_head(embed).squeeze(1)  # B x 1  ->  B
        return ret


def main():
    import tqdm

    from behavioral_cloning import BCTrainer

    trainer = BCTrainer(
        VonMisesAgents(),
        demo_filename="../../data/basic_demo.pickle",
        lr=1e-3,
        minibatch_size=128,
        weight_decay=1e-5,
        grad_clipping=10.0,
    )
    for _ in tqdm.trange(20):
        trainer.train()
    trainer.evaluate_with_render()


if __name__ == "__main__":
    main()
