import warnings

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import VonMises, register_kl

from env import SZ_BLUDGER, SZ_GLOBAL, SZ_SNAFFLE, SZ_WIZARD

ordered_keys = (
    ("global",)
    + tuple(f"wizard{i}" for i in range(4))
    + tuple(f"snaffle{i}" for i in range(7))
    + tuple(f"bludger{i}" for i in range(2))
)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        self.d_model = d_model

        self.global_prep = nn.Linear(SZ_GLOBAL, d_model)
        self.wizard_prep = nn.Linear(SZ_WIZARD, d_model)
        self.snaffle_prep = nn.Linear(SZ_SNAFFLE, d_model)
        self.bludger_prep = nn.Linear(SZ_BLUDGER, d_model)

        norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
            ),
            num_layers=num_layers,
            norm=norm,
        )

    @property
    def device(self):
        return self.global_prep.weight.device

    def forward(self, obs, batch_idx=None, flip_augment=None):
        for k in ordered_keys:
            assert k in obs or k.startswith("snaffle")
        if batch_idx is None:
            x = torch.zeros((len(obs), 1, self.d_model), device=self.device)
            for i, k in enumerate(k for k in ordered_keys if k in obs):
                v = obs[k]
                if k.startswith("global"):
                    x[i, :] = self.global_prep(torch.tensor(v, device=self.device))
                elif k.startswith("wizard"):
                    x[i, :] = self.wizard_prep(torch.tensor(v, device=self.device))
                elif k.startswith("snaffle"):
                    x[i, :] = self.snaffle_prep(torch.tensor(v, device=self.device))
                elif k.startswith("bludger"):
                    x[i, :] = self.bludger_prep(torch.tensor(v, device=self.device))
                else:
                    warnings.warn(f"unexpected key: {k}")
            ret = self.encoder(x).squeeze(dim=1)  # S x 32
            return ret
        else:
            # S x B x 32
            x = torch.zeros(
                (len(obs), len(batch_idx), self.d_model), device=self.device
            )
            # B x S
            padding_mask = torch.zeros(
                (len(batch_idx), len(obs)), dtype=bool, device=self.device
            )

            for i, k in enumerate(k for k in ordered_keys if k in obs):
                entry = obs[k][batch_idx]
                # shape B, F; F=4 (snaffle), 6 (wizard), etc..

                if k.startswith("snaffle") or (flip_augment and not k == "global"):
                    entry = entry.clone()

                if k.startswith("snaffle"):
                    padding_mask[entry.isnan()[:, 0], i] = 1
                    entry[entry.isnan()] = 0

                if flip_augment and not k == "global":
                    # invert all y positions and y velocities
                    feat = [1, 3, 5, 7] if k.startswith("bludger") else [1, 3]
                    entry[np.arange(len(batch_idx) // 2)[:, None], feat] *= -1

                if k.startswith("global"):
                    x[i, :] = self.global_prep(entry)
                elif k.startswith("wizard"):
                    x[i, :] = self.wizard_prep(entry)
                elif k.startswith("snaffle"):
                    x[i, :] = self.snaffle_prep(entry)
                elif k.startswith("bludger"):
                    x[i, :] = self.bludger_prep(entry)
                else:
                    warnings.warn(f"unexpected key: {k}")

            ret = self.encoder(x, src_key_padding_mask=padding_mask)  # S x B x 32

            return ret


class GaussianAgents(nn.Module):
    def __init__(
        self,
        num_layers=1,
        d_model=32,
        nhead=2,
        dim_feedforward=64,
        norm_action_mean=False,
        dropout=0,
        flip_augment=True,
    ):
        super().__init__()
        self.norm_action_mean = norm_action_mean
        self.flip_augment = flip_augment

        self.policy_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.value_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.move_head = nn.Linear(d_model, 4)
        self.throw_head = nn.Linear(d_model, 4)
        self.value_head = nn.Linear(d_model, 1)
        self.aux_value_head = nn.Linear(d_model, 1)

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

                mu = logits[:2]
                if self.norm_action_mean == "normed_euclidean":
                    mu = mu / torch.norm(mu)
                sigma = F.softplus(
                    logits[2:] + self._std_offset.to(device=logits.device)
                )
                distr = distributions.Normal(mu, sigma, validate_args=False)

                action = distr.sample()
                logp = distr.log_prob(action)

                actions["target"][i] = action.cpu().numpy()
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
        # S x B x 32
        z = self.policy_encoder(rollout["obs"], batch_idx, self.flip_augment)
        distrs = []
        ret = torch.zeros((batch_idx.shape[0], 2), device=self.device)
        for i in range(2):
            embed = z[i + 1]
            logits = torch.zeros((len(batch_idx), 4), device=self.device)

            throw_turns = rollout["obs"][f"wizard{i}"][batch_idx, 5] == 1
            logits[throw_turns] = self.throw_head(embed[throw_turns])
            logits[~throw_turns] = self.move_head(embed[~throw_turns])

            actions_taken = rollout["act"]["target"][batch_idx, i]

            mu = logits[:, :2]
            if self.norm_action_mean:
                mu = mu / torch.norm(mu, dim=1, keepdim=True)
            if self.flip_augment:
                mu[: len(batch_idx) // 2, 1] *= -1
            sigma = F.softplus(
                logits[:, 2:] + self._std_offset.to(device=logits.device)
            )
            distrs.append(distributions.Normal(mu, sigma, validate_args=False))

            ret[:, i] = distrs[i].log_prob(actions_taken).sum(dim=1)

        return ret, distrs

    def value_forward(self, rollout, batch_idx):
        # S x B x 32
        z = self.value_encoder(rollout["obs"], batch_idx, self.flip_augment)
        ret = torch.zeros((batch_idx.shape[0], 2), device=self.device)
        for i in range(2):
            embed = z[i + 1]  # B x 32
            ret[:, i] = self.value_head(embed).squeeze(1)  # B x 1  ->  B
        return ret


VonMises.entropy = (
    lambda self: self.concentration
    * (
        1
        - torch.special.i1e(self.concentration)
        / (i0e := torch.special.i0e(self.concentration))
    )
    + torch.log(i0e)
    + 1.83787706641
)


@register_kl(VonMises, VonMises)
def kl_vonmises_vonmises(p, q):
    i0e_concentration1 = torch.special.i0e(p.concentration)
    i1e_concentration1 = torch.special.i1e(p.concentration)
    i0e_concentration2 = torch.special.i0e(q.concentration)
    return (
        (q.concentration - p.concentration)
        + torch.log(i0e_concentration2 / i0e_concentration1)
        + (p.concentration - q.concentration * torch.cos(p.loc - q.loc))
        * (i1e_concentration1 / i0e_concentration1)
    )


class VonMisesAgents(nn.Module):
    def __init__(
        self,
        num_layers=1,
        d_model=32,
        nhead=2,
        dim_feedforward=64,
        dropout=0,
        flip_augment=True,
    ):
        super().__init__()
        self.flip_augment = flip_augment

        self.policy_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.value_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.move_head = nn.Linear(d_model, 3)
        self.throw_head = nn.Linear(d_model, 3)
        self.value_head = nn.Linear(d_model, 1)
        # self.aux_value_head = nn.Linear(d_model, 1)

        with torch.no_grad():
            self.move_head.weight *= 0.01
            self.throw_head.weight *= 0.01

    @property
    def device(self):
        return self.move_head.weight.device

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
                distr = distributions.VonMises(
                    angle, concentration, validate_args=False
                )

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
        # S x B x 32
        z = self.policy_encoder(rollout["obs"], batch_idx, self.flip_augment)
        distrs = []
        ret = torch.zeros((batch_idx.shape[0], 2), device=self.device)
        for i in range(2):
            embed = z[i + 1]
            logits = torch.zeros((len(batch_idx), 3), device=self.device)

            throw_turns = rollout["obs"][f"wizard{i}"][batch_idx, 5] == 1
            logits[throw_turns] = self.throw_head(embed[throw_turns])
            logits[~throw_turns] = self.move_head(embed[~throw_turns])

            actions_taken = rollout["act"]["target"][batch_idx, i]

            x = logits[:, 0]
            y = logits[:, 1]
            if self.flip_augment:
                y[: len(batch_idx) // 2] *= -1
            concentration = F.softplus(logits[:, 2]) + 1e-3
            distrs.append(
                distributions.VonMises(
                    torch.atan2(y, x), concentration, validate_args=False
                )
            )

            angles = torch.atan2(actions_taken[:, 1], actions_taken[:, 0])
            ret[:, i] = distrs[i].log_prob(angles)

        return ret, distrs

    def value_forward(self, rollout, batch_idx):
        # S x B x 32
        z = self.value_encoder(rollout["obs"], batch_idx, self.flip_augment)
        ret = torch.zeros((batch_idx.shape[0], 2), device=self.device)
        for i in range(2):
            embed = z[i + 1]  # B x 32
            ret[:, i] = self.value_head(embed).squeeze(1)  # B x 1  ->  B
        return ret
