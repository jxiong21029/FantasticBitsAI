import warnings

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from env import SZ_BLUDGER, SZ_GLOBAL, SZ_SNAFFLE, SZ_WIZARD


class Encoder(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()

        self.d_model = d_model

        self.global_prep = nn.Linear(SZ_GLOBAL, d_model)
        self.wizard_prep = nn.Linear(SZ_WIZARD, d_model)
        self.snaffle_prep = nn.Linear(SZ_SNAFFLE, d_model)
        self.bludger_prep = nn.Linear(SZ_BLUDGER, d_model)

        norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=nhead, dim_feedforward=64, dropout=0
            ),
            num_layers=1,
            norm=norm,
        )

    def forward(self, obs, batch_idx=None):
        if batch_idx is None:
            x = torch.zeros((len(obs), 1, self.d_model))
            for i, (k, v) in enumerate(obs.items()):
                if k.startswith("global"):
                    x[i, :] = self.global_prep(torch.tensor(v))
                elif k.startswith("wizard"):
                    x[i, :] = self.wizard_prep(torch.tensor(v))
                elif k.startswith("snaffle"):
                    x[i, :] = self.snaffle_prep(torch.tensor(v))
                elif k.startswith("bludger"):
                    x[i, :] = self.bludger_prep(torch.tensor(v))
                else:
                    warnings.warn(f"unexpected key: {k}")
            ret = self.encoder(x).squeeze(dim=1)  # S x 32
            # if ret.isnan().any():
            #     raise ValueError
            return ret
        else:
            x = torch.zeros((len(obs), len(batch_idx), self.d_model))  # S x B x 32

            # B x S
            padding_mask = torch.zeros((len(batch_idx), len(obs)), dtype=bool)

            for i, (k, v) in enumerate(obs.items()):
                # shape B, F; F=4 (snaffle), 6 (wizard), etc..
                entry = v[batch_idx]
                if k.startswith("snaffle"):
                    if (
                        not torch.equal(entry.isnan()[:, 0], entry.isnan()[:, 1])
                        or not torch.equal(entry.isnan()[:, 0], entry.isnan()[:, 2])
                        or not torch.equal(entry.isnan()[:, 0], entry.isnan()[:, 3])
                    ):
                        raise ValueError
                    padding_mask[entry.isnan()[:, 0], i] = 1
                    entry = entry.clone()
                    entry[entry.isnan()] = 0
                elif entry.isnan().any():
                    raise ValueError

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

            if ret.isnan().any():
                raise ValueError

            return ret


class Agents(nn.Module):
    def __init__(self, d_model=32, nhead=2):
        super().__init__()

        self.policy_encoder = Encoder(d_model, nhead)
        self.value_encoder = Encoder(d_model, nhead)

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
        z = self.policy_encoder(rollout["obs"], batch_idx)  # S x B x 32
        distrs = []
        ret = torch.zeros((batch_idx.shape[0], 2))
        for i in range(2):
            embed = z[i + 1]
            logits = torch.zeros((len(batch_idx), 4))

            throw_turns = rollout["obs"][f"wizard{i}"][batch_idx, 5] == 1
            logits[throw_turns] = self.throw_head(embed[throw_turns])
            logits[~throw_turns] = self.move_head(embed[~throw_turns])

            mu = logits[:, :2]
            sigma = F.softplus(
                logits[:, 2:] + self._std_offset.to(device=logits.device)
            )
            distrs.append(distributions.Normal(mu, sigma, validate_args=False))

            actions_taken = rollout["act"]["target"][batch_idx, i]
            ret[:, i] = distrs[i].log_prob(actions_taken).sum(dim=1)
        return ret, distrs

    def value_forward(self, rollout, batch_idx):
        z = self.value_encoder(rollout["obs"], batch_idx)  # S x B x 32
        ret = torch.zeros((batch_idx.shape[0], 2))
        for i in range(2):
            embed = z[i + 1]  # B x 32
            ret[:, i] = self.value_head(embed).squeeze(1)  # B x 1  ->  B
        return ret

    def aux_value_forward(self, rollout, batch_idx):
        pass
