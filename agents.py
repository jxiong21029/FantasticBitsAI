import numpy as np

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.global_prep = nn.Linear(4, d_model)
        self.entity_prep = nn.Linear(9, d_model)
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=nhead, dim_feedforward=128, dropout=0
            ),
            num_layers=2,
        )

    def forward(self, obs, batch_idx=None):
        if batch_idx is None:
            assert obs["global_0"].shape == (4,)
            xg = torch.stack(
                [
                    torch.tensor(obs["global_0"]),  # 4
                    torch.tensor(obs["global_1"]),  # 4
                ],
                dim=0,
            )  # 2 x 4
            xg = self.global_prep(xg).unsqueeze(0)  # 1 x 2 x 32

            # S x 9
            xe0 = torch.stack([torch.tensor(entry) for entry in obs["entities_0"]])
            xe1 = torch.stack([torch.tensor(entry) for entry in obs["entities_1"]])

            xe = torch.stack([xe0, xe1], dim=1)  # S x 2 x 9
            xe = self.entity_prep(xe)  # S x 2 x 32

            xc = torch.cat([xg, xe], dim=0)  # (S + 1) x 2 x 32
            z = self.backbone(xc)  # (S + 1) x 2 x 32  --  seq, agent, feat
            return torch.permute(z, (1, 0, 2))  # agent, seq, feat
        else:
            assert obs["global_0"].shape[1:] == (4,)
            xg0 = obs["global_0"][batch_idx]  # B x 4
            xg1 = obs["global_1"][batch_idx]  # B x 4
            xg0 = self.global_prep(xg0).unsqueeze(0)  # 1 x B x 32
            xg1 = self.global_prep(xg1).unsqueeze(0)  # 1 x B x 32

            xe0 = obs["entities_0"][batch_idx]  # B x S x 9
            xe1 = obs["entities_1"][batch_idx]  # B x S x 9
            xe0 = self.entity_prep(xe0)  # B x S x 32
            xe1 = self.entity_prep(xe1)  # B x S x 32
            xe0 = torch.permute(xe0, (1, 0, 2))  # S x B x 32
            xe1 = torch.permute(xe1, (1, 0, 2))  # S x B x 32

            xc0 = torch.cat([xg0, xe0], dim=0)  # (S+1) x B x 32
            xc1 = torch.cat([xg1, xe1], dim=0)  # (S+1) x B x 32

            z0 = self.backbone(xc0)  # (S+1) x B x 32
            z1 = self.backbone(xc1)  # (S+1) x B x 32
            z = torch.stack([z0, z1], dim=0)  # 2 x (S+1) x B x 32
            return torch.permute(z, (0, 2, 1, 3))  # agent, batch, seq, feat


# agent.step(single obs) -> action, value, logp
# agent.predict_values(batch obs) -> values (2xB)
# agent.logp(batch obs) -> logps
class Agents(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()

        self.policy_encoder = Encoder(d_model, nhead)
        self.value_encoder = Encoder(d_model, nhead)

        self.move_head = nn.Linear(d_model, 4)
        self.throw_head = nn.Linear(d_model, 4)
        self.value_head = nn.Linear(d_model, 1)

        self._std_offset = torch.log(torch.exp(torch.tensor(0.5)) - 1)

    def step(self, obs):  # -> action, logp
        with torch.no_grad():
            z = self.policy_encoder(obs)  # 2 x S x 32
        action = {}
        logp = np.zeros(shape=2)
        for i in range(2):
            if obs[f"global_{i}"][3] == 1:  # holding snaffle
                logits = self.throw_head(z[i]).mean(dim=0)
            else:
                logits = self.move_head(z[i]).mean(dim=0)
            # logits: tensor (4,)
            mu = logits[:2]
            sigma = F.softplus(logits[2:] + self._std_offset.to(device=logits.device))
            distr = distributions.TransformedDistribution(
                distributions.Normal(mu, sigma, validate_args=False),
                [distributions.TanhTransform()],
                validate_args=False,
            )

            # 1 if holding snaffle else 0
            act_id = int(obs[f"wizard_{i}"]["global"][3].item())
            act_target = distr.sample()
            logp[i] = distr.log_prob(act_target)

            action[f"id_{i}"] = act_id
            action[f"target_{i}"] = act_target.cpu().numpy()
        return action, logp

    def predict_value(self, obs, _actions):
        # We could use OTHER agent's action in the future, but for now we only use the
        # agent's own observation
        with torch.no_grad():
            z = self.value_encoder(obs)  # 2 x S x 32
        pooled = z.mean(dim=1)
        return self.value_head(pooled).reshape(2)

    def logp(self, rollout, batch_idx):
        pass

    def value_forward(self, rollout, batch_idx):
        pass

