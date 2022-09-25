import torch
import torch.nn as nn


class Agents:
    def __init__(self, d_model=32, nhead=4):
        self.d_model = d_model
        self.global_head = nn.Linear(3, d_model)
        self.entity_head = nn.Linear(9, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=nhead, dim_feedforward=128, dropout=0
            ),
            num_layers=3,
        )

    def forward(self, obs):
        xg = torch.tensor(obs["wizard_0"]["global"])
        xg = self.global_head(xg.reshape(1, -1))

        xe = obs["wizard_0"]["entities"]
        xe = torch.stack([torch.tensor(entry) for entry in xe])
        xe = self.entity_head(xe)

        xc = torch.cat([xg, xe], dim=0).reshape(-1, 1, self.d_model)
        out = self.encoder(xc)[1]


Agents().forward(torch.tensor([0.1, 0.2, 0.3]))
