import numpy as np
import torch
import torch.nn as nn

model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=6, nhead=2, dim_feedforward=64, dropout=0),
    2,
)

x = torch.randn((14, 2, 6))  # S x B x F

mask = torch.zeros((2, 14), dtype=bool)
mask[0, 2] = True

a = model(x, src_key_padding_mask=mask)
x[2, 0] = 0
b = model(x, src_key_padding_mask=mask)
# print(a)
# print(b)
print(
    "first batch unaffected eq:",
    torch.equal(a[:2, 0], b[:2, 0]) and torch.equal(a[3:, 0], b[3:, 0]),
)
print("first batch affected eq:", torch.equal(a[2, 0], b[2, 0]))
print("second batch eq:", torch.equal(a[:, 1], b[:, 1]))
