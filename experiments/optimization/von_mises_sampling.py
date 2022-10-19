import time

import torch
import torch.nn.functional as F
from torch.distributions import VonMises
from torch.special import i0

# data = torch.randn((4, 4))
#
# start = time.perf_counter()
# _log_modified_bessel_fn(data, order=0)
# print(time.perf_counter() - start)
#
# start = time.perf_counter()
# torch.log(i0(data))
# print(time.perf_counter() - start)

loc = torch.randn(4096)
conc = F.softplus(torch.randn(4096)) + 1e-3
vals = torch.randn(4096)

v1 = VonMises(loc, conc, validate_args=False)
start = time.perf_counter()
p1 = v1.log_prob(vals)
print(time.perf_counter() - start)


VonMises.log_prob = lambda self, value: (
    self.concentration * torch.cos(value - self.loc)
    - 1.83787706641
    - torch.log(i0(self.concentration))
)
v2 = VonMises(loc, conc, validate_args=False)
start = time.perf_counter()
p2 = v2.log_prob(vals)
print(time.perf_counter() - start)

print(torch.isclose(p1, p2).all())
