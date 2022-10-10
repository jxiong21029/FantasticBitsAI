import torch
from torch.distributions import VonMises, register_kl


def upgrade():
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
