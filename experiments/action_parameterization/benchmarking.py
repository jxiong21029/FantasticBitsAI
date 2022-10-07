import time

import torch
import torch.distributions as distributions


def sampling_benchmark():
    mu = 2 * torch.rand(4096, 2) - 1
    sigma = torch.rand(4096, 2) / 2
    start = time.perf_counter()
    for i in range(4096):
        distr = distributions.Normal(mu[i], sigma[i])
        distr.sample()
    end = time.perf_counter()
    print(f"euclidean sampling, default: {end - start:.3f}s")

    start = time.perf_counter()
    for i in range(4096):
        normed = mu / torch.norm(mu, dim=1, keepdim=True)
        distr = distributions.Normal(normed, sigma[i])
        distr.sample()
    end = time.perf_counter()
    print(f"euclidean sampling, normed: {end - start:.3f}s")

    locs = 2 * torch.pi * torch.rand(4096)
    concs = 0.5 + torch.rand(4096)
    start = time.perf_counter()
    for i in range(4096):
        distr = distributions.VonMises(locs[i], concs[i])
        distr.sample()
    end = time.perf_counter()
    print(f"von Mises sampling, from angle: {end - start:.3f}s")

    start = time.perf_counter()
    for i in range(4096):
        distr = distributions.VonMises(torch.atan2(mu[i, 1], mu[i, 0]), concs[i])
        distr.sample()
    end = time.perf_counter()
    print(f"von Mises sampling, from coords: {end - start:.3f}s")


def sample_logp_benchmark():
    mu = 2 * torch.rand(4096, 2) - 1
    sigma = torch.rand(4096, 2) / 2
    start = time.perf_counter()
    for i in range(4096):
        distr = distributions.Normal(mu[i], sigma[i])
        distr.log_prob(distr.sample())
    end = time.perf_counter()
    print(f"euclidean logp, default: {end - start:.3f}s")

    start = time.perf_counter()
    for i in range(4096):
        normed = mu[i] / torch.norm(mu[i], dim=0, keepdim=True)
        distr = distributions.Normal(normed, sigma[i])
        distr.log_prob(distr.sample())
    end = time.perf_counter()
    print(f"euclidean logp, normed: {end - start:.3f}s")

    locs = 2 * torch.pi * torch.rand(4096)
    concs = 0.5 + torch.rand(4096)
    start = time.perf_counter()
    for i in range(4096):
        distr = distributions.VonMises(locs[i], concs[i])
        distr.log_prob(distr.sample())
    end = time.perf_counter()
    print(f"von Mises logp, from angle: {end - start:.3f}s")

    start = time.perf_counter()
    for i in range(4096):
        distr = distributions.VonMises(torch.atan2(mu[i, 1], mu[i, 0]), concs[i])
        distr.log_prob(distr.sample())
    end = time.perf_counter()
    print(f"von Mises logp, from coords: {end - start:.3f}s")


def logp_backprop_benchmark():
    actions = torch.randn(4096, 2)

    mu = 2 * torch.rand(4096, 2) - 1
    sigma = torch.rand(4096, 2) / 2
    start = time.perf_counter()
    for i in range(32):
        batch_mu = torch.tensor(mu[128 * i : 128 * i + 128], requires_grad=True)
        batch_sigma = torch.tensor(sigma[128 * i : 128 * i + 128], requires_grad=True)
        distr = distributions.Normal(batch_mu, batch_sigma)
        distr.log_prob(actions[128 * i : 128 * i + 128]).mean().backward()
    end = time.perf_counter()
    print(f"euclidean backprop, default: {end - start:.3f}s")

    start = time.perf_counter()
    for i in range(32):
        batch_mu = torch.tensor(mu[128 * i : 128 * i + 128], requires_grad=True)
        batch_sigma = torch.tensor(sigma[128 * i : 128 * i + 128], requires_grad=True)
        normed = batch_mu / torch.norm(batch_mu, dim=1, keepdim=True)
        distr = distributions.Normal(normed, batch_sigma)
        distr.log_prob(actions[128 * i : 128 * i + 128]).mean().backward()
    end = time.perf_counter()
    print(f"euclidean backprop, normed: {end - start:.3f}s")

    locs = 2 * torch.pi * torch.rand(4096)
    concs = 0.5 + torch.rand(4096)
    start = time.perf_counter()
    for i in range(32):
        batch_loc = torch.tensor(locs[128 * i : 128 * i + 128], requires_grad=True)
        batch_conc = torch.tensor(concs[128 * i : 128 * i + 128], requires_grad=True)
        distr = distributions.VonMises(batch_loc, batch_conc)
        angles = torch.atan2(
            actions[128 * i : 128 * i + 128, 1], actions[128 * i : 128 * i + 128, 0]
        )
        distr.log_prob(angles).mean().backward()
    end = time.perf_counter()
    print(f"von Mises backprop, from angle: {end - start:.3f}s")

    mu = 2 * torch.rand(4096, 2) - 1
    concs = 0.5 + torch.rand(4096)
    start = time.perf_counter()
    for i in range(32):
        batch_mu = torch.tensor(mu[128 * i : 128 * i + 128], requires_grad=True)
        batch_conc = torch.tensor(concs[128 * i : 128 * i + 128], requires_grad=True)
        distr = distributions.VonMises(
            torch.atan2(batch_mu[:, 1], batch_mu[:, 0]), batch_conc
        )
        angles = torch.atan2(
            actions[128 * i : 128 * i + 128, 1], actions[128 * i : 128 * i + 128, 0]
        )
        distr.log_prob(angles).mean().backward()
    end = time.perf_counter()
    print(f"von Mises backprop, from coords: {end - start:.3f}s")


sampling_benchmark()
sample_logp_benchmark()
logp_backprop_benchmark()
