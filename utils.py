import math
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
from ray import tune
from ray.air import session
from scipy.signal import lfilter

matplotlib.use("Tkagg")
seaborn.set_theme()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# adapted from SpinningUp PPO
def discount_cumsum(x, discount):
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class RunningMoments:
    """
    Tracks running mean and variance
    Adapted from github.com/MadryLab/implementation-matters, which took it from
    github.com/joschu/modular_rl. Math in johndcook.com/blog/standard_deviation
    """

    def __init__(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def push(self, x):
        assert isinstance(x, float) or isinstance(x, int)
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            old_m = self.m
            self.m = old_m + (x - old_m) / self.n
            self.s = self.s + (x - old_m) * (x - self.m)

    def mean(self):
        return self.m

    def std(self):
        if self.n > 1:
            return math.sqrt(self.s / (self.n - 1))
        else:
            return self.m


class Logger:
    def __init__(self, logging_prefix=None):
        self.cumulative_data = defaultdict(list)
        self.epoch_data = defaultdict(list)
        self.prefix = logging_prefix

        self.cleared_previous = False

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.epoch_data[k].append(v)

    def step(self):
        pre = "" if self.prefix is None else self.prefix
        for k, v in self.epoch_data.items():
            if len(v) == 1:
                self.cumulative_data[k].append(v[0])
            elif isinstance(v[0], bool):
                self.cumulative_data[pre + k + "_prop"].append(
                    np.mean(v, dtype=np.float32)
                )
            else:
                self.cumulative_data[pre + k + "_mean"].append(np.mean(v))
                self.cumulative_data[pre + k + "_std"].append(np.std(v))

        self.epoch_data.clear()

    def update_from(self, other: "Logger"):
        for k, v in other.cumulative_data.items():
            self.cumulative_data[k].extend(v)

    def tune_report(self):
        tune.report(**{k: v[-1] for k, v in self.cumulative_data.items()})

    def air_report(self, **kwargs):
        session.report({k: v[-1] for k, v in self.cumulative_data.items()}, **kwargs)

    def to_df(self):
        return pd.DataFrame(self.cumulative_data)

    def generate_plots(self, dirname="plotgen"):
        if not self.cleared_previous:
            if os.path.isdir(dirname):
                for filename in os.listdir(dirname):
                    file_path = os.path.join(dirname, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
            else:
                os.mkdir(dirname)
            self.cleared_previous = True

        for k, v in self.cumulative_data.items():
            if k.endswith("_std"):
                continue

            fig, ax = plt.subplots()

            x = np.arange(len(self.cumulative_data[k]))
            v = np.array(v)
            if k.endswith("_mean"):
                name = k[:-5]

                (line,) = ax.plot(x, v, label=k)
                stds = np.array(self.cumulative_data[name + "_std"])
                ax.fill_between(
                    x, v - stds, v + stds, color=line.get_color(), alpha=0.15
                )
            else:
                name = k
                ax.plot(x, v)
            fig.suptitle(name)
            fig.savefig(os.path.join(dirname, name))
            plt.close(fig)

    def estimate_convergence(self, key, smoothing=0.9, eps=1e-8):
        assert key in self.cumulative_data

        m = 0
        v = 0
        data = self.cumulative_data[key]
        if len(data) <= 1:
            return 0

        diffs = [data[i + 1] - data[i] for i in range(len(data) - 1)]

        for d in diffs:
            m = smoothing * m + (1 - smoothing) * d
            v = smoothing * v + (1 - smoothing) * d * d

        mh = m / (1 - smoothing ** len(diffs))
        vh = v / (1 - smoothing ** len(diffs))
        return 1 - abs(mh) / (math.sqrt(vh) + eps)


def grad_norm(module):
    with torch.no_grad():
        if components := [
            torch.norm(p.grad.detach(), 2.0)
            for p in module.parameters()
            if p.grad is not None
        ]:
            return torch.norm(
                torch.stack(components),
                2.0,
            ).item()
        return 0


def component_grad_norms(module, exclude=None):
    total_norm = grad_norm(module)
    component_norms = {}

    names = set(name.split(".")[0] for name, _ in module.named_parameters())
    for name in names:
        if exclude is not None and name in exclude:
            continue
        component_norms[name] = grad_norm(module.__getattr__(name))

    component_norms.update(total=total_norm)
    return component_norms


def profileit(func):
    import cProfile

    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile"  # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper
