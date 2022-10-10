import math
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import torch
from ray import tune


# adapted from SpinningUp PPO
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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
    def __init__(self):
        self.cumulative_data = defaultdict(list)
        self.epoch_data = defaultdict(list)

        self.cleared_previous = False

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.epoch_data[k].append(v)

    def step(self):
        for k, v in self.epoch_data.items():
            if len(v) == 1:
                self.cumulative_data[k].append(v[0])
            elif isinstance(v[0], bool):
                self.cumulative_data[k + "_prop"].append(np.mean(v, dtype=np.float32))
            else:
                self.cumulative_data[k + "_mean"].append(np.mean(v))
                self.cumulative_data[k + "_std"].append(np.std(v))

        self.epoch_data.clear()

    def update_from(self, other: "Logger"):
        for k, v in other.cumulative_data.items():
            self.cumulative_data[k].extend(v)

    def tune_report(self):
        tune.report(**{k: v[-1] for k, v in self.cumulative_data.items()})

    def to_df(self):
        return pd.DataFrame(self.cumulative_data)

    def generate_plots(self, fname_prefix=None):
        if not self.cleared_previous:
            folder = "plotgen/"
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
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
            if fname_prefix is None:
                fig.savefig(f"plotgen/{name}.png")
            else:
                if re.fullmatch("[a-zA-Z0-9]", fname_prefix[-1]):
                    fname_prefix += "_"
                fig.savefig(f"plotgen/{fname_prefix}{name}.png")
            plt.close(fig)


def grad_norm(module):
    with torch.no_grad():
        return torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2.0)
                    for p in module.parameters()
                    if p.grad is not None
                ]
            ),
            2.0,
        ).item()
