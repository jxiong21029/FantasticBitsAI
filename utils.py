import math
from collections import defaultdict
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


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

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.epoch_data[k].append(v)

    def step(self):
        seen = {k: False for k in self.cumulative_data.keys()}
        for k, v in self.epoch_data.items():
            if len(v) == 1:
                self.cumulative_data[k].append(v)
                seen[k] = True
                continue

            self.cumulative_data[k + "_mean"].append(np.mean(v))
            seen[k + "_mean"] = True
            self.cumulative_data[k + "_std"].append(np.std(v))
            seen[k + "_std"] = True

        for k, v in seen.items():
            if not v:
                self.cumulative_data[k].append(np.nan)

        self.epoch_data.clear()

    def generate_plots(self, fname_prefix=None):
        x = np.arange(len(self.cumulative_data[list(self.cumulative_data.keys())[0]]))

        for k, v in self.cumulative_data.items():
            if k.endswith("_std"):
                continue

            fig, ax = plt.subplots()
            v = np.array(v)
            if k.endswith("_mean"):
                name = k[:-5]

                ax.plot(x, v)
                stds = np.array(self.cumulative_data[name + "_std"])
                ax.fill_between(x, v - stds, v + stds)
            else:
                name = k
                ax.plot(x, v)
            fig.suptitle(name.title())
            if fname_prefix is None:
                fig.savefig(f"plotgen/{name}.png")
            else:
                if re.fullmatch("[a-zA-Z0-9]", fname_prefix[-1]):
                    fname_prefix += "_"
                fig.savefig(f"plotgen/{fname_prefix}{name}.png")
