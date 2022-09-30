import itertools
import math
from collections import deque
from fractions import Fraction
from typing import Callable

import ray


# convenience functions
def log_binary_search(*values):
    return {"log_binary_search": values}


def grid_search(*values):
    return {"grid_search": values}


@ray.remote
class Worker:
    def __init__(self):
        pass


class Tuner:
    """
    Expected usage:
    tuner = Tuner(
        Trainer,
        search_space={
            "lr": log_binary_search(1e-3, 1e-2, 1e-1),
            "batch_size": grid_search(32, 64),
            "weight_decay": 1e-4,
        },
        metric="eval_mean_goals",
        mode="max",
    )
    tuner.fit(depth=2)
    """

    def __init__(self, trainer: Callable, search_space: dict, metric: str, mode: str):
        self.trainer_fn = trainer
        self.search_space = {}

        self.metric = metric
        if mode not in ("min", "max"):
            raise ValueError(f"expected mode 'min' or 'max', but received {mode=}")
        self.mode = mode

        self.log_binary_spaces = set()

        for k, v in search_space.items():
            if isinstance(v, dict):
                if len(v) != 1:
                    raise TypeError
                space_type = list(v.keys())[0]
                self.search_space[k] = v[space_type]

                if space_type == "log_binary_search":
                    self.log_binary_spaces.add(k)
                    if len(v[space_type]) <= 1 or len(v[space_type]) % 2 == 0:
                        raise ValueError(
                            f"log binary search specified with n={len(v[space_type])} "
                            f"values, but expected n to be both odd and >= 3"
                        )
                else:
                    raise ValueError(f"unexpected search space: {space_type}")
            else:
                self.search_space[k] = [v]

        for k in self.log_binary_spaces:
            vals = self.search_space[k]
            exponents = [Fraction(math.log10(val)).limit_denominator(2) for val in vals]

            diffs = [exponents[i + 1] - exponents[i] for i in range(len(exponents) - 1)]
            if min(diffs) != max(diffs):
                raise ValueError(
                    f"expected evenly spaced (logarithmically speaking) specification "
                    f"for log binary search, but received exponents {exponents} for "
                    f"key {k}"
                )

            self.search_space[k] = exponents

        self.candidates = deque()
        self.next_suggester = 0

    @staticmethod
    def _hashabled(config):
        return [(k, v) for k, v in sorted(config.items())]

    def suggest(self, config):
        nearest = {}
        for k in self.log_binary_spaces:
            pass

        #    x       x       x       x       x
        #    0       4       8      12      16
        #        2       6      10      14
        #      1   3   5   7   9  11  13  15

        for spec in itertools.product(
            *(range(len(v)) for v in self.search_space.values())
        ):
            new_config = {}
            for k, i in zip(self.search_space.keys(), spec):
                if k in self.log_binary_spaces:
                    j = i - (len(self.search_space[k]) // 2)
                else:
                    new_config[k] = self.search_space[k][i]

    # should always be able to suggest a potential config
    # should be able to early-stop configs which are no longer near leader of prev depth

    def fit(self, depth):
        pass
