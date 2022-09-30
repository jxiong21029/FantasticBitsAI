import itertools
import math
from collections import deque
from fractions import Fraction

from ray.tune.search import Searcher


def log_binary_search(*values):
    return {"log_binary_search": values}


def grid_search(*values):
    return {"grid_search": values}


class LogBinarySearch(Searcher):
    def __init__(self, search_space, depth, metric, mode, groups=None):
        """
        expected input format example:
        search_space = {
            "lr": log_binary_search(1e-3, 1e-2, 1e-1),
            "batch_size": grid_search(32, 64, 128),
            "weight_decay": 1e-5,
        }
        """

        super(LogBinarySearch, self).__init__(metric=metric, mode=mode)

        self.groups = groups

        self.max_depth = depth
        self.grid_searches = {}
        self.log_bin_scales = {}
        self.log_bin_sizes = {}

        keys = []
        start_values = []

        for k, v in search_space.items():
            keys.append(k)
            if not isinstance(v, dict):
                self.grid_searches[k] = [v]
                start_values.append([v])
            else:
                assert len(v) == 1

                space_type = list(v.keys())[0]
                vals = v[space_type]

                if space_type == "grid_search":
                    self.grid_searches[k] = v[space_type]
                    start_values.append(vals)
                elif space_type == "log_binary_search":
                    if len(vals) <= 1 or len(vals) % 2 == 0:
                        raise ValueError(
                            f"log binary search specified with n={len(vals)} "
                            f"values, but expected n to be both odd and >= 3"
                        )
                    start_values.append(vals)

                    exponents = [
                        Fraction(math.log10(val)).limit_denominator(2) for val in vals
                    ]
                    diffs = [
                        exponents[i + 1] - exponents[i]
                        for i in range(len(exponents) - 1)
                    ]
                    if (diff := min(diffs)) != max(diffs):
                        raise ValueError(
                            f"expected evenly spaced (logarithmically speaking) "
                            f"specification for log binary search, but received "
                            f"exponents {exponents} for key {k}"
                        )
                    else:
                        self.log_bin_scales[k] = diff

                    self.log_bin_sizes[k] = len(vals) // 2
                else:
                    raise ValueError(f"unexpected search space: {space_type}")

        self.candidates = deque()
        for cfg in itertools.product(start_values):
            self.candidates.append({k: v for k, v in zip(keys, cfg)})

        self.best_configs = [None for _ in range(self.max_depth)]
        self.seen = set()

    def _neighbors_of(self, config, new_depth):
        keys = []
        values = []

        for k, v in self.grid_searches.items():
            keys.append(k)
            values.append(v)

        for k, p in self.log_bin_scales.items():
            keys.append(k)
            values.append(
                [
                    10 ** (config[k] + c * self.log_bin_scales[k] / (2**new_depth))
                    for c in range(-self.log_bin_sizes[k], self.log_bin_sizes[k] + 1)
                ]
            )
            assert len(values) == self.log_bin_sizes[k] * 2 + 1  # TODO remove

        for cfg in itertools.product(values):
            yield {k: v for k, v in zip(keys, cfg)}

    def suggest(self, trial_id):
        # TODO: rewrite with SearchNode, generate new nodes from candidates
        # TODO: use a refcount dictionary, which serves secondary purpose as seen
        # TODO: config hash is just hash of values (with a consistent ordering)
        # TODO: make sure that children recurse failures to their children
        # TODO: mitigate bias by randomly shuffling within each depth
        if len(self.candidates) == 0:
            for d in range(self.max_depth - 1):
                found = False
                for neighbor_cfg in self._neighbors_of(self.best_configs[d], d + 1):
                    if neighbor_cfg not in self.seen:
                        self.seen.add(neighbor_cfg)
                        self.candidates.append(neighbor_cfg)
                        found = True
                if found:
                    break

        return self.candidates.popleft()

    def on_trial_complete(self, trial_id, result=None, error=False):
        pass
