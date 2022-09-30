import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction

from ray.tune.search import Searcher


def log_binary_search(*values):
    return {"log_binary_search": values}


def grid_search(*values):
    return {"grid_search": values}


@dataclass
class SearchNode:
    config: dict
    parent_config: dict
    depth: int
    expanded: bool = False


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

        self.candidates = []
        for spec in itertools.product(*start_values):
            self.candidates.append(
                SearchNode({k: v for k, v in zip(keys, spec)}, None, 0)
            )

        self.best_configs = {}
        self.best_scores = {}
        self.in_progress = {}
        self.completed = set()

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
                    10.0 ** (Fraction(math.log10(config[k])).limit_denominator(2 * (2 ** self.max_depth)) + c * self.log_bin_scales[k] / (2**new_depth))
                    for c in range(-self.log_bin_sizes[k], self.log_bin_sizes[k] + 1)
                ]
            )
            assert len(values) == self.log_bin_sizes[k] * 2 + 1  # TODO remove

        for cfg in itertools.product(*values):
            yield {k: v for k, v in zip(keys, cfg)}

    def suggest(self, trial_id):
        self.candidates = [
            node
            for node in self.candidates
            if (
                node.depth - 1 not in self.best_configs
                or node.parent_config == self.best_configs[node.depth - 1]
            )
            and not any(nd.config == node.config for nd in self.in_progress.values())
        ]

        if len(self.candidates) == 0:
            possible_parents = [node for node in self.in_progress.values() if not node.expanded and node.depth < self.max_depth]
            if len(possible_parents) == 0:
                return Searcher.FINISHED
            minimum_depth = min(node.depth for node in possible_parents)
            parent = random.choice(
                [node for node in possible_parents if node.depth == minimum_depth]
            )

            print(f">> expanding {parent} <<")

            parent.expanded = True
            for config in self._neighbors_of(parent.config, parent.depth + 1):
                self.candidates.append(
                    SearchNode(config, parent.config, parent.depth + 1)
                )

            return self.suggest(trial_id)

        valid = [node for node in self.candidates if node.depth == self.candidates[0].depth]
        selected = random.choice(valid)
        self.candidates.remove(selected)
        self.in_progress[trial_id] = selected
        print(f"<< suggesting {selected} >>")
        return selected.config

    def on_trial_complete(self, trial_id, result=None, error=False):
        node = self.in_progress[trial_id]
        if (
            node.depth not in self.best_configs
            or (
                self.mode == "max"
                and result[self.metric] > self.best_scores[node.depth]
            )
            or (
                self.mode == "min"
                and result[self.metric] < self.best_scores[node.depth]
            )
        ):
            self.best_scores[node.depth] = result[self.metric]
            self.best_configs[node.depth] = node.config

        del self.in_progress[trial_id]


searcher = LogBinarySearch(
    {
        "lr": log_binary_search(1e-3, 1e-2, 1e-1),
        "batch_size": grid_search(32, 64),
        "n_layers": 2,
    },
    depth=2,
    metric="val_acc",
    mode="max",
)
for i in range(100):
    searcher.suggest(str(i))
