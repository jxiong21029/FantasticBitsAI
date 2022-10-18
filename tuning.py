import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from ray.tune.search import Searcher


def log_halving_search(*values):  # could be used for learning rate
    return {"log_halving_search": values}


def q_log_halving_search(*values):  # could be used for batch size
    return {"q_log_halving_search": values}


def uniform_halving_search(*values):  # could be used for dropout
    return {"uniform_halving_search": values}


def q_uniform_halving_search(*values):  # could be used for model depth
    return {"q_uniform_halving_search": values}


def grid_search(*values):  # could be used for everything else
    return {"grid_search": values}


@dataclass
class SearchNode:
    config: dict
    parent_config: dict
    depth: int
    expanded: bool = False

    def __hash__(self):
        return hash(
            tuple(self.config[k] for k in sorted(self.config.keys())) + (self.depth,)
        )

    def __eq__(self, other):
        return self.config == other.config and self.depth == other.depth


class IntervalHalvingSearch(Searcher):
    def __init__(self, search_space, depth, metric, mode, seed=None):
        """
        expected input format example:
        search_space = {
            "lr": log_halving_search(1e-3, 1e-2, 1e-1),
            "batch_size": grid_search(32, 64, 128),
            "weight_decay": 1e-5,
        }
        """

        super().__init__(metric=metric, mode=mode)
        self.rng = np.random.default_rng(seed=seed)

        self.max_depth = depth
        self.search_space = search_space
        self.grid_searches = {}
        self.keys = list(search_space.keys())

        self.neighbors = defaultdict(dict)

        start_values = []

        has_halving = False
        for k in self.keys:
            specification = search_space[k]
            if not isinstance(specification, dict):
                self.grid_searches[k] = [specification]
                start_values.append([specification])
                continue

            assert len(specification) == 1

            space_type = list(specification.keys())[0]
            vals = specification[space_type]
            start_values.append(vals)

            if space_type == "grid_search":
                self.grid_searches[k] = vals
                continue

            assert space_type in (
                "log_halving_search",
                "q_log_halving_search",
                "uniform_halving_search",
                "q_uniform_halving_search",
            )
            assert len(vals) >= 3 and len(vals) % 2 == 1

            has_halving = True

            depth_vals = [sorted(vals)]
            sz = len(vals) // 2
            if "log" in space_type:
                base_factor = vals[1] / vals[0]
                for d in range(1, self.max_depth + 1):
                    factor = math.pow(base_factor, math.pow(2, -d))
                    depth_vals.append(
                        [
                            depth_vals[d - 1][0] / math.pow(factor, i + 1)
                            for i in reversed(range(sz))
                        ]
                    )
                    for entry in depth_vals[d - 1][:-1]:
                        depth_vals[d].append(entry)
                        depth_vals[d].append(entry * factor)
                    depth_vals[d].extend(
                        depth_vals[d - 1][-1] * math.pow(factor, i)
                        for i in range(sz + 1)
                    )
            else:
                base_diff = vals[1] - vals[0]
                for d in range(1, self.max_depth + 1):
                    diff = base_diff * math.pow(2, -d)
                    depth_vals.append(
                        [
                            depth_vals[d - 1][0] - diff * (i + 1)
                            for i in reversed(range(sz))
                        ]
                    )
                    for entry in depth_vals[d - 1][:-1]:
                        depth_vals[d].append(entry)
                        depth_vals[d].append(entry + diff)
                    depth_vals[d].extend(
                        depth_vals[d - 1][-1] + diff * i for i in range(sz + 1)
                    )

            if space_type.startswith("q"):
                for d in range(self.max_depth + 1):
                    depth_vals[d] = [round(entry) for entry in depth_vals[d]]

            for d in range(self.max_depth):
                for entry in depth_vals[d]:
                    if "log" in space_type:
                        candidates = sorted(
                            depth_vals[d + 1], key=lambda v: abs(math.log(v / entry))
                        )
                    else:
                        candidates = sorted(
                            depth_vals[d + 1], key=lambda v: abs(v - entry)
                        )
                    self.neighbors[k][(entry, d)] = sorted(
                        set(candidates[: 2 * sz + 1])
                    )

        if not has_halving:
            self.max_depth = 0

        self.candidates = []
        for cfg in itertools.product(*start_values):
            self.candidates.append(
                SearchNode({k: v for k, v in zip(self.keys, cfg)}, None, 0)
            )

        self.best_config = {}
        self.best_score = {}
        self.in_progress = {}
        self.all_deployed = set()

    def config_neighbors(self, config, new_depth):
        values = []

        for k in self.keys:
            if k in self.grid_searches:
                values.append(self.grid_searches[k])
            else:
                values.append(self.neighbors[k][(config[k], new_depth - 1)])

        for cfg in itertools.product(*values):
            yield {k: v for k, v in zip(self.keys, cfg)}

    def suggest(self, trial_id):
        self.candidates = [
            node
            for node in self.candidates
            if (
                node.depth - 1 not in self.best_config
                or node.parent_config == self.best_config[node.depth - 1]
                or node.parent_config
                in [
                    n.config
                    for n in self.in_progress.values()
                    if n.depth == node.depth - 1
                ]
            )
            and node not in self.all_deployed
        ]

        if len(self.candidates) == 0:
            possible_parents = [
                node
                for node in self.in_progress.values()
                if not node.expanded and node.depth < self.max_depth
            ]
            if len(possible_parents) == 0:
                return Searcher.FINISHED
            minimum_depth = min(node.depth for node in possible_parents)
            parent = self.rng.choice(
                [node for node in possible_parents if node.depth == minimum_depth]
            )

            parent.expanded = True
            for config in self.config_neighbors(parent.config, parent.depth + 1):
                self.candidates.append(
                    SearchNode(config, parent.config, parent.depth + 1)
                )

            return self.suggest(trial_id)

        valid = [
            node for node in self.candidates if node.depth == self.candidates[0].depth
        ]
        selected = self.rng.choice(valid)
        self.candidates.remove(selected)
        self.in_progress[trial_id] = selected
        self.all_deployed.add(selected)
        return selected.config

    def on_trial_complete(self, trial_id, result=None, error=False):
        if error:
            del self.in_progress[trial_id]
            return

        node = self.in_progress[trial_id]
        if (
            node.depth not in self.best_config
            or (
                self.mode == "max" and result[self.metric] > self.best_score[node.depth]
            )
            or (
                self.mode == "min" and result[self.metric] < self.best_score[node.depth]
            )
        ):
            self.best_score[node.depth] = result[self.metric]
            self.best_config[node.depth] = node.config

            if node.depth < self.max_depth and not node.expanded:
                node.expanded = True
                for config in self.config_neighbors(node.config, node.depth + 1):
                    self.candidates.append(
                        SearchNode(config, node.config, node.depth + 1)
                    )

        del self.in_progress[trial_id]


@dataclass
class IndependentGroupsNode:
    parent: "IndependentGroupsNode"
    parent_config: dict
    group: int
    searcher: IntervalHalvingSearch = None
    expanded: bool = False
    deepest: bool = False


class IndependentGroupsSearch(Searcher):
    def __init__(
        self,
        search_space: dict,
        depth: int,
        defaults: dict,
        groups: tuple,
        metric: str,
        mode: str,
        repeat=1,
        seed=None,
    ):
        if search_space.keys() != defaults.keys():
            raise ValueError("expected search_space and defaults to have same keys")
        for group in groups:
            for k in group:
                if k not in search_space.keys():
                    raise ValueError("expected group keys to be in search_space")

        super().__init__(metric=metric, mode=mode)
        self.search_space = search_space
        self.depth = depth
        self.groups = groups * repeat
        self.defaults = defaults

        self.nodes = [
            IndependentGroupsNode(
                parent=None,
                parent_config=self.defaults,
                group=0,
                expanded=True,
            )
        ]
        self.best_config = {}
        self.best_score = {}
        self.in_progress = {}

        self.rng = np.random.default_rng(seed)

    def init_search(self, node: IndependentGroupsNode):
        node.searcher = IntervalHalvingSearch(
            {
                k: v if k in self.groups[node.group] else node.parent_config[k]
                for k, v in self.search_space.items()
            },
            depth=self.depth,
            metric=self.metric,
            mode=self.mode,
            seed=self.rng.integers(2**31),
        )

    def suggest(self, trial_id):
        self.nodes = deque(sorted(self.nodes, key=lambda n: n.group))
        while self.nodes:
            selected = self.nodes[0]
            if (
                selected.group - 1 in self.best_config
                and self.best_config[selected.group - 1] != selected.parent_config
                and selected not in self.in_progress.values()
            ):
                self.nodes.popleft()
                continue
            if selected.searcher is None:
                self.init_search(selected)
            suggestion = selected.searcher.suggest(trial_id)
            if suggestion == Searcher.FINISHED:
                self.nodes.popleft()
                continue
            self.in_progress[trial_id] = IndependentGroupsNode(
                parent=selected,
                parent_config=suggestion,
                group=selected.group + 1,
                deepest=(
                    selected.searcher.in_progress[trial_id].depth
                    == selected.searcher.max_depth
                ),
            )
            return suggestion

        for node in self.in_progress.values():
            if node.group < len(self.groups) and not node.expanded and node.deepest:
                node.expanded = True
                self.nodes.append(node)
                return self.suggest(trial_id)
        return Searcher.FINISHED

    def on_trial_complete(self, trial_id, result=None, error=False):
        node = self.in_progress[trial_id]
        node.parent.searcher.on_trial_complete(trial_id, result, error)

        del self.in_progress[trial_id]
        if error:
            return

        score = result[self.metric]
        if node.deepest and (
            node.group - 1 not in self.best_score
            or self.mode == "max"
            and score > self.best_score[node.group - 1]
            or self.mode == "min"
            and score < self.best_score[node.group - 1]
        ):
            self.best_score[node.group - 1] = score
            self.best_config[node.group - 1] = node.parent_config
            if node.group < len(self.groups) and not node.expanded and node.deepest:
                self.nodes.append(node)
