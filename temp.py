from collections import deque
from dataclasses import dataclass

from ray.tune.search import Searcher

from tuning import IntervalHalvingSearch


@dataclass
class ICSearchNode:
    searcher: IntervalHalvingSearch
    parent_subconfig: dict
    group: int

    def __hash__(self):
        return hash(
            tuple(
                self.parent_subconfig[k] for k in sorted(self.parent_subconfig.keys())
            )
            + (self.group,)
        )


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
        verbose=False,
    ):
        if search_space.keys() != defaults.keys():
            raise ValueError("expected search_space and defaults to have same keys")
        for group in groups:
            for k in group:
                if k not in search_space.keys():
                    raise ValueError("expected group keys to be in search_space")

        super().__init__(metric=metric, mode=mode)
        self.depth = depth
        self.seed = seed
        self.verbose = verbose

        self.search_space = search_space
        self.groups = groups * repeat
        self.defaults = defaults

        self.candidates = deque(
            [
                ICSearchNode(
                    IntervalHalvingSearch(
                        {
                            k: v if k in self.groups[0] else defaults[k]
                            for k, v in self.search_space.items()
                        },
                        depth=self.depth,
                        metric=self.metric,
                        mode=self.mode,
                    ),
                    {},
                    0,
                )
            ]
        )
        self.in_progress = {}  # trial_id -> (config, group)?

        self.best_config = {}
        self.best_score = {}

    def searcher_from(self, config, new_group_idx):
        parent_subconfig = {}
        search_subspace = {}
        for k in self.search_space.keys():
            if k in self.groups[new_group_idx]:
                search_subspace[k] = self.search_space[k]
            else:
                search_subspace[k] = config[k]
                parent_subconfig[k] = config[k]
        return ICSearchNode(
            IntervalHalvingSearch(
                search_space=search_subspace,
                depth=self.depth,
                metric=self.metric,
                mode=self.mode,
            ),
            parent_subconfig=parent_subconfig,
            group=new_group_idx,
        )

    def suggest(self, trial_id):
        def contains(config, subconfig):
            return all(config[k] == v for k, v in subconfig.items())

        self.candidates = deque(
            [
                node
                for node in self.candidates
                if (
                    node.group not in self.best_config.keys()
                    or contains(self.best_config[node.group - 1], node.parent_subconfig)
                    or any(
                        contains(config, node.parent_subconfig)
                        for config, _ in self.in_progress.values()
                    )
                )
                and True  # TODO: check if subconfig + group isn't already being checked
            ]
        )

        for candidate in self.candidates:
            suggestion = candidate.searcher.suggest(trial_id)
            if isinstance(suggestion, dict):
                self.in_progress[trial_id] = (suggestion, self.candidates[0].group)
                return suggestion

        while len(self.candidates) > 0:
            done = False
            for node in self.candidates[0].searcher.in_progress.values():
                if node.depth == self.candidates[0].searcher.max_depth:
                    self.candidates.append(
                        self.searcher_from(node.config, self.candidates[0].group + 1)
                    )
                    done = True
            self.candidates.popleft()
            if done:
                break
        else:
            return Searcher.FINISHED

        return self.suggest(trial_id)

    def on_trial_complete(self, trial_id, result=None, error=False):
        pass
