from collections import deque
from dataclasses import dataclass

from ray.tune.search import Searcher

from tuning import IntervalHalvingSearch


@dataclass
class ICSearchNode:
    config: dict
    group: int

    def __hash__(self):
        return hash(
            tuple(self.config[k] for k in sorted(self.config.keys())) + (self.group,)
        )

    def __eq__(self, other):
        return self.config == other.config and self.group == other.group


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

        self.searcher_best = IntervalHalvingSearch(
            {
                k: v if k in self.groups[0] else defaults[k]
                for k, v in self.search_space.items()
            },
            depth=self.depth,
            metric=self.metric,
            mode=self.mode,
        )
        self.searcher_greedy = None
        self.candidates = []
        self.curr_group = 0
        self.best_config = [None for _ in range(len(self.groups))]
        self.best_score = [None for _ in range(len(self.groups))]
        self.id_to_group = {}
        self.id_to_config = {}

    def suggest(self, trial_id):
        if self.searcher_best is not None:
            suggestion = self.searcher_best.suggest(trial_id)
            if suggestion == Searcher.FINISHED:
                self.searcher_best = None
        else:
            suggestion = self.searcher_greedy.suggest(trial_id)
            if suggestion == Searcher.FINISHED:
                self.searcher_best = None

        if suggestion == Searcher.FINISHED:
            for candidate in self.candidates:
                pass

        if self.searcher.in_progress[trial_id].depth == self.searcher.max_depth:
            self.candidates.append(ICSearchNode(suggestion, self.curr_group))

        self.id_to_group[trial_id] = self.curr_group
        self.id_to_config[trial_id] = suggestion
        return suggestion

    def on_trial_complete(self, trial_id, result=None, error=False):
        # TODO: 2 cases
        #   case 1: beats known best / first result in group
        #       a. preceding group: reset and expand if not already explored
        #           -> so we do need to store past searchers...
        #       b. current group, same search: nothing
        #       c. current group, diff search: nothing
        #       d. following group: nothing
        #   case 2: loses to best: nothing

        trial_group = self.id_to_group[trial_id]
        trial_config = self.id_to_config[trial_id]
        score = result[self.metric]
        if (
            self.best_score[trial_group] is None
            or self.mode == "max"
            and score > self.best_score[trial_group]
            or self.mode == "min"
            and score < self.best_score[trial_group]
        ):
            if trial_group < self.curr_group:
                for k in self.search_space.keys():
                    pass
                    # if k not in self.groups[self.curr_group] and self.search

            self.best_score[trial_group] = score
            self.best_config[trial_group] = trial_config
