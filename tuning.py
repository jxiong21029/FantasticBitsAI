import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction

import torch
from ray import air, tune
from ray.tune.search import Searcher

from agents import Agents
from env import FantasticBits
from ppo import Trainer


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

    def __hash__(self):
        return hash(
            tuple(self.config[k] for k in sorted(self.config.keys())) + (self.depth,)
        )

    def __eq__(self, other):
        return self.config == other.config and self.depth == other.depth


class LogBinarySearch(Searcher):
    def __init__(self, search_space, depth, metric, mode):
        """
        expected input format example:
        search_space = {
            "lr": log_binary_search(1e-3, 1e-2, 1e-1),
            "batch_size": grid_search(32, 64, 128),
            "weight_decay": 1e-5,
        }
        """

        super().__init__(metric=metric, mode=mode)

        self.max_depth = depth
        self.grid_searches = {}
        self.log_bin_scales = {}
        self.log_bin_sizes = {}

        keys = []
        start_values = []

        has_binary = False
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
                    self.grid_searches[k] = vals
                    start_values.append(vals)
                elif space_type == "log_binary_search":
                    has_binary = True
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
                    if (diff := min(diffs)) == max(diffs):
                        self.log_bin_scales[k] = diff
                    else:
                        raise ValueError(
                            f"expected evenly spaced (logarithmically speaking) "
                            f"specification for log binary search, but received "
                            f"exponents {exponents} for key {k}"
                        )

                    self.log_bin_sizes[k] = len(vals) // 2
                else:
                    raise ValueError(f"unexpected search space: {space_type}")

        if not has_binary:
            self.max_depth = 0

        self.candidates = []
        for cfg in itertools.product(*start_values):
            self.candidates.append(
                SearchNode({k: v for k, v in zip(keys, cfg)}, None, 0)
            )

        self.best_configs = {}
        self.best_scores = {}
        self.in_progress = {}
        self.all_deployed = set()

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
                    10.0
                    ** (
                        Fraction(math.log10(config[k])).limit_denominator(
                            2 * (2**self.max_depth)
                        )
                        + c * self.log_bin_scales[k] / (2**new_depth)
                    )
                    for c in range(-self.log_bin_sizes[k], self.log_bin_sizes[k] + 1)
                ]
            )

        for cfg in itertools.product(*values):
            yield {k: v for k, v in zip(keys, cfg)}

    def suggest(self, trial_id):
        self.candidates = [
            node
            for node in self.candidates
            if (
                node.depth - 1 not in self.best_configs
                or node.parent_config == self.best_configs[node.depth - 1]
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
            parent = random.choice(
                [node for node in possible_parents if node.depth == minimum_depth]
            )

            parent.expanded = True
            for config in self._neighbors_of(parent.config, parent.depth + 1):
                self.candidates.append(
                    SearchNode(config, parent.config, parent.depth + 1)
                )

            return self.suggest(trial_id)

        valid = [
            node for node in self.candidates if node.depth == self.candidates[0].depth
        ]
        selected = random.choice(valid)
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

            if node.depth < self.max_depth:
                node.expanded = True
                for config in self._neighbors_of(node.config, node.depth + 1):
                    self.candidates.append(
                        SearchNode(config, node.config, node.depth + 1)
                    )

        del self.in_progress[trial_id]


class IndependentComponentsSearch(Searcher):
    def __init__(
        self,
        search_space: dict,
        depth: int,
        defaults: dict,
        components: tuple,
        metric: str,
        mode: str,
        repeat=1,
    ):
        if search_space.keys() != defaults.keys():
            raise ValueError("expected search_space and defaults to have same keys")
        for comp in components:
            for k in comp:
                if k not in search_space.keys():
                    raise ValueError("expected component keys to be in search_space")

        super().__init__(metric=metric, mode=mode)
        self.depth = depth

        filled_components = components + tuple(
            k for k in search_space.keys() if not any(k in comp for comp in components)
        )
        self.components = filled_components * repeat
        self.search_space = search_space
        self.defaults = defaults

        self._curr_group = 0
        self.id_to_config = {}
        self.best_score = None
        self.best_config = None

        self._curr_searcher_ids = set()
        self._curr_searcher: LogBinarySearch = self._init_group()

    def _init_group(self):
        self._curr_searcher_ids.clear()
        search_subspace = {
            k: (v if k in self.components[self._curr_group] else self.defaults[k])
            for k, v in self.search_space.items()
        }
        return LogBinarySearch(
            search_space=search_subspace,
            depth=self.depth,
            metric=self.metric,
            mode=self.mode,
        )

    def suggest(self, trial_id):
        ret = self._curr_searcher.suggest(trial_id)
        self._curr_searcher_ids.add(trial_id)

        if ret == Searcher.FINISHED:
            self._curr_group += 1
            if self._curr_group == len(self.components):
                return Searcher.FINISHED
            self._curr_searcher = self._init_group()
            ret = self._curr_searcher.suggest(trial_id)

        self.id_to_config[trial_id] = ret
        return ret

    def on_trial_complete(self, trial_id, result=None, error=False):
        if error:
            self._curr_searcher.on_trial_complete(trial_id, result, error)
            return

        if (
            self.best_score is None
            or (self.mode == "max" and result[self.metric] > self.best_score)
            or (self.mode == "min" and result[self.metric] < self.best_score)
        ):
            self.best_score = result[self.metric]
            self.best_config = self.id_to_config[trial_id]

            config = self.id_to_config[trial_id]
            reinit = False
            for k in self.search_space.keys():
                if config[k] != self.defaults[k]:
                    self.defaults[k] = config[k]
                    if k not in self.components[self._curr_group]:
                        reinit = True
            if reinit:
                self._curr_searcher = self._init_group()

        if trial_id in self._curr_searcher_ids:
            self._curr_searcher.on_trial_complete(trial_id, result, error)


def train(config):
    results = []
    for run in range(3):
        trainer = Trainer(
            Agents(),
            FantasticBits,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            rollout_steps=config["rollout_steps"],
            minibatch_size=config["minibatch_size"],
            epochs=config["epochs"],
            entropy_reg=config["entropy_reg"],
            env_kwargs={"shape_snaffle_dist": True},
        )

        for i in range(101):
            trainer.train()
            if i % 20 == 0:
                trainer.evaluate(num_episodes=100)
                tune.report(
                    **{k: v[-1] for k, v in trainer.logger.cumulative_data.items()}
                )

                torch.save(trainer.agents.state_dict(), f"./agents_{run}.pth")
        results.append({k: v[-1] for k, v in trainer.logger.cumulative_data.items()})
    tune.report(
        **{
            "mo3_" + k: (results[0][k] + results[1][k] + results[2][k]) / 3
            for k in results[0].keys()
        }
    )


def main():
    import os

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    search_alg = IndependentComponentsSearch(
        search_space={
            "lr": log_binary_search(1e-4, 1e-3, 1e-2),
            "weight_decay": log_binary_search(1e-7, 1e-5, 1e-3),
            "gamma": grid_search(0.95, 0.97, 0.98, 0.99, 0.995),
            "gae_lambda": grid_search(0.5, 0.8, 0.9, 0.95, 0.97),
            "rollout_steps": 4096,
            "minibatch_size": grid_search(32, 64, 128),
            "epochs": grid_search(1, 2, 3, 4, 5),
            "entropy_reg": log_binary_search(1e-7, 1e-5, 1e-3),
        },
        depth=2,
        defaults={
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "rollout_steps": 4096,
            "minibatch_size": 64,
            "epochs": 3,
            "entropy_reg": 1e-5,
        },
        components=(
            ("lr", "weight_decay"),
            ("lr", "minibatch_size"),
            ("lr", "epochs"),
            ("gamma", "gae_lambda"),
        ),
        metric="mo3_eval_goals_scored_mean",
        mode="max",
    )

    tuner = tune.Tuner(
        train,
        tune_config=tune.TuneConfig(
            num_samples=-1,
            search_alg=search_alg,
            max_concurrent_trials=8,
        ),
        run_config=air.RunConfig(
            name="full_tune",
            local_dir="ray_results/",
            verbose=1,
        ),
    )
    tuner.fit()

    print("best score:", search_alg.best_score)
    print("best config:", search_alg.best_config)


if __name__ == "__main__":
    main()
