import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from ray import air, tune
from ray.tune.search import Searcher

from agents import Agents
from env import FantasticBits
from ppo import Trainer


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

        self.best_configs = {}
        self.best_scores = {}
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
                for config in self.config_neighbors(node.config, node.depth + 1):
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
        seed=None,
    ):
        if search_space.keys() != defaults.keys():
            raise ValueError("expected search_space and defaults to have same keys")
        for comp in components:
            for k in comp:
                if k not in search_space.keys():
                    raise ValueError("expected component keys to be in search_space")

        super().__init__(metric=metric, mode=mode)
        self.seed = seed
        self.depth = depth

        self.components = components * repeat
        self.search_space = search_space
        self.defaults = defaults

        self.curr_comp = 0
        self.id_to_config = {}
        self.id_to_component = {}
        self.curr_searcher: IntervalHalvingSearch = self.init_search()

        self.best_score = [None for _ in range(len(self.components))]
        self.best_config = [None for _ in range(len(self.components))]
        self.done = False

    def init_search(self):
        search_subspace = {
            k: (v if k in self.components[self.curr_comp] else self.defaults[k])
            for k, v in self.search_space.items()
        }
        return IntervalHalvingSearch(
            search_space=search_subspace,
            depth=self.depth,
            metric=self.metric,
            mode=self.mode,
            seed=self.seed,
        )

    def suggest(self, trial_id):
        if self.done:
            return Searcher.FINISHED
        ret = self.curr_searcher.suggest(trial_id)

        if ret == Searcher.FINISHED:
            self.curr_comp += 1
            if self.curr_comp == len(self.components):
                self.done = True
                return Searcher.FINISHED
            self.curr_searcher = self.init_search()
            ret = self.curr_searcher.suggest(trial_id)

        self.id_to_config[trial_id] = ret
        self.id_to_component[trial_id] = self.curr_comp
        return ret

    def on_trial_complete(self, trial_id, result=None, error=False):
        trial_comp = self.id_to_component[trial_id]
        if error:
            if trial_comp == self.curr_comp:
                self.curr_searcher.on_trial_complete(trial_id, result, error)
            return

        score = result[self.metric]
        if (
            self.best_score[trial_comp] is None
            or (self.mode == "max" and score > self.best_score[trial_comp])
            or (self.mode == "min" and score < self.best_score[trial_comp])
        ):

            config = self.id_to_config[trial_id]
            self.best_score[trial_comp] = score
            self.best_config[trial_comp] = config

            if trial_comp <= self.curr_comp:
                reinit = False
                for k in self.search_space.keys():
                    if config[k] != self.defaults[k]:
                        self.defaults[k] = config[k]
                        if k not in self.components[self.curr_comp]:
                            reinit = True
                if reinit:
                    self.curr_comp = trial_comp + 1
                    self.curr_searcher = self.init_search()

        if trial_comp == self.curr_comp:
            self.curr_searcher.on_trial_complete(trial_id, result, error)


def train(config):
    budget = 200 * 4096
    iters = 1 + round(budget / config["rollout_steps"])
    steps_per_report = 20 * 4096
    iters_per_report = round(steps_per_report / config["rollout_steps"])

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
            env_kwargs={
                "bludgers_enabled": False,
                "opponents_enabled": False,
                "shape_snaffle_dist": True,
            },
        )

        for i in range(iters):
            trainer.train()
            if i % iters_per_report == 0:
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
            "lr": log_halving_search(1e-4, 1e-3, 1e-2),
            "minibatch_size": q_log_halving_search(32, 128, 512),
            "weight_decay": log_halving_search(1e-7, 1e-5, 1e-3),
            "epochs": grid_search(1, 2, 3, 4, 5),
            "gamma": grid_search(0.95, 0.97, 0.98, 0.99, 0.995),
            "gae_lambda": grid_search(0.5, 0.8, 0.9, 0.95, 0.97),
            "entropy_reg": log_halving_search(1e-7, 1e-5, 1e-3),
            "rollout_steps": grid_search(1024, 2048, 4096, 8192),
        },
        depth=1,
        defaults={
            "lr": 1e-4,
            "minibatch_size": 64,
            "weight_decay": 1e-4,
            "epochs": 3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "entropy_reg": 1e-5,
            "rollout_steps": 4096,
        },
        components=(
            ("lr", "minibatch_size"),
            ("weight_decay",),
            ("epochs",),
            ("gamma", "gae_lambda"),
            ("entropy_reg",),
            ("rollout_steps",),
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
            name="full_tune_3",
            local_dir="ray_results/",
            verbose=1,
        ),
    )
    tuner.fit()

    print("best score:", search_alg.best_score)
    print("best config:", search_alg.best_config)


if __name__ == "__main__":
    main()
