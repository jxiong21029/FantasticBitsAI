import itertools
import math
import random
from fractions import Fraction

import numpy as np
import pytest

from tuning import (
    IndependentComponentsSearch,
    IntervalHalvingSearch,
    grid_search,
    log_halving_search,
    q_log_halving_search,
    q_uniform_halving_search,
    uniform_halving_search,
)


@pytest.fixture
def base_searcher():
    return IntervalHalvingSearch(
        {
            "lr": log_halving_search(1e-3, 1e-2, 1e-1, 1e0, 1e1),
            "batch_size": grid_search(32, 64),
            "n_layers": 2,
        },
        depth=2,
        metric="val_acc",
        mode="max",
    )


def test_log_halving_nondecreasing_depth(base_searcher):
    suggestions = [base_searcher.suggest(str(i)) for i in range(100)]
    suggestions = [s for s in suggestions if isinstance(s, dict)]
    assert len(suggestions) == len(base_searcher.all_deployed)
    assert len(suggestions) == 2 * (5 + 13 + 29)

    def k(d):
        return d["lr"], d["batch_size"], d["n_layers"]

    n0 = [n for n in base_searcher.in_progress.values() if n.depth == 0]
    assert sorted(suggestions[: len(n0)], key=k) == sorted(
        [n.config for n in n0], key=k
    )

    n1 = [n for n in base_searcher.in_progress.values() if n.depth == 1]
    assert sorted(suggestions[len(n0) : len(n0) + len(n1)], key=k) == sorted(
        [n.config for n in n1], key=k
    )

    n2 = [n for n in base_searcher.in_progress.values() if n.depth == 2]
    assert sorted(suggestions[len(n0) + len(n1) :], key=k) == sorted(
        [n.config for n in n2], key=k
    )


def test_log_halving_denominators_match_depth(base_searcher):
    for i in range(100):
        base_searcher.suggest(str(i))

    assert len(base_searcher.in_progress) > 0
    largest_1 = None
    largest_2 = None
    for node in base_searcher.in_progress.values():
        denom = Fraction(math.log10(node.config["lr"])).limit_denominator(8).denominator
        if node.depth == 0:
            assert denom == 1
        elif node.depth == 1:
            assert 1 <= denom <= 2
            if largest_1 is None or denom > largest_1:
                largest_1 = denom
        else:
            assert denom in (1, 2, 4)
            if largest_2 is None or denom > largest_2:
                largest_2 = denom
    assert largest_1 == 2
    assert largest_2 == 4


def test_log_halving_does_not_expand_worse_result(base_searcher):
    for i in range(10):
        suggestion = base_searcher.suggest(str(i))
        if suggestion["lr"] == 1e-3:
            base_searcher.on_trial_complete(str(i), result={"val_acc": 0.7})
        else:
            base_searcher.on_trial_complete(str(i), result={"val_acc": 0.8})

    suggestions = []
    for i in range(10, 100):
        suggestions.append(base_searcher.suggest(str(i)))
    assert {"lr": 1e-4, "batch_size": 32, "n_layers": 2} not in suggestions


def test_log_halving_only_expands_best_result(base_searcher):
    found = False
    for i in range(10):
        suggestion = base_searcher.suggest(str(i))
        if suggestion == {"lr": 1e-1, "batch_size": 64, "n_layers": 2}:
            found = True
            base_searcher.on_trial_complete(str(i), result={"val_acc": 0.99})
        else:
            base_searcher.on_trial_complete(str(i), result={"val_acc": 0.50})
    assert found

    depth_1_suggestions = []
    for j in range(10, 20):
        depth_1_suggestions.append(base_searcher.suggest(str(j)))

    depth_1_suggestions.sort(key=lambda d: (d["lr"], d["batch_size"], d["n_layers"]))

    for i, cfg in enumerate(
        itertools.product(
            [10 ** (-2), 10 ** (-1.5), 10 ** (-1), 10 ** (-0.5), 1],
            [32, 64],
        )
    ):
        assert np.isclose(depth_1_suggestions[i]["lr"], cfg[0])
        assert depth_1_suggestions[i]["batch_size"] == cfg[1]


@pytest.fixture
def searcher_2():
    return IntervalHalvingSearch(
        search_space={
            "lr": log_halving_search(1e-4, 1e-3, 1e-2),
            "batch_size": q_log_halving_search(32, 64, 128),
            "dropout": uniform_halving_search(0.1, 0.2, 0.3),
            "n_layers": q_uniform_halving_search(2, 8, 14),
        },
        depth=2,
        metric="val_loss",
        mode="min",
    )


def test_special_search_spaces_suggestion_count(searcher_2):
    for i in range(3**5):
        assert isinstance(searcher_2.suggest(str(i)), dict)
        searcher_2.on_trial_complete(str(i), {"val_loss": random.random()})
    assert not isinstance(searcher_2.suggest("243"), dict)


def test_independent_components_search():
    for seed in range(10):
        searcher = IndependentComponentsSearch(
            search_space={
                "lr": log_halving_search(1e-5, 1e-3, 1e-1),
                "batch_size": grid_search(32, 64),
                "weight_decay": log_halving_search(1e-5, 1e-3, 1e-1),
                "dropout": grid_search(0.1, 0.2, 0.3),
                "n_layers": 2,
            },
            depth=2,
            defaults={
                "lr": 1e-4,
                "batch_size": 64,
                "weight_decay": 1e-4,
                "dropout": 0.2,
                "n_layers": 2,
            },
            components=(
                ("lr", "batch_size"),
                ("weight_decay", "dropout"),
            ),
            metric="val_acc",
            mode="max",
            repeat=2,
            seed=2**(seed // 2) + seed
        )
        s0 = searcher.suggest("0")
        s1 = searcher.suggest("1")
        searcher.on_trial_complete("0", {"val_acc": 0.99})

        assert (s0["lr"] != s1["lr"]) or (s0["batch_size"] != s1["batch_size"])
        assert s0["weight_decay"] == s1["weight_decay"] == 1e-4
        assert s0["dropout"] == s1["dropout"] == 0.2
        assert s0["n_layers"] == s1["n_layers"] == 2

        other_suggestions = []
        for i in range(2, 2 * (3 + 7 + 15)):
            other_suggestions.append(searcher.suggest(str(i)))

        assert searcher.curr_comp == 0
        sa = searcher.suggest("a")
        assert searcher.curr_comp == 1
        sb = searcher.suggest("b")
        assert searcher.curr_comp == 1

        assert (sa["weight_decay"] != sb["weight_decay"]) or (
            sa["dropout"] != sb["dropout"]
        )
        assert sa["lr"] == sb["lr"]
        assert sa["batch_size"] == sb["batch_size"]

        for i, s in enumerate(other_suggestions):
            if s["lr"] != s0["lr"]:
                assert searcher.id_to_config[str(i + 2)] == s
                searcher.on_trial_complete(str(i + 2), {"val_acc": 0.999})
                break

        assert len(searcher.curr_searcher.in_progress) == 0


def test_independent_components_no_wasted_sweeps():
    searcher = IndependentComponentsSearch(
        {
            "a": 1,
            "b": 1,
            "c": grid_search(1, 2),
            "d": grid_search(1, 2, 3),
            "e": grid_search(1, 2, 3, 4, 5),
            "f": log_halving_search(1e-3, 1e-2, 1e-1),
        },
        depth=1,
        defaults={
            "a": 1,
            "b": 1,
            "c": 2,
            "d": 2,
            "e": 3,
            "f": 1e-2,
        },
        components=(("c",), ("d", "e")),
        metric="whatever",
        mode="max",
        repeat=2,
    )

    suggestions = []
    for i in range(100):
        suggestions.append(searcher.suggest(str(i)))
    suggestions = [s for s in suggestions if isinstance(s, dict)]
    assert len(suggestions) == 34
