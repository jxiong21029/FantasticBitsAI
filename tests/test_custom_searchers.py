import itertools
import math
import random
from fractions import Fraction

import numpy as np
import pytest
from ray import tune

from tuning import (
    IndependentGroupsSearch,
    IntervalHalvingScheduler,
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
    assert not isinstance(searcher_2.suggest("last"), dict)


@pytest.fixture
def group_searchers_big_50():
    ret = []
    for seed in range(50):
        ret.append(
            IndependentGroupsSearch(
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
                groups=(
                    ("lr", "batch_size"),
                    ("weight_decay", "dropout"),
                ),
                metric="val_acc",
                mode="max",
                seed=2**seed - seed,
            )
        )
    return ret


@pytest.fixture
def rng_50():
    rng = np.random.default_rng(2**16 - 1)
    ret = []
    for seed in range(50):
        ret.append(np.random.default_rng(seed=rng.integers(2**31)))
    return ret


def test_independent_groups_search_parameters(group_searchers_big_50):
    for searcher in group_searchers_big_50:
        for i in range(3 + 7 + 15):
            a = searcher.suggest(f"a{i}")
            b = searcher.suggest(f"b{i}")

            assert (a["lr"] != b["lr"]) or (a["batch_size"] != b["batch_size"])
            assert a["weight_decay"] == b["weight_decay"] == 1e-4
            assert a["dropout"] == b["dropout"] == 0.2
            assert a["n_layers"] == b["n_layers"] == 2

        b0 = searcher.suggest("b0")
        b1 = searcher.suggest("b1")

        assert (b0["weight_decay"] != b1["weight_decay"]) or (
            b0["dropout"] != b1["dropout"]
        )
        assert b0["lr"] == b1["lr"]
        assert b0["batch_size"] == b1["batch_size"]


def test_independent_groups_maximum_trials(group_searchers_big_50):
    first = 3 * 2 + 7 * 2 + 15 * 2
    second = 3 * 3 + 7 * 3 + 15 * 3
    correct = first + 15 * 2 * second
    for searcher in group_searchers_big_50[:5]:
        i = 0
        while True:
            suggestion = searcher.suggest(str(i))
            if suggestion == searcher.FINISHED:
                break
            i += 1
        assert i == correct


def test_independent_groups_minimum_trials(group_searchers_big_50, rng_50):
    correct = 6 * 3 + 9 * 3
    for searcher, rng in zip(group_searchers_big_50, rng_50):
        i = 0
        while True:
            suggestion = searcher.suggest(str(i))
            if suggestion == searcher.FINISHED:
                break
            searcher.on_trial_complete(str(i), {"val_acc": rng.random()})
            i += 1
        if i == 18:
            breakpoint()
        assert i == correct


def test_independent_groups_stress(group_searchers_big_50, rng_50):
    for iteration, (searcher, rng) in enumerate(zip(group_searchers_big_50, rng_50)):
        always_incr = iteration < 25

        if iteration < 5:
            p = 0
        elif iteration < 10:
            p = 0.03
        elif iteration < 15:
            p = 0.97
        elif iteration < 20:
            p = 1
        else:
            p = rng.random()

        i = 0
        in_prog = set()
        score = rng.random()
        while True:
            if searcher.suggest(str(i)) == searcher.FINISHED:
                break
            in_prog.add(str(i))
            if rng.random() < p:
                rm = rng.choice(list(in_prog))
                in_prog.remove(rm)
                searcher.on_trial_complete(rm, {"val_acc": score})

            if always_incr:
                score = 1 - (1 - score) * 0.99
            else:
                score = rng.random()
            i += 1


def test_independent_groups_just_one():
    searcher = IndependentGroupsSearch(
        search_space={"a": grid_search(1)},
        depth=99,
        defaults={"a": 1},
        groups=(("a",),),
        metric="whatever",
        mode="max",
    )
    assert isinstance(searcher.suggest("0"), dict)
    assert searcher.suggest("1") == searcher.FINISHED

    searcher_2 = IndependentGroupsSearch(
        search_space={"a": grid_search(1), "b": grid_search(2)},
        depth=99,
        defaults={"a": 1, "b": 2},
        groups=(("a", "b"),),
        metric="whatever",
        mode="max",
    )
    assert isinstance(searcher_2.suggest("0"), dict)
    assert searcher_2.suggest("1") == searcher_2.FINISHED

    searcher_3 = IndependentGroupsSearch(
        search_space={"a": grid_search(1), "b": grid_search(2)},
        depth=99,
        defaults={"a": 1, "b": 2},
        groups=(("a",), ("b",)),
        metric="whatever",
        mode="max",
    )
    assert isinstance(searcher_3.suggest("0"), dict)
    assert isinstance(searcher_3.suggest("1"), dict)
    assert searcher_3.suggest("2") == searcher_3.FINISHED


def test_independent_groups_half_reported():
    searcher = IndependentGroupsSearch(
        {"a": log_halving_search(1e-3, 1e-2, 1e-1), "b": grid_search(4, 5, 6)},
        depth=1,
        defaults={"a": 1e-2, "b": 5},
        groups=(("a",), ("b",)),
        metric="val_loss",
        mode="min",
        seed=2**15 - 1,
    )

    for i in range(3):
        suggestion = searcher.suggest(f"a{i}")
        if suggestion["a"] == 1e-2:
            searcher.on_trial_complete(f"a{i}", {"val_loss": 0.3})
        else:
            searcher.on_trial_complete(f"a{i}", {"val_loss": 0.4})

    found1 = False
    found2 = False
    found3 = False
    for i in range(3):
        suggestion = searcher.suggest(f"b{i}")
        if suggestion["a"] == 1e-2:
            searcher.on_trial_complete(f"b{i}", {"val_loss": 0.6})
            found1 = True
        elif np.isclose(suggestion["a"], 10 ** (-2.5), rtol=1e-3):
            searcher.on_trial_complete(f"b{i}", {"val_loss": 0.5})
            found2 = True
        else:
            found3 = True

    assert found1
    assert found2
    assert found3
    assert len(searcher.in_progress) == 1

    group_2_suggestions = []
    while True:
        suggestion = searcher.suggest(f"c{len(group_2_suggestions)}")
        if suggestion == searcher.FINISHED:
            break
        group_2_suggestions.append(suggestion)

    assert len(group_2_suggestions) == 6
    for suggestion in group_2_suggestions:
        assert np.isclose(suggestion["a"], 10 ** (-1.5), rtol=1e-3) or np.isclose(
            suggestion["a"], 10 ** (-2.5), rtol=1e-3
        )
    assert set(suggestion["b"] for suggestion in group_2_suggestions) == {4, 5, 6}


def test_interval_halving_scheduler():
    searcher = IntervalHalvingSearch(
        search_space={
            "lr": log_halving_search(1e-5, 1e-3, 1e-1),
        },
        depth=2,
        metric="valid_loss",
        mode="min",
    )

    scheduler = IntervalHalvingScheduler(searcher, 5)

    def trainable(config):
        tune.report(valid_loss=0.1)

    tune.run(
        trainable,
    )
    # TODO finish writing test
