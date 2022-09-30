import math
from fractions import Fraction

import pytest

from tuning import LogBinarySearch, grid_search, log_binary_search


@pytest.fixture
def base_searcher():
    return LogBinarySearch(
        {
            "lr": log_binary_search(1e-3, 1e-2, 1e-1, 1e0, 1e1),
            "batch_size": grid_search(32, 64),
            "n_layers": 2,
        },
        depth=2,
        metric="val_acc",
        mode="max",
    )


def test_nondecreasing_depth(base_searcher):
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


def test_denominators_match_depth(base_searcher):
    for i in range(100):
        base_searcher.suggest(str(i))

    assert len(base_searcher.in_progress) > 0
    for node in base_searcher.in_progress.values():
        if node.depth == 0:
            assert (
                Fraction(math.log10(node.config["lr"])).limit_denominator(8).denominator
                == 1
            )
        elif node.depth == 1:
            assert (
                Fraction(math.log10(node.config["lr"])).limit_denominator(8).denominator
                <= 2
            )
        else:
            assert (
                Fraction(math.log10(node.config["lr"])).limit_denominator(8).denominator
                <= 4
            )


def test_only_selects_best(base_searcher):  # TODO
    pass
