"""Certificates preserve exhaustive ELS choices and candidate mask order."""

from itertools import product
from math import ceil, nextafter

import pytest
import torch
from torch import nn

from ttsafety.behaviors import bestfirst_layers
from ttsafety.els_bounds import can_prune_binary_rate


def test_ties_and_float_boundaries():
    rate = 7 / 25
    assert ceil(rate * 25) == 8  # A ceil threshold incorrectly misses this tie.
    assert can_prune_binary_rate(7, 25, rate)
    assert not can_prune_binary_rate(7, 25, nextafter(rate, 1))
    assert can_prune_binary_rate(7, 25, nextafter(rate, 0))
    assert can_prune_binary_rate(0, 42, 0)
    assert not can_prune_binary_rate(1, 42, 0.1)  # Not 1 / evaluated_count.
    assert can_prune_binary_rate(21, 42, 0.5)


@pytest.mark.parametrize("args", [(0, 0, .5), (0, -1, .5), (0, 2.5, .5),
                                  (-1, 42, .5), (43, 42, .5), (1.5, 42, .5),
                                  (True, 42, .5), (0, 42, float("nan")),
                                  (0, 42, float("inf")), (0, 42, -0.1),
                                  (0, 42, 1.1)])
def test_invalid_certificate_inputs(args):
    with pytest.raises(ValueError):
        can_prune_binary_rate(*args)


def run_search(outcomes, *, bounded, eps=0, infeasible=(), batch_size=1,
               base_metric=1., fail=False):
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(10, 1, bias=False) for _ in range(2)])
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(1)
    trace, evaluated, incumbents, ppl_calls = [], [], [], []

    def scorer(model, directions, mu_a, mu_b, layers, components):
        assert all(torch.all(layer.weight == 1) for layer in model.layers)
        return {f"layers.{l}": torch.ones(1, 10) for l in layers}

    def record():
        candidate = tuple(i for i, layer in enumerate(model.layers)
                          if (layer.weight == 0).any())
        trace.append((candidate, tuple(tuple(torch.where(layer.weight.flatten() == 0)[0].tolist())
                                       for layer in model.layers)))
        return candidate

    def ppl(candidate):
        ppl_calls.append(candidate)
        return 20. if candidate in infeasible else 10.

    def measure():
        candidate = record()
        return sum(outcomes[candidate]) / len(outcomes[candidate]), ppl(candidate)

    def bounded_measure(incumbent):
        candidate = record()
        incumbents.append(incumbent)
        if fail:
            raise RuntimeError("evaluation failed")
        values = outcomes[candidate]
        positives = 0
        if can_prune_binary_rate(0, len(values), incumbent):
            evaluated.append(0)
            return None
        for start in range(0, len(values), batch_size):
            positives += sum(values[start:start + batch_size])
            if can_prune_binary_rate(positives, len(values), incumbent):
                evaluated.append(min(start + batch_size, len(values)))
                return None
        evaluated.append(len(values))
        return positives / len(values), ppl(candidate)

    try:
        selected = bestfirst_layers(model, {}, {}, {}, [0, 1], "both", measure,
                                    base_metric, 10., eps=eps, test_frac=.1,
                                    score_fn=scorer,
                                    bounded_measure=bounded_measure if bounded else None)
    finally:
        assert all(torch.all(layer.weight == 1) for layer in model.layers)
    return selected, trace, evaluated, incumbents, ppl_calls


@pytest.mark.parametrize("eps", [0., .5])
@pytest.mark.parametrize("infeasible", [(), ((0,),)])
def test_exhaustive_binary_outcomes_match_full_search(eps, infeasible):
    # Every assignment of two binary outcomes to all three possible pools.
    binary_rows = list(product([0, 1], repeat=2))
    for rows in product(binary_rows, repeat=3):
        outcomes = dict(zip([(0,), (1,), (0, 1)], rows))
        full = run_search(outcomes, bounded=False, eps=eps, infeasible=infeasible)
        bound = run_search(outcomes, bounded=True, eps=eps, infeasible=infeasible)
        assert bound[:2] == full[:2]  # Selected layers AND every exact candidate mask.


def test_infeasible_candidate_does_not_update_incumbent():
    outcomes = {(0,): [0, 0], (1,): [0, 1], (0, 1): [0, 1]}
    result = run_search(outcomes, bounded=True, infeasible=((0,),))
    assert result[0] == [1]
    assert result[3][:2] == [1., 1.]


@pytest.mark.parametrize("batch_size", [7, 42])
def test_elimination_at_partial_or_full_42_skips_ppl(batch_size):
    outcomes = {key: [1] * 42 for key in [(0,), (1,), (0, 1)]}
    result = run_search(outcomes, bounded=True, base_metric=.5, batch_size=batch_size)
    assert result[0] == []
    assert result[2] == [21 if batch_size == 7 else 42] * 2
    assert result[4] == []


def test_zero_incumbent_keeps_candidate_trace_without_questions():
    outcomes = {key: [0] * 42 for key in [(0,), (1,), (0, 1)]}
    full = run_search(outcomes, bounded=False, base_metric=0.)
    bounded = run_search(outcomes, bounded=True, base_metric=0.)
    assert bounded[:2] == full[:2]
    assert bounded[2] == [0, 0]
    assert bounded[4] == []


def test_callback_exception_restores_weights():
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run_search({}, bounded=True, fail=True)
