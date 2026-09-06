"""Pure CPU tests for full-pool backward ELS decisions and restart state."""

import pytest

from ttsafety.els_backward import EvaluationBudgetExceeded, backward_layers


class TableEvaluator:
    def __init__(self, table, budget=None):
        self.table = table
        self.budget = budget
        self.calls = []

    def __call__(self, layers, incumbent=None, need_metric=True):
        if self.budget is not None and len(self.calls) >= self.budget:
            raise EvaluationBudgetExceeded
        self.calls.append((layers, incumbent, need_metric))
        ppl, metric = self.table[layers]
        return {"ppl": ppl, "metric": metric if need_metric else None}


def test_infeasible_start_repairs_by_ppl_only_then_searches_by_metric():
    table = {
        (0, 1, 2): (12.0, .8),
        (1, 2): (11.0, .6),   # remove 0
        (0, 2): (10.8, .7),   # remove 1: repair winner, still infeasible
        (0, 1): (10.9, .2),   # tempting metric must not affect repair
        (2,): (10.2, .4),     # remove 0
        (0,): (10.3, .1),     # remove 2: feasible but not minimum PPL
    }
    evaluator = TableEvaluator(table)
    result = backward_layers([2, 0, 1], evaluator, base_metric=1., base_ppl=10.)

    assert result.status == "complete"
    assert result.layers == (2,)
    assert [(step.phase, step.removed_layer) for step in result.steps] == [
        ("repair", 1), ("repair", 0)]
    assert all(not need_metric for _, _, need_metric in evaluator.calls[:6])
    assert evaluator.calls[6] == ((2,), None, True)


def test_repair_ties_use_ascending_removed_layer_and_can_stall():
    table = {
        (0, 1, 2): (12., .5),
        (1, 2): (11.999995, .2),
        (0, 2): (11.999995, .1),
        (0, 1): (12.1, .0),
    }
    result = backward_layers([2, 1, 0], TableEvaluator(table),
                             base_metric=1., base_ppl=10.)
    assert result.status == "repair_stalled"
    assert result.layers == (0, 1, 2)
    assert result.best_feasible_layers is None


def test_feasible_search_requires_strict_eps_and_metric_ties_remove_low_layer():
    table = {
        (0, 1, 2): (10., .8),
        (1, 2): (10., .6),
        (0, 2): (10., .6),
        (0, 1): (10., .7),
        (2,): (10., .595),
        (1,): (10., .595),
    }
    result = backward_layers([0, 1, 2], TableEvaluator(table),
                             base_metric=1., base_ppl=10., eps=.005)
    assert result.layers == (1, 2)
    assert [step.removed_layer for step in result.steps] == [0]
    assert result.metric == .6
    # The better observed candidate does not become accepted without a full
    # round winner exceeding eps, but is retained for truncated-run reporting.
    assert result.best_feasible_layers == (1,)


def test_metric_bound_none_is_skipped_and_gets_round_incumbent():
    calls = []
    table = {(0, 1): (10., .8), (1,): (10., .5), (0,): (10., .4)}

    def evaluate(layers, incumbent=None, need_metric=True):
        calls.append((layers, incumbent, need_metric))
        ppl, metric = table[layers]
        if need_metric and layers == (0,):
            return {"ppl": ppl, "metric": None}
        return {"ppl": ppl, "metric": metric if need_metric else None}

    result = backward_layers([0, 1], evaluate, base_metric=1., base_ppl=10.)
    assert result.layers == (1,)
    assert ((0,), .5, True) in calls


def test_budget_returns_distinct_infeasible_partial_without_swallowing_errors():
    table = {(0, 1): (12., .8), (1,): (11., .5), (0,): (10., .4)}
    result = backward_layers([0, 1], TableEvaluator(table, budget=2),
                             base_metric=1., base_ppl=10.)
    assert result.status == "truncated"
    assert not result.feasible
    assert result.best_feasible_layers is None

    def broken(*args, **kwargs):
        raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        backward_layers([0], broken, base_metric=1., base_ppl=10.)


def test_nonfinite_initial_ppl_is_evaluation_failure():
    result = backward_layers(
        [0], lambda *a, **k: {"ppl": float("nan"), "metric": None},
        base_metric=1., base_ppl=10.)
    assert result.status == "evaluation_failed"
    assert not result.feasible


def test_progress_can_truncate_after_recording_best_feasible_candidate():
    table = {(0, 1): (10., .8), (1,): (10., .5), (0,): (10., .4)}
    snapshots = []

    def progress(snapshot):
        snapshots.append(snapshot)
        if snapshot.best_feasible_layers == (1,):
            raise EvaluationBudgetExceeded

    result = backward_layers([0, 1], TableEvaluator(table), base_metric=1.,
                             base_ppl=10., progress=progress)
    assert result.status == "truncated"
    assert result.layers == (0, 1)
    assert result.best_feasible_layers == (1,)
    assert snapshots[-1].candidate_layers == (1,)


def test_empty_pool_is_baseline_without_evaluation():
    def forbidden(*args, **kwargs):
        raise AssertionError("empty pool must not be evaluated")

    result = backward_layers([], forbidden, base_metric=.25, base_ppl=10.)
    assert result.status == "complete"
    assert result.layers == ()
    assert result.feasible
    assert result.metric == .25
    assert result.evaluation_calls == 0


def test_feasibility_uses_same_relative_expression_as_forward_els():
    # In binary float, (1.05 - 1) / 1 is just above .05 even though
    # 1.05 <= 1 * (1 + .05).  Preserve the existing forward ELS comparison.
    evaluator = TableEvaluator({(0,): (1.05, .5)})
    result = backward_layers([0], evaluator, base_metric=1., base_ppl=1.,
                             beta=.05)
    assert result.layers == ()
    assert result.steps[0].phase == "repair"


@pytest.mark.parametrize("pool", [[1, 1], [True], [1.5]])
def test_invalid_pool(pool):
    with pytest.raises(ValueError):
        backward_layers(pool, lambda *a, **k: {}, base_metric=1., base_ppl=10.)
