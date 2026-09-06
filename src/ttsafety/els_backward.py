"""Pure full-pool backward elimination for effective-layer selection.

Model mutation, measurement caching, and request budgets belong to the
``evaluate`` callback.  Keeping them outside this module makes a stopped run
restartable by replaying a persistent evaluator cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping, Optional, Sequence


class EvaluationBudgetExceeded(RuntimeError):
    """Raised by an evaluator or progress callback when its budget is spent."""


@dataclass(frozen=True)
class LayerEvaluation:
    ppl: float
    metric: Optional[float] = None


@dataclass(frozen=True)
class BackwardStep:
    phase: str
    removed_layer: int
    layers: tuple[int, ...]
    ppl: float
    metric: Optional[float]


@dataclass(frozen=True)
class BackwardProgress:
    phase: str
    current_layers: tuple[int, ...]
    current_ppl: float
    current_metric: Optional[float]
    best_feasible_layers: Optional[tuple[int, ...]]
    best_feasible_ppl: Optional[float]
    best_feasible_metric: Optional[float]
    candidate_layers: Optional[tuple[int, ...]]
    removed_layer: Optional[int]
    evaluation_calls: int


@dataclass(frozen=True)
class BackwardResult:
    status: str
    layers: tuple[int, ...]
    ppl: float
    metric: Optional[float]
    feasible: bool
    best_feasible_layers: Optional[tuple[int, ...]]
    best_feasible_ppl: Optional[float]
    best_feasible_metric: Optional[float]
    steps: tuple[BackwardStep, ...]
    evaluation_calls: int


Evaluate = Callable[..., Mapping[str, object] | LayerEvaluation]
Progress = Callable[[BackwardProgress], None]


def backward_layers(
    pool: Sequence[int],
    evaluate: Evaluate,
    *,
    base_metric: float,
    base_ppl: float,
    beta: float = 0.05,
    eps: float = 0.005,
    ppl_improvement_tol: float = 1e-6,
    progress: Optional[Progress] = None,
) -> BackwardResult:
    """Run full-pool backward ELS and return its accepted path.

    ``evaluate(layers, incumbent=None, need_metric=True)`` receives normalized
    layer tuples.  A PPL-only call uses ``need_metric=False``.  A metric call
    may return ``metric=None`` only to certify that the candidate cannot beat
    its supplied incumbent.  Non-finite measurements make a candidate fail.

    Repair chooses the minimum ``(ppl, removed_layer)`` over every deletion.
    Once feasible, search chooses minimum ``(metric, removed_layer)`` and only
    accepts a strict improvement beyond ``eps``.  The empty set is the supplied
    baseline and never invokes ``evaluate``.
    """
    layers = _normalize_pool(pool)
    _validate_config(base_metric, base_ppl, beta, eps, ppl_improvement_tol)
    calls = 0
    steps: list[BackwardStep] = []
    current_ppl = base_ppl if not layers else float("nan")
    current_metric: Optional[float] = base_metric if not layers else None
    best: Optional[tuple[tuple[int, ...], float, float]] = (
        ((), base_ppl, base_metric) if not layers else None
    )
    phase = "initial"
    candidate: Optional[tuple[int, ...]] = None
    removed: Optional[int] = None

    def call(candidate_layers: tuple[int, ...], *, incumbent: Optional[float],
             need_metric: bool) -> LayerEvaluation:
        nonlocal calls
        if not candidate_layers:
            return LayerEvaluation(base_ppl, base_metric if need_metric else None)
        calls += 1
        raw = evaluate(candidate_layers, incumbent=incumbent,
                       need_metric=need_metric)
        return _coerce_evaluation(raw)

    def update_best(candidate_layers: tuple[int, ...], result: LayerEvaluation) -> None:
        nonlocal best
        if (result.metric is None or not _feasible(result.ppl, base_ppl, beta)
                or not isfinite(result.metric)):
            return
        key = (result.metric, candidate_layers)
        if best is None or key < (best[2], best[0]):
            best = (candidate_layers, result.ppl, result.metric)

    def notify(progress_phase: str) -> None:
        if progress is None:
            return
        progress(BackwardProgress(
            phase=progress_phase,
            current_layers=layers,
            current_ppl=current_ppl,
            current_metric=current_metric,
            best_feasible_layers=None if best is None else best[0],
            best_feasible_ppl=None if best is None else best[1],
            best_feasible_metric=None if best is None else best[2],
            candidate_layers=candidate,
            removed_layer=removed,
            evaluation_calls=calls,
        ))

    try:
        if not layers:
            notify("done")
            return _result("complete", layers, current_ppl, current_metric,
                           True, best, steps, calls)

        initial = call(layers, incumbent=None, need_metric=False)
        current_ppl = initial.ppl
        notify("initial")
        if not isfinite(current_ppl):
            return _result("evaluation_failed", layers, current_ppl, None,
                           False, best, steps, calls)
        if _feasible(current_ppl, base_ppl, beta):
            measured = call(layers, incumbent=None, need_metric=True)
            current_ppl = measured.ppl
            current_metric = measured.metric
            if not _complete_feasible(measured, base_ppl, beta):
                return _result("evaluation_failed", layers, current_ppl,
                               current_metric, False, best, steps, calls)
            update_best(layers, measured)
            notify("initial")
            phase = "search"
        else:
            phase = "repair"

        while phase == "repair":
            repair_candidates: list[tuple[float, int, tuple[int, ...]]] = []
            for removed in layers:
                candidate = tuple(layer for layer in layers if layer != removed)
                result = call(candidate, incumbent=None, need_metric=False)
                if isfinite(result.ppl):
                    repair_candidates.append((result.ppl, removed, candidate))
                notify("repair")
            if not repair_candidates:
                return _result("repair_stalled", layers, current_ppl, None,
                               False, best, steps, calls)
            new_ppl, removed, candidate = min(repair_candidates)
            if (not _feasible(new_ppl, base_ppl, beta)
                    and (current_ppl - new_ppl) / base_ppl <= ppl_improvement_tol):
                return _result("repair_stalled", layers, current_ppl, None,
                               False, best, steps, calls)
            layers, current_ppl = candidate, new_ppl
            steps.append(BackwardStep("repair", removed, layers,
                                      current_ppl, None))
            notify("repair")
            if _feasible(current_ppl, base_ppl, beta):
                measured = call(layers, incumbent=None, need_metric=True)
                current_ppl, current_metric = measured.ppl, measured.metric
                if not _complete_feasible(measured, base_ppl, beta):
                    return _result("evaluation_failed", layers, current_ppl,
                                   current_metric, False, best, steps, calls)
                update_best(layers, measured)
                notify("repair")
                phase = "search"

        while layers:
            assert current_metric is not None
            round_best: Optional[tuple[float, int, tuple[int, ...], float]] = None
            for removed in layers:
                candidate = tuple(layer for layer in layers if layer != removed)
                ppl_result = call(candidate, incumbent=None, need_metric=False)
                if _feasible(ppl_result.ppl, base_ppl, beta):
                    incumbent = current_metric if round_best is None else round_best[0]
                    metric_result = call(candidate, incumbent=incumbent,
                                         need_metric=True)
                    if _complete_feasible(metric_result, base_ppl, beta):
                        update_best(candidate, metric_result)
                        item = (metric_result.metric, removed, candidate,
                                metric_result.ppl)
                        if round_best is None or item[:2] < round_best[:2]:
                            round_best = item  # type: ignore[assignment]
                notify("search")
            if round_best is None or not round_best[0] < current_metric - eps:
                notify("done")
                return _result("complete", layers, current_ppl, current_metric,
                               True, best, steps, calls)
            current_metric, removed, layers, current_ppl = round_best
            steps.append(BackwardStep("search", removed, layers, current_ppl,
                                      current_metric))
            notify("search")

        notify("done")
        return _result("complete", layers, current_ppl, current_metric, True,
                       best, steps, calls)
    except EvaluationBudgetExceeded:
        return _result("truncated", layers, current_ppl, current_metric,
                       _feasible(current_ppl, base_ppl, beta)
                       and current_metric is not None
                       and isfinite(current_metric),
                       best, steps, calls)


def _normalize_pool(pool: Sequence[int]) -> tuple[int, ...]:
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in pool):
        raise ValueError("pool layers must be integers")
    if len(set(pool)) != len(pool):
        raise ValueError("pool layers must be unique")
    return tuple(sorted(pool))


def _validate_config(base_metric: float, base_ppl: float, beta: float,
                     eps: float, ppl_improvement_tol: float) -> None:
    values = (base_metric, base_ppl, beta, eps, ppl_improvement_tol)
    if not all(isfinite(value) for value in values):
        raise ValueError("search configuration must be finite")
    if base_ppl <= 0 or beta < 0 or eps < 0 or ppl_improvement_tol < 0:
        raise ValueError("base_ppl must be positive and tolerances nonnegative")


def _coerce_evaluation(raw: Mapping[str, object] | LayerEvaluation) -> LayerEvaluation:
    if isinstance(raw, LayerEvaluation):
        return raw
    try:
        ppl = float(raw["ppl"])
        metric_raw = raw.get("metric")
        metric = None if metric_raw is None else float(metric_raw)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("evaluate must return numeric ppl and optional metric") from exc
    return LayerEvaluation(ppl, metric)


def _feasible(ppl: float, base_ppl: float, beta: float) -> bool:
    return isfinite(ppl) and (ppl - base_ppl) / base_ppl <= beta


def _complete_feasible(result: LayerEvaluation, base_ppl: float,
                       beta: float) -> bool:
    return (_feasible(result.ppl, base_ppl, beta) and result.metric is not None
            and isfinite(result.metric))


def _result(status: str, layers: tuple[int, ...], ppl: float,
            metric: Optional[float], feasible: bool,
            best: Optional[tuple[tuple[int, ...], float, float]],
            steps: list[BackwardStep], calls: int) -> BackwardResult:
    return BackwardResult(
        status=status, layers=layers, ppl=ppl, metric=metric, feasible=feasible,
        best_feasible_layers=None if best is None else best[0],
        best_feasible_ppl=None if best is None else best[1],
        best_feasible_metric=None if best is None else best[2],
        steps=tuple(steps), evaluation_calls=calls,
    )
