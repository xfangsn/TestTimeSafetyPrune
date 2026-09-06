"""Exact elimination certificates for minimizing a fixed binary outcome rate."""

from math import isfinite
from numbers import Integral


def can_prune_binary_rate(positives: int, total: int, incumbent: float) -> bool:
    """Whether observed positives already rule out a strict improvement.

    ``total`` is the fixed FULL dataset size, never the evaluated prefix size.
    Evaluate the original batches in their original order, accumulating binary
    positives, and call this after each batch (or before any batch). Unseen
    outcomes could all be zero, so positives / total is a lower bound, not an
    estimated or complete rate. Ties cannot win bestfirst_layers' strict <.

    Compare the division directly, as for a complete rate. Multiplication and
    ceil-based integer thresholds can change decisions at floating boundaries.
    This helper does not compute PPL or certify PPL feasibility.
    """
    if isinstance(total, bool) or not isinstance(total, Integral) or total <= 0:
        raise ValueError("total must be a positive integer full dataset size")
    if (isinstance(positives, bool) or not isinstance(positives, Integral)
            or not 0 <= positives <= total):
        raise ValueError("positives must be an integer between zero and total")
    if not isfinite(incumbent) or not 0 <= incumbent <= 1:
        raise ValueError("incumbent must be a finite binary rate in [0, 1]")
    return positives / total >= incumbent
