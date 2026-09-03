"""Reversible unstructured pruning of selected residual-writer weights."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


def flat_to_row_col(flat_indices: torch.Tensor, n_cols: int):
    """Convert flat row-major indices to matrix coordinates."""
    if n_cols <= 0:
        raise ValueError("n_cols must be positive")
    indices = flat_indices.to(torch.long)
    return indices // n_cols, indices % n_cols


def rank_weight_indices(
    scores: dict[str, torch.Tensor],
    max_fraction: float,
    *,
    largest: bool = True,
    per_matrix_cap: float = 0.10,
) -> dict:
    """Build a global ranking while enforcing a per-matrix candidate cap."""
    if not 0 < max_fraction <= per_matrix_cap <= 1:
        raise ValueError("require 0 < max_fraction <= per_matrix_cap <= 1")
    names = sorted(scores)
    total = sum(value.numel() for value in scores.values())
    global_k = max(1, round(max_fraction * total))
    candidate_values = []
    candidate_local = []
    candidate_matrix = []
    for matrix_id, name in enumerate(names):
        flat = scores[name].float().flatten()
        cap = max(1, int(per_matrix_cap * flat.numel()))
        values, local = torch.topk(flat, cap, largest=largest, sorted=False)
        candidate_values.append(values)
        candidate_local.append(local)
        candidate_matrix.append(
            torch.full((cap,), matrix_id, dtype=torch.int16)
        )
    values = torch.cat(candidate_values)
    local = torch.cat(candidate_local)
    matrix_ids = torch.cat(candidate_matrix)
    # per-matrix caps use floor() while global_k uses round(), so global_k can
    # exceed the pooled candidate count (e.g. max_fraction == per_matrix_cap);
    # clamp to avoid torch.topk "k out of range".
    global_k = min(global_k, values.numel())
    _, order = torch.topk(values, global_k, largest=largest, sorted=True)
    return {
        "names": names,
        "matrix_ids": matrix_ids[order],
        "flat_indices": local[order],
        "max_fraction": max_fraction,
        "total_pool_weights": total,
        "largest": largest,
        "per_matrix_cap": per_matrix_cap,
    }


def selection_from_ranking(ranking: dict, fraction: float) -> dict[str, torch.Tensor]:
    """Take a prefix of a global ranking and group indices by matrix."""
    if not 0 < fraction <= ranking["max_fraction"]:
        raise ValueError("fraction outside ranking range")
    k = max(1, round(fraction * ranking["total_pool_weights"]))
    matrix_ids = ranking["matrix_ids"][:k]
    local = ranking["flat_indices"][:k]
    return {
        name: local[matrix_ids == matrix_id]
        for matrix_id, name in enumerate(ranking["names"])
        if (matrix_ids == matrix_id).any()
    }


def matrixwise_set_difference(
    safety_scores: dict[str, torch.Tensor],
    utility_scores: dict[str, torch.Tensor],
    *,
    safety_fraction: float,
    utility_fraction: float,
) -> dict[str, torch.Tensor]:
    """Return matrix-global top-safety weights excluding top-utility weights.

    This follows ``prune_wandg_set_difference`` in the official Wei et al.
    (2024) code: flatten each matrix, form S_s(q) and S_u(p), and return
    S_s(q) \\ S_u(p). Returned indices use flattened row-major order.
    """
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")
    if not 0 < utility_fraction <= 1:
        raise ValueError("utility_fraction must be in (0, 1]")
    if set(safety_scores) != set(utility_scores):
        raise ValueError("safety and utility score keys must match")

    selection = {}
    for name in sorted(safety_scores):
        safety = safety_scores[name]
        utility = utility_scores[name]
        if safety.ndim != 2 or utility.ndim != 2:
            raise ValueError(f"scores for {name!r} must be matrices")
        if safety.shape != utility.shape:
            raise ValueError(f"score shapes differ for {name!r}")
        safety_flat = safety.float().flatten()
        utility_flat = utility.float().flatten()
        k_safety = max(1, int(safety_flat.numel() * safety_fraction))
        k_utility = max(1, int(utility_flat.numel() * utility_fraction))
        safety_indices = torch.topk(
            safety_flat, k_safety, largest=True, sorted=False
        ).indices
        utility_indices = torch.topk(
            utility_flat, k_utility, largest=True, sorted=False
        ).indices
        utility_mask = torch.zeros(
            utility_flat.numel(), dtype=torch.bool, device=utility_indices.device
        )
        utility_mask[utility_indices] = True
        selection[name] = safety_indices[~utility_mask[safety_indices]].cpu()
    return selection


def random_scores_like(scores: dict[str, torch.Tensor], seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.rand(value.shape, generator=generator, dtype=torch.float16)
        for name, value in scores.items()
    }


def _resolve_modules(model: nn.Module, names: list[str]) -> dict[str, nn.Module]:
    modules = dict(model.named_modules())
    resolved = {}
    for name in names:
        module = modules.get(name)
        if module is None:
            matches = [value for key, value in modules.items() if key.endswith(name)]
            if len(matches) != 1:
                raise KeyError(f"cannot uniquely resolve module {name!r}")
            module = matches[0]
        resolved[name] = module
    return resolved


@contextmanager
def pruned_weights(model: nn.Module, selection: dict[str, torch.Tensor]):
    """Zero selected flat weight entries and restore their exact values on exit."""
    modules = _resolve_modules(model, list(selection))
    backups = {}
    with torch.no_grad():
        for name, indices in selection.items():
            weight = modules[name].weight
            device_indices = indices.to(weight.device, torch.long)
            flat = weight.view(-1)
            backups[name] = flat[device_indices].detach().cpu().clone()
            flat[device_indices] = 0
    try:
        yield
    finally:
        with torch.no_grad():
            for name, indices in selection.items():
                weight = modules[name].weight
                device_indices = indices.to(weight.device, torch.long)
                weight.view(-1)[device_indices] = backups[name].to(
                    weight.device, weight.dtype
                )



def make_pruning_factory(model: nn.Module, selection: dict[str, torch.Tensor]):
    """Prepare reusable device indices/backups for paired base/edit metrics."""
    modules = _resolve_modules(model, list(selection))
    prepared = {}
    with torch.no_grad():
        for name, indices in selection.items():
            weight = modules[name].weight
            device_indices = indices.to(weight.device, torch.long)
            prepared[name] = (
                weight,
                device_indices,
                weight.view(-1)[device_indices].detach().clone(),
            )

    @contextmanager
    def context():
        with torch.no_grad():
            for weight, indices, _backup in prepared.values():
                weight.view(-1)[indices] = 0
        try:
            yield
        finally:
            with torch.no_grad():
                for weight, indices, backup in prepared.values():
                    weight.view(-1)[indices] = backup

    return context
