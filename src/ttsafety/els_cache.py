"""Fit-local cache of ELS per-matrix candidates, preserving CPU topk ties.

The wrapped scorer must return immutable CPU scores and be separable by layer
(as score_edges and score_edges_g are). Call only on the restored, original
model, outside pruning contexts. Create a new cache for each research fit or
when weights, directions, moments, Q, lambda, or scorer settings change. Object
identity checks catch replacements, but cannot detect in-place mutations or
changes inside a scorer closure. ``clear()`` explicitly invalidates all state.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass(frozen=True)
class _Candidates:
    values: torch.Tensor
    local: torch.Tensor
    numel: int


class ELSCandidateCache:
    """Lazily score each layer once; recompute global ranking for every pool.

    Pass ``cache.rank`` as ``ranking_fn`` to ELS helpers. The scorer keeps its
    existing six-argument API; bind Q/lam with functools.partial or a closure.
    Values retain the scorer's precision before conversion to FP32, exactly as
    rank_weight_indices does. No per-layer final masks are cached.
    """

    def __init__(self, score_fn, *, largest=True, per_matrix_cap=0.10):
        if not 0 < per_matrix_cap <= 1:
            raise ValueError("per_matrix_cap must be in (0, 1]")
        self._score_fn = score_fn
        self._largest = largest
        self._per_matrix_cap = per_matrix_cap
        self.clear()

    def clear(self):
        """Discard candidates, argument bindings, and benchmark counters."""
        self._layers = {}
        self._binding = None
        self._components = None
        self._stats = dict(hits=0, misses=0, cached_bytes=0, score_seconds=0.0,
                           local_topk_seconds=0.0, global_rank_seconds=0.0,
                           rank_calls=0)

    @property
    def stats(self):
        """Snapshot; hits/misses count layer requests, bytes count tensor storage."""
        return {**self._stats, "cached_layers": len(self._layers)}

    @torch.no_grad()
    def rank(self, model, directions, mu_a, mu_b, layers, components,
             max_fraction):
        """Return the same ranking dictionary as rank_weight_indices."""
        if not 0 < max_fraction <= self._per_matrix_cap <= 1:
            raise ValueError("require 0 < max_fraction <= per_matrix_cap <= 1")
        # Preserve the public scorer's string forms (notably "both").
        components = components if isinstance(components, str) else tuple(components)
        binding = (model, directions, mu_a, mu_b)
        if self._binding is None:
            self._binding, self._components = binding, components
        elif (any(old is not new for old, new in zip(self._binding, binding))
              or components != self._components):
            raise ValueError("cache inputs changed; create a new cache or clear()")

        candidates = {}
        for layer in dict.fromkeys(layers):
            if layer in self._layers:
                self._stats["hits"] += 1
            else:
                self._stats["misses"] += 1
                start = perf_counter()
                scores = self._score_fn(model, directions, mu_a, mu_b,
                                        [layer], components)
                self._stats["score_seconds"] += perf_counter() - start
                start = perf_counter()
                entries = {}
                for name in sorted(scores):
                    score = scores[name]
                    if score.device.type != "cpu":
                        raise ValueError("ELS cache requires CPU scorer outputs")
                    flat = score.float().flatten()
                    cap = max(1, int(self._per_matrix_cap * flat.numel()))
                    values, local = torch.topk(flat, cap, largest=self._largest,
                                              sorted=False)
                    entries[name] = _Candidates(values, local, flat.numel())
                self._stats["local_topk_seconds"] += perf_counter() - start
                self._layers[layer] = entries
                self._stats["cached_bytes"] += sum(
                    e.values.numel() * e.values.element_size()
                    + e.local.numel() * e.local.element_size()
                    for e in entries.values())
            candidates.update(self._layers[layer])

        start = perf_counter()
        names = sorted(candidates)
        total = sum(e.numel for e in candidates.values())
        values = torch.cat([candidates[n].values for n in names])
        local = torch.cat([candidates[n].local for n in names])
        matrix_ids = torch.cat([
            torch.full((candidates[n].values.numel(),), i, dtype=torch.int16)
            for i, n in enumerate(names)])
        global_k = min(max(1, round(max_fraction * total)), values.numel())
        _, order = torch.topk(values, global_k, largest=self._largest, sorted=True)
        ranking = {
            "names": names,
            "matrix_ids": matrix_ids[order],
            "flat_indices": local[order],
            "max_fraction": max_fraction,
            "total_pool_weights": total,
            "largest": self._largest,
            "per_matrix_cap": self._per_matrix_cap,
        }
        self._stats["global_rank_seconds"] += perf_counter() - start
        self._stats["rank_calls"] += 1
        return ranking

    __call__ = rank
