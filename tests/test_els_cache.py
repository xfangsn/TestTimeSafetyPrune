"""Exact ELS cache equivalence, including topk's order-dependent ties."""

from collections import Counter
from functools import partial

import pytest
import torch
from torch import nn

from ttsafety.els_cache import ELSCandidateCache
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking


def assert_same(left, right):
    assert left.keys() == right.keys()
    for name in left:
        if isinstance(left[name], torch.Tensor):
            assert torch.equal(left[name], right[name]), name
        else:
            assert left[name] == right[name], name


@pytest.mark.parametrize("kind", ["random", "ties", "zeros", "negative_inf"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("largest", [True, False])
def test_exact_multi_layer_rankings_and_selections(kind, dtype, largest):
    generator = torch.Generator().manual_seed(37)
    scores = {}
    for layer in [0, 2, 10]:
        scores[layer] = {}
        for component, size in [("z", 117), ("a", 51)]:
            score = torch.randn(size, generator=generator)
            if kind == "ties":
                score = score.round()
            elif kind == "zeros":
                score.zero_()
            elif kind == "negative_inf":
                score.fill_(-float("inf"))
                score[:2] = 1
            scores[layer][f"layers.{layer}.{component}"] = score.to(dtype)
    calls = Counter()

    def scorer(model, directions, mu_a, mu_b, layers, components):
        calls.update(layers)
        return {n: s for l in layers for n, s in scores[l].items()}

    cache = ELSCandidateCache(scorer, largest=largest)
    args = (object(), {}, {}, {})
    pools = [[10], [2, 0], [10, 2, 0], [0, 10], [2], [0, 2, 10]]
    for layers in pools:
        expected_scores = {n: s for l in layers for n, s in scores[l].items()}
        for maximum in [0.005, 0.01, 0.057, 0.1]:
            actual = cache.rank(*args, layers, ["z", "a"], maximum)
            expected = rank_weight_indices(expected_scores, maximum, largest=largest)
            assert_same(actual, expected)
            for fraction in [maximum / 5, maximum / 2, maximum]:
                assert_same(selection_from_ranking(actual, fraction),
                            selection_from_ranking(expected, fraction))
    assert calls == {0: 1, 2: 1, 10: 1}
    assert cache.stats["misses"] == 3
    assert cache.stats["hits"] == sum(map(len, pools)) * 4 - 3
    assert cache.stats["rank_calls"] == len(pools) * 4
    assert cache.stats["cached_bytes"] == 3 * (11 + 5) * (4 + 8)
    for entries in cache._layers.values():
        for entry in entries.values():
            assert entry.values.dtype == torch.float32
            assert entry.values.device.type == "cpu"
    # Results cannot mutate stored candidates, and clear forces fresh scoring.
    actual["flat_indices"].zero_()
    cache.clear()
    assert cache.stats["cached_bytes"] == 0
    assert cache.stats["cached_layers"] == 0
    actual = cache(*args, [10, 2, 0], ["z", "a"], 0.1)
    assert calls == {0: 2, 2: 2, 10: 2}
    assert_same(actual, expected)


def test_reject_changed_binding_and_invalid_fraction():
    cache = ELSCandidateCache(lambda *args: {"weight": torch.ones(31)})
    args = (object(), {}, {}, {})
    cache(*args, [0], ["mlp"], 0.1)
    for index in range(4):
        changed = list(args)
        changed[index] = object()
        with pytest.raises(ValueError, match="inputs changed"):
            cache(*changed, [0], ["mlp"], 0.1)
    with pytest.raises(ValueError, match="inputs changed"):
        cache(*args, [0], ["attn"], 0.1)
    for fraction in [0, -0.1, 0.11]:
        with pytest.raises(ValueError, match="require"):
            cache(*args, [0], ["mlp"], fraction)


def test_els_helpers_match_and_score_original_weights_only():
    from ttsafety.behaviors import bestfirst_layers, solo_layer_pool

    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(20, 10, bias=False) for _ in range(3)])
    with torch.no_grad():
        for module in model.layers:
            module.weight.fill_(1)
    calls = Counter()

    def scorer(model, directions, mu_a, mu_b, layers, components):
        assert all(torch.all(module.weight == 1) for module in model.layers)
        calls.update(layers)
        return {f"layers.{l}": torch.ones_like(model.layers[l].weight) for l in layers}

    def measure():
        zeros = sum((module.weight == 0).sum().item() for module in model.layers)
        return 1 - zeros / 100, 10.0

    args = (model, {}, {}, {})

    def run(ranking_fn=None):
        pool = solo_layer_pool(*args, [0, 1, 2], ["mlp"], lambda: 10., 10.,
                               score_fn=scorer, ranking_fn=ranking_fn)
        selected = bestfirst_layers(*args, pool, ["mlp"], measure, 1., 10.,
                                    score_fn=scorer, ranking_fn=ranking_fn)
        return pool, selected

    reference = run()
    calls.clear()
    cache = ELSCandidateCache(scorer)
    assert run(cache.rank) == reference == ([0, 1, 2], [0, 1, 2])
    assert calls == {0: 1, 1: 1, 2: 1}
    assert all(torch.all(module.weight == 1) for module in model.layers)


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("components", ["both", "mlp", "attn"])
def test_real_scorer_with_string_components(generic, components):
    from ttsafety.sycophancy import score_edges, score_edges_g
    from ttsafety.weight_edit import iter_residual_writers

    generator = torch.Generator().manual_seed(5)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    for _ in range(3):
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.down_proj = nn.Linear(16, 8, bias=False)
        block.self_attn = nn.Module()
        block.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        model.model.layers.append(block)
    directions = {l: torch.randn(8, generator=generator) for l in range(3)}
    writers = dict(iter_residual_writers(model, [0, 1, 2], components))
    mu_a = {n: torch.randn(w.in_features, generator=generator) for n, w in writers.items()}
    mu_b = {n: torch.zeros(w.in_features) for n, w in writers.items()}
    Q = {n: torch.rand(w.weight.shape, generator=generator) for n, w in writers.items()}
    scorer = partial(score_edges_g, Q=Q, lam=0.01) if generic else score_edges
    cache = ELSCandidateCache(scorer)
    for layers in [[2], [0, 2], [1, 2], [0, 1, 2]]:
        actual = cache.rank(model, directions, mu_a, mu_b, layers, components, 0.1)
        expected = rank_weight_indices(
            scorer(model, directions, mu_a, mu_b, layers, components), 0.1)
        assert_same(actual, expected)
        assert_same(selection_from_ranking(actual, 0.05),
                    selection_from_ranking(expected, 0.05))
    assert cache.stats["misses"] == 3
