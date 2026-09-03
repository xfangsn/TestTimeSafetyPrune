"""Tests for global individual-weight ranking and reversible pruning."""

import torch
from torch import nn

from ttsafety.weight_prune import (
    flat_to_row_col,
    make_pruning_factory,
    pruned_weights,
    random_scores_like,
    rank_weight_indices,
    matrixwise_set_difference,
    selection_from_ranking,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 3, bias=False)
        self.second = nn.Linear(4, 2, bias=False)


def test_flat_index_roundtrip():
    flat = torch.tensor([0, 3, 4, 11])
    rows, cols = flat_to_row_col(flat, 4)
    assert torch.equal(rows * 4 + cols, flat)


def test_ranking_and_matrix_cap():
    scores = {
        "first": torch.arange(12).view(3, 4).float(),
        "second": torch.arange(8).view(2, 4).float() + 100,
    }
    ranking = rank_weight_indices(
        scores, 0.10, largest=True, per_matrix_cap=0.5
    )
    selected = selection_from_ranking(ranking, 0.10)
    assert sum(x.numel() for x in selected.values()) == 2
    assert set(selected) == {"second"}


def test_selected_weights_zero_and_restore_exactly():
    torch.manual_seed(0)
    model = TinyModel()
    before = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    selection = {
        "first": torch.tensor([0, 5, 11]),
        "second": torch.tensor([2]),
    }
    with pruned_weights(model, selection):
        assert torch.equal(model.first.weight.view(-1)[selection["first"]], torch.zeros(3))
        assert torch.equal(model.second.weight.view(-1)[selection["second"]], torch.zeros(1))
    assert torch.equal(model.first.weight, before["first"])
    assert torch.equal(model.second.weight, before["second"])


def test_random_scores_reproducible():
    template = {"x": torch.empty(3, 4), "y": torch.empty(2, 2)}
    left = random_scores_like(template, 7)
    right = random_scores_like(template, 7)
    other = random_scores_like(template, 8)
    assert all(torch.equal(left[key], right[key]) for key in left)
    assert any(not torch.equal(left[key], other[key]) for key in left)


def test_matrixwise_set_difference_matches_official_flattened_rule():
    safety = {
        "x": torch.tensor([
            [9.0, 8.0, 1.0, 0.0],
            [0.0, 7.0, 6.0, 1.0],
        ])
    }
    utility = {
        "x": torch.tensor([
            [10.0, 0.0, 0.0, 1.0],
            [0.0, 9.0, 1.0, 0.0],
        ])
    }
    selected = matrixwise_set_difference(
        safety, utility, safety_fraction=0.5, utility_fraction=0.25
    )
    # Matrix-global safety top-4 are {0,1,5,6}; utility top-2 protect {0,5}.
    assert torch.equal(torch.sort(selected["x"]).values, torch.tensor([1, 6]))


def test_matrixwise_set_difference_validates_matching_inputs():
    scores = {"x": torch.ones(2, 3)}
    try:
        matrixwise_set_difference(
            scores, {"y": torch.ones(2, 3)},
            safety_fraction=0.5, utility_fraction=0.5,
        )
    except ValueError as exc:
        assert "keys" in str(exc)
    else:
        raise AssertionError("mismatched keys should fail")



def test_reusable_factory_restores_exactly_across_entries():
    torch.manual_seed(3)
    model = TinyModel()
    selection = {"first": torch.tensor([1, 7])}
    before = model.first.weight.detach().clone()
    factory = make_pruning_factory(model, selection)
    for _ in range(2):
        with factory():
            assert torch.equal(
                model.first.weight.view(-1)[selection["first"]],
                torch.zeros(2),
            )
        assert torch.equal(model.first.weight, before)
