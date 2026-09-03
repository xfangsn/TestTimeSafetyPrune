"""Tests for forward-only contrastive refusal-flow scoring."""

import torch

from ttsafety.refusal_flow import (
    _merge_batch_moments,
    crfp_matrix_score,
    response_prediction_span,
)


def test_response_prediction_span_applies_causal_shift():
    prompt = [10, 11, 12]
    full = [10, 11, 12, 20, 21, 99]
    assert response_prediction_span(prompt, full) == (2, 5)


def test_response_prediction_span_handles_boundary_retokenization():
    prompt = [10, 11, 12]
    full = [10, 11, 42, 20, 99]
    assert response_prediction_span(prompt, full) == (1, 4)


def test_parallel_welford_matches_reference():
    torch.manual_seed(0)
    values = torch.randn(11, 7)
    count = 0
    mean = torch.zeros(7)
    m2 = torch.zeros(7)
    for chunk in (values[:3], values[3:8], values[8:]):
        count, mean, m2 = _merge_batch_moments(count, mean, m2, chunk)
    assert count == len(values)
    assert torch.allclose(mean, values.mean(0), atol=1e-6)
    assert torch.allclose(m2 / (count - 1), values.var(0), atol=1e-6)


def test_edge_flow_sums_to_directional_write():
    weight = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    direction = torch.tensor([0.6, 0.8])
    delta = torch.tensor([2.0, -1.0])
    edge_flow = direction[:, None] * weight * delta[None, :]
    assert torch.allclose(edge_flow.sum(), direction @ (weight @ delta))


def test_deleting_edge_changes_direct_flow_by_negative_edge_value():
    weight = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    direction = torch.tensor([0.6, 0.8])
    delta = torch.tensor([2.0, -1.0])
    row, col = 1, 0
    before = direction @ (weight @ delta)
    edited = weight.clone()
    edited[row, col] = 0
    after = direction @ (edited @ delta)
    edge = direction[row] * weight[row, col] * delta[col]
    assert torch.allclose(after - before, -edge)


def test_crfp_score_is_nonnegative_and_filters_by_benefit():
    weight = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
    paired = {
        "count": 4,
        "mean": torch.tensor([1.0, -1.0]),
        "variance": torch.zeros(2),
        "positive_fraction": torch.tensor([1.0, 0.0]),
    }
    score, info = crfp_matrix_score(
        weight,
        torch.tensor([1.0, 0.0]),
        paired,
        torch.ones(2),
        eligibility_fraction=1.0,
    )
    assert score.dtype == torch.float16
    assert torch.all(score >= 0)
    assert score[0, 0] > 0
    assert score[0, 1] == 0
    assert score[1].sum() == 0
    assert info["positive_benefit_fraction"] == 0.25


def test_lcb_can_remove_uncertain_edge():
    weight = torch.ones(1, 1)
    paired = {
        "count": 4,
        "mean": torch.tensor([0.1]),
        "variance": torch.tensor([4.0]),
        "positive_fraction": torch.tensor([0.5]),
    }
    score, info = crfp_matrix_score(
        weight, torch.ones(1), paired, torch.ones(1), beta=1.0
    )
    assert score.item() == 0
    assert info["positive_benefit_fraction"] == 0

