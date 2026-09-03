"""Weight-level refusal-direction edit tests."""

import pytest
import torch
from torch import nn

from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, load_model
from ttsafety.weight_edit import (
    iter_residual_writers,
    orthogonalized_weights,
    project_embeddings,
    project_residual_writes,
    project_weight,
    replace_residual_writes,
)


def test_project_weight_formula_and_orthogonality():
    torch.manual_seed(0)
    weight = torch.randn(7, 11)
    direction = torch.randn(7)
    unit = direction / direction.norm()
    edited = project_weight(weight, direction, strength=1.0)
    expected = weight - unit[:, None] * (unit @ weight)[None, :]
    assert torch.allclose(edited, expected, atol=1e-6)
    assert (unit @ edited).abs().max() < 1e-5


def test_project_weight_partial_and_norm_preserving():
    torch.manual_seed(1)
    weight = torch.randn(5, 9)
    direction = torch.randn(5)
    half = project_weight(weight, direction, strength=0.5)
    full = project_weight(weight, direction, strength=1.0)
    assert torch.allclose(half, (weight + full) / 2, atol=1e-6)
    preserved = project_weight(weight, direction, strength=1.0, norm_preserve=True)
    assert torch.allclose(
        preserved.float().norm(dim=0), weight.norm(dim=0), atol=1e-5
    )


def test_invalid_inputs():
    with pytest.raises(ValueError):
        project_weight(torch.randn(3, 4), torch.zeros(3))
    with pytest.raises(ValueError):
        project_weight(torch.randn(3, 4), torch.ones(3), strength=1.1)


@torch.no_grad()
def _logits(model, tokenizer):
    prompt = chat_wrap(tokenizer, "What is the capital of France?")
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    return model(**enc, use_cache=False).logits


def test_writer_discovery_shapes():
    model, _ = load_model()
    writers = dict(iter_residual_writers(model, [8], "both"))
    assert writers["layers.8.mlp.down_proj"].weight.shape == (3072, 8192)
    assert writers["layers.8.self_attn.o_proj"].weight.shape == (3072, 3072)


def test_hook_changes_output_and_restores():
    model, tokenizer = load_model()
    direction = torch.load(
        "data/directions/refusal_llama32_3b_instruct.pt", weights_only=True
    )[8]
    base = _logits(model, tokenizer)
    with project_residual_writes(model, direction, [8], "both", 1.0):
        edited = _logits(model, tokenizer)
    after = _logits(model, tokenizer)
    assert not torch.equal(base, edited)
    assert torch.equal(base, after)


def test_lambda_zero_exact_noop():
    model, tokenizer = load_model()
    direction = torch.randn(3072)
    base = _logits(model, tokenizer)
    with project_residual_writes(model, direction, [8], "both", 0.0):
        edited = _logits(model, tokenizer)
    assert torch.equal(base, edited)


def test_materialized_edit_restores_weights_bit_exact():
    model, _ = load_model()
    direction = torch.randn(3072)
    block = get_decoder_layers(model)[8]
    before = block.mlp.down_proj.weight.detach().cpu().clone()
    with orthogonalized_weights(model, direction, [8], "mlp", 1.0):
        assert not torch.equal(before, block.mlp.down_proj.weight.cpu())
    assert torch.equal(before, block.mlp.down_proj.weight.cpu())



def test_replacement_hook_changes_output_without_mutating_weight():
    model, tokenizer = load_model()
    direction = torch.randn(3072)
    block = get_decoder_layers(model)[8]
    before = block.self_attn.o_proj.weight.detach().cpu().clone()
    base = _logits(model, tokenizer)
    with replace_residual_writes(
        model, direction, [8], "attn", 1.0, norm_preserve=True
    ):
        edited = _logits(model, tokenizer)
    assert not torch.equal(base, edited)
    assert torch.equal(before, block.self_attn.o_proj.weight.cpu())
    assert torch.equal(base, _logits(model, tokenizer))


def test_embedding_projection_changes_output_and_restores():
    model, tokenizer = load_model()
    direction = torch.randn(3072)
    base = _logits(model, tokenizer)
    with project_embeddings(model, direction, 1.0):
        edited = _logits(model, tokenizer)
    assert not torch.equal(base, edited)
    assert torch.equal(base, _logits(model, tokenizer))
