"""M2 tests: steering context manager behavior."""

import torch

from ttsafety.generate import generate_texts
from ttsafety.models import load_model
from ttsafety.steer import steer

PROMPT = "What is the capital of France?"


def _rand_vec(seed=0):
    return torch.randn(3072, generator=torch.Generator().manual_seed(seed))


def test_alpha_zero_is_exact_noop():
    model, tokenizer = load_model()
    base = generate_texts(model, tokenizer, [PROMPT], max_new_tokens=32, batch_size=1)
    with steer(model, _rand_vec(), layer=14, alpha=0.0):
        out = generate_texts(model, tokenizer, [PROMPT], max_new_tokens=32, batch_size=1)
    assert out == base


@torch.no_grad()
def _logits(model, tokenizer):
    from ttsafety.models import chat_wrap

    enc = tokenizer(
        chat_wrap(tokenizer, PROMPT), return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    return model(**enc).logits


def test_nonzero_alpha_changes_output():
    model, tokenizer = load_model()
    base = _logits(model, tokenizer)
    with steer(model, _rand_vec(), layer=14, alpha=5.0):
        out = _logits(model, tokenizer)
    assert not torch.equal(out, base)
    assert (out - base).abs().max() > 1e-3


def test_hooks_removed_after_exit():
    model, tokenizer = load_model()
    base = generate_texts(model, tokenizer, [PROMPT], max_new_tokens=32, batch_size=1)
    with steer(model, {10: _rand_vec(1), 18: _rand_vec(2)}, alpha=5.0):
        generate_texts(model, tokenizer, [PROMPT], max_new_tokens=32, batch_size=1)
    after = generate_texts(model, tokenizer, [PROMPT], max_new_tokens=32, batch_size=1)
    assert after == base
