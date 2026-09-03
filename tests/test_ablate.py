"""N3 tests: neuron ablation hook behavior."""

import torch

from ttsafety.ablate import ablate_neurons, random_selection, top_k_selection
from ttsafety.models import chat_wrap, load_model

PROMPT = "What is the capital of France?"


@torch.no_grad()
def _logits(model, tokenizer):
    enc = tokenizer(chat_wrap(tokenizer, PROMPT), return_tensors="pt",
                    add_special_tokens=False).to(model.device)
    return model(**enc).logits


def test_empty_selection_is_noop():
    model, tokenizer = load_model()
    base = _logits(model, tokenizer)
    with ablate_neurons(model, {}):
        out = _logits(model, tokenizer)
    assert torch.equal(out, base)


def test_ablation_changes_output_and_restores():
    model, tokenizer = load_model()
    base = _logits(model, tokenizer)
    sel = random_selection(64, seed=0)
    with ablate_neurons(model, sel):
        out = _logits(model, tokenizer)
    assert not torch.equal(out, base)
    after = _logits(model, tokenizer)
    assert torch.equal(after, base)  # hooks removed


def test_top_k_selection_structure():
    imp = {v: torch.rand(28, 8192) for v in ("softmax", "l1", "z")}
    sel = top_k_selection(imp, 256, rule="rankagg")
    total = sum(len(v) for v in sel.values())
    assert total == 256
    for layer, idx in sel.items():
        assert 0 <= layer < 28
        assert idx.min() >= 0 and idx.max() < 8192
        assert len(idx.unique()) == len(idx)
    # rank-aggregate should pick the globally agreed top neuron
    imp2 = {v: torch.zeros(28, 8192) for v in ("softmax", "l1", "z")}
    for v in imp2:
        imp2[v][9, 42] = 10.0
    sel2 = top_k_selection(imp2, 1, rule="rankagg")
    assert sel2 == {9: torch.tensor([42])}
