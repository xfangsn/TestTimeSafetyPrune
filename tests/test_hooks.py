"""M2 tests: residual-stream capture on the real model."""

import torch

from ttsafety.hooks import capture_last_token, get_decoder_layers
from ttsafety.models import chat_wrap, load_model

N_LAYERS = 28
HIDDEN = 3072


def test_get_decoder_layers():
    model, _ = load_model()
    blocks = get_decoder_layers(model)
    assert len(blocks) == N_LAYERS
    assert model.config.hidden_size == HIDDEN


def test_capture_last_token_shapes():
    model, tokenizer = load_model()
    prompts = [
        chat_wrap(tokenizer, s)
        for s in ["What is 2+2?", "Name a primary color.", "Say hi."]
    ]
    acts = capture_last_token(model, tokenizer, prompts, batch_size=3)
    assert set(acts.keys()) == set(range(N_LAYERS))
    for layer, t in acts.items():
        assert t.shape == (len(prompts), HIDDEN), f"layer {layer}"
        assert t.dtype == torch.float32
        assert torch.isfinite(t).all()


def test_capture_layer_subset():
    model, tokenizer = load_model()
    prompts = [chat_wrap(tokenizer, "Hello there.")]
    acts = capture_last_token(model, tokenizer, prompts, layers=[5, 14], batch_size=1)
    assert set(acts.keys()) == {5, 14}
    assert acts[14].shape == (1, HIDDEN)


def test_capture_position_matches_single_forward():
    """Batched (right-padded) capture must match unpadded single-sequence capture."""
    model, tokenizer = load_model()
    short = chat_wrap(tokenizer, "Hi.")
    long = chat_wrap(
        tokenizer,
        "Explain in detail how photosynthesis converts sunlight into chemical energy.",
    )
    batched = capture_last_token(model, tokenizer, [short, long], batch_size=2)
    solo_short = capture_last_token(model, tokenizer, [short], batch_size=1)
    solo_long = capture_last_token(model, tokenizer, [long], batch_size=1)
    # bf16 batched kernels are not bit-identical to single-sequence ones;
    # allow 5% relative error (a wrong capture position would give O(1) error)
    for layer in (0, 14, N_LAYERS - 1):
        for b_row, solo in ((batched[layer][0], solo_short[layer][0]),
                            (batched[layer][1], solo_long[layer][0])):
            rel = (b_row - solo).norm() / solo.norm()
            assert rel < 0.05, f"layer {layer}: relative diff {rel:.4f}"
    # different lengths must not collapse onto the same padded position
    assert not torch.allclose(batched[N_LAYERS - 1][0], batched[N_LAYERS - 1][1])
