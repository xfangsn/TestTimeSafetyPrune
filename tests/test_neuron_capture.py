"""N0 tests: neuron capture facility (shapes, determinism, steering locality)."""

import torch

from ttsafety.hooks import capture_neurons, get_decoder_layers
from ttsafety.models import chat_wrap, load_model
from ttsafety.steer import steer

N_LAYERS = 28
HIDDEN = 3072
INTERMEDIATE = 8192

PROMPTS = ["What is 2+2?", "Name a primary color."]


def _encode(model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [chat_wrap(tokenizer, p) for p in PROMPTS]
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    padding_side="right", add_special_tokens=False)
    return enc


def test_capture_shapes_and_dtype():
    model, tokenizer = load_model()
    enc = _encode(model, tokenizer)
    layers = [7, 8, 14]
    caps = capture_neurons(model, enc["input_ids"], enc["attention_mask"], layers)
    b, t = enc["input_ids"].shape
    assert set(caps["mlp"]) == set(layers) == set(caps["resid"])
    for l in layers:
        assert caps["mlp"][l].shape == (b, t, INTERMEDIATE)
        assert caps["resid"][l].shape == (b, t, HIDDEN)
        assert caps["mlp"][l].dtype == torch.float32
        assert torch.isfinite(caps["mlp"][l]).all()


def test_down_proj_input_is_post_swiglu():
    """Verify the captured 'neuron' tensor equals act(gate(x)) * up(x)."""
    model, tokenizer = load_model()
    enc = _encode(model, tokenizer)
    blocks = get_decoder_layers(model)
    layer = 8
    mlp_in = {}

    pre = blocks[layer].mlp.register_forward_pre_hook(
        lambda m, args: mlp_in.setdefault("x", args[0].detach()))
    caps = capture_neurons(model, enc["input_ids"], enc["attention_mask"], [layer])
    pre.remove()

    mlp = blocks[layer].mlp
    manual = mlp.act_fn(mlp.gate_proj(mlp_in["x"])) * mlp.up_proj(mlp_in["x"])
    assert torch.equal(caps["mlp"][layer].to(manual.dtype), manual.cpu())


def test_two_forwards_bit_identical():
    model, tokenizer = load_model()
    enc = _encode(model, tokenizer)
    layers = [7, 8, 14]
    a = capture_neurons(model, enc["input_ids"], enc["attention_mask"], layers)
    b = capture_neurons(model, enc["input_ids"], enc["attention_mask"], layers)
    for kind in ("mlp", "resid"):
        for l in layers:
            assert torch.equal(a[kind][l], b[kind][l]), f"{kind} L{l}"


def test_steering_locality():
    """Steering at L8 must change L8+ captures and leave L0-L7 untouched."""
    model, tokenizer = load_model()
    enc = _encode(model, tokenizer)
    directions = torch.load(
        "data/directions/refusal_llama32_3b_instruct.pt", weights_only=True)
    layers = list(range(0, 15))
    base = capture_neurons(model, enc["input_ids"], enc["attention_mask"], layers)
    with steer(model, directions[8], layer=8, alpha=-2.0):
        steered = capture_neurons(model, enc["input_ids"], enc["attention_mask"], layers)
    for l in range(0, 8):  # strictly upstream of the steering layer
        assert torch.equal(base["mlp"][l], steered["mlp"][l]), f"mlp L{l}"
        assert torch.equal(base["resid"][l], steered["resid"][l]), f"resid L{l}"
    # resid of L8 is the block output -> steered; its mlp input is upstream -> same
    assert not torch.equal(base["resid"][8], steered["resid"][8])
    assert torch.equal(base["mlp"][8], steered["mlp"][8])
    for l in range(9, 15):  # downstream: both views change
        assert not torch.equal(base["mlp"][l], steered["mlp"][l]), f"mlp L{l}"
        assert not torch.equal(base["resid"][l], steered["resid"][l]), f"resid L{l}"
