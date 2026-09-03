"""Decoder-layer discovery and batched residual-stream capture."""

import torch
from torch import nn


class _StopForward(Exception):
    """Raised inside a hook to abort the forward pass early."""


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the nn.ModuleList of decoder blocks (``model.model.layers`` & friends).

    Picks the longest ModuleList found under the usual base modules.
    """
    bases = []
    getter = getattr(model, "get_decoder", None)
    if callable(getter):
        try:
            bases.append(getter())
        except Exception:  # pragma: no cover - exotic architectures
            pass
    for attr in ("model", "transformer", "gpt_neox"):
        sub = getattr(model, attr, None)
        if sub is not None:
            bases.append(sub)
    bases.append(model)

    best = None
    for base in bases:
        for m in base.modules():
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                if best is None or len(m) > len(best):
                    best = m
    if best is None:
        raise RuntimeError("no decoder ModuleList found in model")
    return best


@torch.no_grad()
def capture_last_token(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    layers: list[int] | None = None,
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """Residual stream at each prompt's last non-pad token, all layers in one pass.

    Prompts must already be chat-formatted. Right padding is used, so the
    captured position is attention_mask.sum(1) - 1 per row. Aborts the forward
    after the deepest requested layer (_StopForward). Returns
    {layer_idx: (N, hidden) fp32 CPU tensor}.
    """
    blocks = get_decoder_layers(model)
    if layers is None:
        layers = list(range(len(blocks)))
    layers = sorted(layers)
    deepest = layers[-1]
    device = next(model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    acc: dict[int, list[torch.Tensor]] = {i: [] for i in layers}
    state: dict = {}

    def make_hook(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            rows = torch.arange(h.shape[0], device=h.device)
            per_row = h[rows, state["last_idx"]]  # (B, hidden)
            acc[idx].append(per_row.detach().to("cpu", torch.float32))
            if idx == deepest:
                raise _StopForward

        return hook

    handles = [blocks[i].register_forward_hook(make_hook(i)) for i in layers]
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
            ).to(device)
            state["last_idx"] = enc["attention_mask"].sum(dim=1) - 1
            try:
                model(**enc, use_cache=False)
            except _StopForward:
                pass
    finally:
        for hd in handles:
            hd.remove()

    return {i: torch.cat(acc[i], dim=0) for i in layers}


@torch.no_grad()
def capture_span_mean(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    layers: list[int] | None = None,
    batch_size: int = 16,
    eot: str = "<|eot_id|>",
) -> dict[int, torch.Tensor]:
    """Mean residual-stream activation over the RESPONSE tokens per example.

    Full text = prompt + response + <|eot_id|>; the response span starts at
    the common-prefix length of tokenized prompt vs full text (robust to
    tokenization boundary effects). Returns {layer_idx: (N, hidden) fp32 CPU}.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must align")
    blocks = get_decoder_layers(model)
    if layers is None:
        layers = list(range(len(blocks)))
    layers = sorted(layers)
    deepest = layers[-1]
    device = next(model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_texts = [p + r + eot for p, r in zip(prompts, responses)]
    spans = []
    for prompt, full in zip(prompts, full_texts):
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        m = 0
        for a, b in zip(p_ids, f_ids):
            if a != b:
                break
            m += 1
        if m >= len(f_ids):
            raise ValueError("empty response span after tokenization")
        spans.append((m, len(f_ids)))

    acc: dict[int, list[torch.Tensor]] = {i: [] for i in layers}
    state: dict = {}

    def make_hook(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            rows = torch.stack(
                [h[b, s:e].mean(dim=0) for b, (s, e) in enumerate(state["spans"])]
            )
            acc[idx].append(rows.detach().to("cpu", torch.float32))
            if idx == deepest:
                raise _StopForward

        return hook

    handles = [blocks[i].register_forward_hook(make_hook(i)) for i in layers]
    try:
        for start in range(0, len(full_texts), batch_size):
            batch = full_texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
            ).to(device)
            state["spans"] = spans[start : start + batch_size]
            try:
                model(**enc, use_cache=False)
            except _StopForward:
                pass
    finally:
        for hd in handles:
            hd.remove()

    return {i: torch.cat(acc[i], dim=0) for i in layers}


@torch.no_grad()
def capture_neurons(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    layers: list[int] | None = None,
) -> dict[str, dict[int, torch.Tensor]]:
    """Capture MLP post-SwiGLU neuron activations and residual stream per layer.

    The "neuron" tensor is the input of ``mlp.down_proj`` (verified to equal
    ``act_fn(gate_proj(x)) * up_proj(x)`` on transformers 5.x LlamaMLP).
    "resid" is the decoder block output. Both are full [batch, seq, dim] fp32
    CPU tensors; the forward aborts after the deepest requested layer.
    Returns {"mlp": {layer: (B, T, 8192)}, "resid": {layer: (B, T, 3072)}}.
    """
    blocks = get_decoder_layers(model)
    if layers is None:
        layers = list(range(len(blocks)))
    layers = sorted(layers)
    deepest = layers[-1]
    device = next(model.parameters()).device

    out: dict[str, dict[int, torch.Tensor]] = {"mlp": {}, "resid": {}}

    def make_pre(idx):
        def hook(module, args):
            out["mlp"][idx] = args[0].detach().to("cpu", torch.float32)

        return hook

    def make_post(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            out["resid"][idx] = h.detach().to("cpu", torch.float32)
            if idx == deepest:
                raise _StopForward

        return hook

    handles = []
    for idx in layers:
        handles.append(
            blocks[idx].mlp.down_proj.register_forward_pre_hook(make_pre(idx)))
        handles.append(blocks[idx].register_forward_hook(make_post(idx)))
    try:
        try:
            model(
                input_ids=input_ids.to(device),
                attention_mask=(
                    attention_mask.to(device) if attention_mask is not None else None
                ),
                use_cache=False,
            )
        except _StopForward:
            pass
    finally:
        for hd in handles:
            hd.remove()
    return out
