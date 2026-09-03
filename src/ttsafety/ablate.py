"""Neuron ablation: variant-robust top-k selection + zeroing hooks."""

from contextlib import contextmanager

import torch

from .hooks import get_decoder_layers

N_LAYERS = 28
N_NEURONS = 8192


def _flat_to_selection(flat_idx: torch.Tensor) -> dict[int, torch.Tensor]:
    """Global flat indices (layer * 8192 + neuron) -> {layer: neuron indices}."""
    sel: dict[int, list[int]] = {}
    for i in flat_idx.tolist():
        sel.setdefault(i // N_NEURONS, []).append(i % N_NEURONS)
    return {l: torch.tensor(sorted(v), dtype=torch.long) for l, v in sel.items()}


def top_k_selection(imp: dict[str, torch.Tensor], k: int,
                    rule: str = "rankagg") -> dict[int, torch.Tensor]:
    """Global top-k neurons by importance.

    rule="rankagg": average per-variant rank over softmax/l1/z (lower = more
    important). rule="softmax": softmax variant only.
    """
    if rule == "rankagg":
        ranks = []
        for v in ("softmax", "l1", "z"):
            m = imp[v].flatten()
            r = torch.empty(m.numel())
            r[m.argsort(descending=True)] = torch.arange(m.numel()).float()
            ranks.append(r)
        order = torch.stack(ranks).mean(dim=0).argsort()  # best first
    elif rule == "softmax":
        order = imp["softmax"].flatten().argsort(descending=True)
    else:
        raise ValueError(f"unknown rule: {rule!r}")
    return _flat_to_selection(order[:k])


def random_selection(k: int, seed: int) -> dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(N_LAYERS * N_NEURONS, generator=g)[:k]
    return _flat_to_selection(flat)


@contextmanager
def ablate_neurons(model, selection: dict[int, torch.Tensor]):
    """Zero the selected MLP intermediate (post-SwiGLU) neurons at all token
    positions via a forward_pre_hook on mlp.down_proj. Empty selection = no-op.
    """
    blocks = get_decoder_layers(model)
    handles = []
    for layer, idx in selection.items():
        idx = torch.as_tensor(idx, dtype=torch.long)
        if idx.numel() == 0:
            continue

        def make_pre(idx):
            def hook(module, args):
                h = args[0]
                h[..., idx.to(h.device)] = 0
                return None

            return hook

        handles.append(
            blocks[layer].mlp.down_proj.register_forward_pre_hook(make_pre(idx)))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()
