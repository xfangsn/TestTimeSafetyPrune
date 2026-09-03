"""Activation steering via forward hooks on decoder blocks."""

from contextlib import contextmanager

import torch

from .hooks import get_decoder_layers


@contextmanager
def steer(model, vectors, layer: int | None = None, alpha: float = 1.0, mode: str = "raw"):
    """Steer the residual stream: h <- h + alpha * v_hat at ALL token positions.

    vectors: {layer_idx: tensor} for multi-layer, or a single tensor with
    `layer` given. v_hat is the unit vector for mode="raw"; mode="relative"
    additionally scales it by the batch's mean token norm at that layer.
    alpha=0 is an exact no-op (no hooks are registered). All hooks are
    removed on exit.
    """
    if isinstance(vectors, dict):
        vec_map = {int(k): v for k, v in vectors.items()}
    else:
        if layer is None:
            raise ValueError("layer is required when vectors is a single tensor")
        vec_map = {int(layer): vectors}
    if alpha == 0 or not vec_map:
        yield
        return
    if mode not in ("raw", "relative"):
        raise ValueError(f"unknown mode: {mode!r}")

    blocks = get_decoder_layers(model)

    def make_hook(vec: torch.Tensor):
        unit = vec / vec.norm()

        def hook(module, args, output):
            is_seq = isinstance(output, (tuple, list))
            h = output[0] if is_seq else output
            v = unit.to(h.device, h.dtype)
            if mode == "relative":
                v = v * h.norm(dim=-1).mean()
            h = h + alpha * v
            return (h, *output[1:]) if is_seq else h

        return hook

    handles = [
        blocks[i].register_forward_hook(make_hook(vec_map[i])) for i in sorted(vec_map)
    ]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()
