"""Direction-aligned edits of Transformer residual-stream writer weights."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .hooks import get_decoder_layers

VALID_COMPONENTS = ("mlp", "attn")


def _component_set(components: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(components, str):
        components = (components,) if components != "both" else VALID_COMPONENTS
    out = tuple(dict.fromkeys(components))
    unknown = set(out) - set(VALID_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components: {sorted(unknown)}")
    return out


def _unit(direction: torch.Tensor, *, device=None, dtype=torch.float32) -> torch.Tensor:
    r = direction.detach().to(device=device, dtype=dtype)
    norm = r.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("direction must have a finite, non-zero norm")
    return r / norm


_ATTN_WRITER_PATHS = ("self_attn.o_proj", "attention.dense", "attn.c_proj",
                      "attn.out_proj", "self_attention.dense")
_MLP_WRITER_PATHS = ("mlp.down_proj", "mlp.dense_4h_to_h", "mlp.c_proj", "mlp.fc2")


def _find_writer(block, paths):
    """Return (dotted-path, module) for the first attribute path that resolves."""
    for path in paths:
        m = block
        ok = True
        for p in path.split("."):
            m = getattr(m, p, None)
            if m is None:
                ok = False
                break
        if ok:
            return path, m
    return None, None


def iter_residual_writers(
    model: nn.Module,
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
) -> Iterator[tuple[str, nn.Linear]]:
    """Yield named MLP/attention modules that write to the residual stream.

    Architecture-aware: resolves the attn/mlp output projections across Llama/
    Qwen/Gemma/Phi (self_attn.o_proj, mlp.down_proj) and GPT-NeoX/Pythia
    (attention.dense, mlp.dense_4h_to_h).
    """
    blocks = get_decoder_layers(model)
    comps = _component_set(components)
    for layer in sorted(set(int(x) for x in layers)):
        if not 0 <= layer < len(blocks):
            raise IndexError(f"layer {layer} outside [0, {len(blocks)})")
        block = blocks[layer]
        if "mlp" in comps:
            path, mod = _find_writer(block, _MLP_WRITER_PATHS)
            if mod is None:
                raise AttributeError(f"no MLP residual writer found in block {layer}")
            yield f"layers.{layer}.{path}", mod
        if "attn" in comps:
            path, mod = _find_writer(block, _ATTN_WRITER_PATHS)
            if mod is None:
                raise AttributeError(f"no attn residual writer found in block {layer}")
            yield f"layers.{layer}.{path}", mod


@contextmanager
def project_residual_writes(
    model: nn.Module,
    direction: torch.Tensor | dict[int, torch.Tensor],
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
    strength: float = 1.0,
):
    """Project selected module outputs away from a refusal direction.

    For y = W x, applies y' = y - strength * (y dot r_hat) r_hat.  A mapping
    direction[layer] enables the per-destination-layer robustness variant.
    The model parameters are never mutated.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    layer_list = sorted(set(int(x) for x in layers))
    if strength == 0.0 or not layer_list:
        yield
        return

    handles = []
    for name, module in iter_residual_writers(model, layer_list, components):
        layer = int(name.split(".")[1])
        raw = direction[layer] if isinstance(direction, dict) else direction
        unit = _unit(raw, device=module.weight.device, dtype=torch.float32)

        def make_hook(unit_direction):
            def hook(_module, _args, output):
                if not torch.is_tensor(output):
                    raise TypeError("residual writer output must be a tensor")
                y = output.float()
                projected = y - strength * (y @ unit_direction).unsqueeze(-1) * unit_direction
                return projected.to(output.dtype)

            return hook

        handles.append(module.register_forward_hook(make_hook(unit)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def replace_residual_writes(
    model: nn.Module,
    direction: torch.Tensor | dict[int, torch.Tensor],
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
    strength: float = 1.0,
    norm_preserve: bool = False,
):
    """Replace selected Linear outputs with outputs from materialized edited weights.

    This leaves parameters untouched and supports the norm-preserving W3
    variant, which cannot be represented from the original module output alone.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    handles = []
    edited_weights = []
    if strength == 0.0:
        yield
        return
    for name, module in iter_residual_writers(model, layers, components):
        layer = int(name.split(".")[1])
        raw = direction[layer] if isinstance(direction, dict) else direction
        edited = project_weight(
            module.weight, raw, strength=strength, norm_preserve=norm_preserve
        )
        edited_weights.append(edited)

        def make_hook(weight):
            def hook(_module, args, _output):
                return F.linear(args[0], weight, _module.bias)

            return hook

        handles.append(module.register_forward_hook(make_hook(edited)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def project_embeddings(
    model: nn.Module,
    direction: torch.Tensor,
    strength: float = 1.0,
):
    """Project input-embedding outputs away from a residual-stream direction."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if strength == 0.0:
        yield
        return
    embedding = model.get_input_embeddings()
    unit = _unit(direction, device=embedding.weight.device, dtype=torch.float32)

    def hook(_module, _args, output):
        value = output.float()
        projected = value - strength * (value @ unit).unsqueeze(-1) * unit
        return projected.to(output.dtype)

    handle = embedding.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def project_weight(
    weight: torch.Tensor,
    direction: torch.Tensor,
    strength: float = 1.0,
    norm_preserve: bool = False,
) -> torch.Tensor:
    """Return (I-strength*r*r^T)W, computed in fp32."""
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    r = _unit(direction, device=weight.device, dtype=torch.float32)
    w = weight.detach().float()
    edited = w - strength * r[:, None] * (r @ w)[None, :]
    if norm_preserve:
        old_norm = w.norm(dim=0)
        new_norm = edited.norm(dim=0)
        scale = torch.where(new_norm > 0, old_norm / new_norm, torch.ones_like(new_norm))
        edited = edited * scale[None, :]
    return edited.to(weight.dtype)


def materialize_orthogonalization(
    model: nn.Module,
    direction: torch.Tensor | dict[int, torch.Tensor],
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
    strength: float = 1.0,
    norm_preserve: bool = False,
) -> dict[str, torch.Tensor]:
    """Mutate selected weights and return exact CPU backups for restoration."""
    backups: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in iter_residual_writers(model, layers, components):
            layer = int(name.split(".")[1])
            raw = direction[layer] if isinstance(direction, dict) else direction
            backups[name] = module.weight.detach().cpu().clone()
            module.weight.copy_(
                project_weight(module.weight, raw, strength, norm_preserve)
            )
    return backups


def restore_weights(model: nn.Module, backups: dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, value in backups.items():
            module = modules.get(name)
            if module is None:
                # Hugging Face names include the model prefix. Match by suffix.
                matches = [m for n, m in modules.items() if n.endswith(name)]
                if len(matches) != 1:
                    raise KeyError(f"cannot uniquely resolve module {name!r}")
                module = matches[0]
            module.weight.copy_(value.to(module.weight.device, module.weight.dtype))


@contextmanager
def orthogonalized_weights(
    model: nn.Module,
    direction: torch.Tensor | dict[int, torch.Tensor],
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
    strength: float = 1.0,
    norm_preserve: bool = False,
):
    backups = materialize_orthogonalization(
        model, direction, layers, components, strength, norm_preserve
    )
    try:
        yield
    finally:
        restore_weights(model, backups)


@torch.no_grad()
def weight_delta_stats(
    model: nn.Module,
    direction: torch.Tensor | dict[int, torch.Tensor],
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
    strength: float = 1.0,
    norm_preserve: bool = False,
) -> dict:
    per_matrix = {}
    delta_sq = weight_sq = 0.0
    direction_energy_sq = 0.0
    n_params = 0
    for name, module in iter_residual_writers(model, layers, components):
        layer = int(name.split(".")[1])
        raw = direction[layer] if isinstance(direction, dict) else direction
        edited = project_weight(module.weight, raw, strength, norm_preserve)
        w = module.weight.detach().float()
        delta = edited.float() - w
        d2 = delta.square().sum().item()
        w2 = w.square().sum().item()
        delta_sq += d2
        r = _unit(raw, device=w.device, dtype=torch.float32)
        directional = (r @ w).square().sum().item()
        direction_energy_sq += directional
        weight_sq += w2
        n_params += w.numel()
        per_matrix[name] = {
            "n_params": w.numel(),
            "relative_delta_fro": (d2 / w2) ** 0.5 if w2 else 0.0,
            "direction_energy_fraction": directional / w2 if w2 else 0.0,
        }
    return {
        "n_matrices": len(per_matrix),
        "n_params": n_params,
        "relative_delta_fro": (delta_sq / weight_sq) ** 0.5 if weight_sq else 0.0,
        "direction_energy_fraction": (
            direction_energy_sq / weight_sq if weight_sq else 0.0
        ),
        "removed_direction_energy_fraction": (
            strength ** 2 * direction_energy_sq / weight_sq if weight_sq else 0.0
        ),
        "per_matrix": per_matrix,
    }


def random_unit_direction(hidden_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return _unit(torch.randn(hidden_size, generator=generator))
