"""Forward-only contrastive refusal-flow attribution for residual writers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .models import chat_wrap
from .weight_edit import iter_residual_writers


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def response_prediction_span(
    prompt_ids: list[int], full_ids: list[int]
) -> tuple[int, int]:
    """Positions whose hidden states predict the response (including EOT).

    If response targets begin at token ``m``, causal-LM hidden position ``m-1``
    predicts the first response token. The returned interval is a Python slice
    ``[start, end)`` over hidden states.
    """
    response_start = common_prefix_length(prompt_ids, full_ids)
    if response_start <= 0:
        raise ValueError("response has no usable causal prefix")
    start = response_start - 1
    end = len(full_ids) - 1
    if end <= start:
        raise ValueError("empty response-prediction span")
    return start, end


def _backbone(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_decoder", None)
    if not callable(getter):
        raise TypeError("model does not expose get_decoder(); cannot skip LM head")
    decoder = getter()
    if decoder is model:
        raise TypeError("get_decoder returned the full causal LM")
    return decoder


def assert_gradient_free(model: nn.Module) -> None:
    enabled = [name for name, value in model.named_parameters() if value.requires_grad]
    gradients = [name for name, value in model.named_parameters() if value.grad is not None]
    if enabled:
        raise AssertionError(f"parameters still require gradients: {enabled[:3]}")
    if gradients:
        raise AssertionError(f"parameter gradients were allocated: {gradients[:3]}")


def _merge_batch_moments(
    count: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    values: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Stable parallel-Welford merge for a [batch, features] tensor."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must be a non-empty [batch, features] tensor")
    values = values.float()
    batch_count = values.shape[0]
    batch_mean = values.mean(0)
    batch_m2 = (values - batch_mean).square().sum(0)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + delta.square() * (count * batch_count / total)
    return total, merged_mean, merged_m2


@torch.inference_mode()
def collect_paired_writer_moments(
    model: nn.Module,
    tokenizer,
    pairs: list[dict],
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
    batch_pairs: int = 4,
    progress_every: int = 20,
) -> dict[str, dict[str, torch.Tensor | int]]:
    """Collect R-C response-prediction input moments with forward hooks."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be positive")
    model.requires_grad_(False)
    assert_gradient_free(model)
    writers = dict(iter_residual_writers(model, layers, components))
    device = next(model.parameters()).device
    decoder = _backbone(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stats = {
        name: {
            "count": 0,
            "mean": torch.zeros(module.in_features, device=device, dtype=torch.float32),
            "m2": torch.zeros(module.in_features, device=device, dtype=torch.float32),
            "positive": torch.zeros(
                module.in_features, device=device, dtype=torch.int32
            ),
        }
        for name, module in writers.items()
    }
    state: dict = {}

    def make_hook(name):
        def hook(_module, args):
            values = args[0].float()
            means = torch.stack([
                values[row, start:end].mean(0)
                for row, (start, end) in enumerate(state["spans"])
            ])
            delta = means[0::2] - means[1::2]
            item = stats[name]
            count, mean, m2 = _merge_batch_moments(
                item["count"], item["mean"], item["m2"], delta
            )
            item["count"] = count
            item["mean"] = mean
            item["m2"] = m2
            item["positive"].add_((delta > 0).sum(0).to(torch.int32))

        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in writers.items()
    ]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for offset in range(0, len(pairs), batch_pairs):
            chunk = pairs[offset : offset + batch_pairs]
            texts: list[str] = []
            spans: list[tuple[int, int]] = []
            for pair in chunk:
                prompt = chat_wrap(tokenizer, pair["instruction"])
                prompt_ids = tokenizer(
                    prompt, add_special_tokens=False
                )["input_ids"]
                for key in ("refusal", "compliance"):
                    full = prompt + pair[key] + "<|eot_id|>"
                    full_ids = tokenizer(
                        full, add_special_tokens=False
                    )["input_ids"]
                    texts.append(full)
                    spans.append(response_prediction_span(prompt_ids, full_ids))
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            state["spans"] = spans
            decoder(**encoded, use_cache=False, return_dict=True)
            done = min(offset + batch_pairs, len(pairs))
            if progress_every and (done % progress_every == 0 or done == len(pairs)):
                print(f"CRFP paired moments {done}/{len(pairs)}", flush=True)
    finally:
        tokenizer.padding_side = old_side
        for handle in handles:
            handle.remove()

    output = {}
    for name, item in stats.items():
        count = int(item["count"])
        if count != len(pairs):
            raise RuntimeError(f"{name}: collected {count}, expected {len(pairs)}")
        variance = item["m2"] / max(count - 1, 1)
        output[name] = {
            "count": count,
            "mean": item["mean"].cpu(),
            "variance": variance.cpu(),
            "positive_fraction": (item["positive"].float() / count).cpu(),
        }
    assert_gradient_free(model)
    return output


@torch.inference_mode()
def collect_harmless_writer_moments(
    model: nn.Module,
    tokenizer,
    instructions: list[str],
    layers: Iterable[int],
    components: str | Iterable[str] = "both",
    batch_size: int = 8,
    progress_every: int = 40,
) -> dict[str, torch.Tensor]:
    """Collect E[x^2] over valid harmless prompt tokens, forward-only."""
    if not instructions:
        raise ValueError("instructions must be non-empty")
    model.requires_grad_(False)
    assert_gradient_free(model)
    writers = dict(iter_residual_writers(model, layers, components))
    device = next(model.parameters()).device
    decoder = _backbone(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sums = {
        name: torch.zeros(module.in_features, device=device, dtype=torch.float32)
        for name, module in writers.items()
    }
    state: dict = {}

    def make_hook(name):
        def hook(_module, args):
            selected = args[0].float()[state["mask"]]
            sums[name].add_(selected.square().sum(0))

        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in writers.items()
    ]
    total_tokens = 0
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for offset in range(0, len(instructions), batch_size):
            texts = [
                chat_wrap(tokenizer, value)
                for value in instructions[offset : offset + batch_size]
            ]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            state["mask"] = encoded["attention_mask"].bool()
            decoder(**encoded, use_cache=False, return_dict=True)
            total_tokens += int(state["mask"].sum())
            done = min(offset + batch_size, len(instructions))
            if progress_every and (done % progress_every == 0 or done == len(instructions)):
                print(f"CRFP harmless moments {done}/{len(instructions)}", flush=True)
    finally:
        tokenizer.padding_side = old_side
        for handle in handles:
            handle.remove()
    if total_tokens == 0:
        raise RuntimeError("no harmless tokens collected")
    assert_gradient_free(model)
    return {name: value.cpu() / total_tokens for name, value in sums.items()}


@torch.no_grad()
def crfp_matrix_score(
    weight: torch.Tensor,
    direction: torch.Tensor,
    paired: dict[str, torch.Tensor | int],
    harmless_second_moment: torch.Tensor,
    *,
    beta: float = 1.0,
    eligibility_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict]:
    """Compute the primary LCB + tempered-Wanda score for one matrix."""
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    if not 0 < eligibility_fraction <= 1:
        raise ValueError("eligibility_fraction must be in (0, 1]")
    count = int(paired["count"])
    if count <= 0:
        raise ValueError("paired count must be positive")
    device = weight.device
    mean = paired["mean"].to(device=device, dtype=torch.float32)
    variance = paired["variance"].to(device=device, dtype=torch.float32)
    harmless = harmless_second_moment.to(device=device, dtype=torch.float32)
    if mean.numel() != weight.shape[1] or harmless.numel() != weight.shape[1]:
        raise ValueError("input statistics do not match matrix input dimension")
    unit = direction.to(device=device, dtype=torch.float32)
    unit = unit / unit.norm().clamp_min(1e-12)
    if unit.numel() != weight.shape[0]:
        raise ValueError("direction does not match matrix output dimension")

    coefficient = unit[:, None] * weight.detach().float()
    standard_error = (variance.clamp_min(0) / count).sqrt()
    benefit = coefficient * mean[None, :]
    benefit.sub_(coefficient.abs() * (beta * standard_error[None, :]))
    benefit.clamp_min_(0)
    positive_count = int((benefit > 0).sum())

    harmless_cost = weight.detach().float().abs()
    harmless_cost.mul_(harmless.clamp_min(0).sqrt()[None, :])
    tau = float(harmless_cost.median())
    tau = max(tau, 1e-12)
    score = benefit / (harmless_cost + tau).sqrt()

    if positive_count:
        eligible_k = max(1, int(eligibility_fraction * positive_count))
        eligible_k = min(eligible_k, score.numel())
        threshold = torch.topk(
            benefit.flatten(), eligible_k, largest=True, sorted=False
        ).values.min()
        score.masked_fill_(benefit < threshold, 0)
    else:
        eligible_k = 0
        score.zero_()
    diagnostics = {
        "shape": list(weight.shape),
        "positive_benefit_fraction": positive_count / weight.numel(),
        "eligible_count": eligible_k,
        "eligible_fraction_of_matrix": eligible_k / weight.numel(),
        "tau": tau,
        "score_max": float(score.max()),
        "score_mean": float(score.mean()),
    }
    return score.cpu().to(torch.float16), diagnostics

