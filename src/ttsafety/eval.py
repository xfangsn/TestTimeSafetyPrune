"""Side-effect metrics for language-model interventions."""

import math
import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .models import chat_wrap


def _offline_text(env_var: str):
    """Return the contents of the file named by `env_var`, if set and present.
    Lets air-gapped compute nodes (e.g. Hazel) read calibration text that was
    materialized on an internet-connected login node. Returns None otherwise."""
    p = os.environ.get(env_var)
    if p and Path(p).is_file():
        return Path(p).read_text(encoding="utf-8")
    return None


def load_wikitext_text(config: str = "wikitext-2-raw-v1", split: str = "test") -> str:
    """WikiText test text. Prefers the offline file in $TTS_WIKI_FILE (for air-gapped
    nodes), else the local HF cache / download."""
    cached = _offline_text("TTS_WIKI_FILE")
    if cached is not None:
        return cached
    from datasets import load_dataset

    try:
        ds = load_dataset("Salesforce/wikitext", config, split=split)
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    lines = [t for t in ds["text"] if t.strip()]
    return "\n".join(lines)


def load_c4_text(n_docs: int = 80) -> str:
    """C4 (en) calibration text, streamed deterministically (first n_docs docs).
    Used to CALIBRATE the ELS perplexity budget, so reported WikiText ppl stays
    held-out (mirrors Wanda: calibrate on C4, evaluate on WikiText).
    Prefers the offline file in $TTS_C4_FILE (for air-gapped nodes)."""
    cached = _offline_text("TTS_C4_FILE")
    if cached is not None:
        return cached
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    out = []
    for i, r in enumerate(ds):
        if i >= n_docs:
            break
        out.append(r["text"])
    return "\n\n".join(out)


@torch.no_grad()
def teacher_forced_ppl(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 50_000,
    window: int = 1024,
    batch_size: int = 8,
) -> float:
    """Mean teacher-forced ppl over non-overlapping windows of the token stream."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0, :max_tokens]
    n_windows = ids.numel() // window
    windows = ids[: n_windows * window].view(n_windows, window)
    device = next(model.parameters()).device

    losses = []
    for start in range(0, n_windows, batch_size):
        batch = windows[start : start + batch_size].to(device)
        # equal token counts per window, so a plain mean over batch losses is exact
        losses.append(model(input_ids=batch, labels=batch).loss.item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def teacher_forced_nll(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 50_000,
    window: int = 1024,
    batch_size: int = 8,
) -> tuple[float, int]:
    """Mean teacher-forced NLL (nats / scored token) and the scored-token count.

    Non-overlapping windows of `window` tokens; each window scores `window-1`
    next-token targets (the first token has no scored predecessor). With N whole
    windows the scored-token count is N*(window-1), NOT N*window -- this is the
    honest denominator (e.g. 4 x 1023 = 4092, not 4096). Any active steering
    context applies during the forward, so ΔNLL = nll_edit - nll_base gives a
    baseline-independent capability cost; relative ppl change = exp(ΔNLL) - 1.
    """
    ids = tokenizer(text, return_tensors="pt").input_ids[0, :max_tokens]
    n_windows = ids.numel() // window
    windows = ids[: n_windows * window].view(n_windows, window)
    device = next(model.parameters()).device
    losses = []
    for start in range(0, n_windows, batch_size):
        batch = windows[start : start + batch_size].to(device)
        losses.append(model(input_ids=batch, labels=batch).loss.item())
    return sum(losses) / len(losses), n_windows * (window - 1)


@torch.no_grad()
def prompt_kl(
    model,
    tokenizer,
    prompts: list[str],
    edit_context: Callable = nullcontext,
    max_length: int = 128,
    batch_size: int = 2,
) -> float:
    """Exact mean KL(base || edit) over non-padding prompt-token predictions.

    Base and edited forwards are paired batch-by-batch, so this does not need a
    second model or a multi-gigabyte logits cache. edit_context must return a
    fresh context manager on each call.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_kl = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(prompts), batch_size):
            texts = [chat_wrap(tokenizer, x) for x in prompts[start : start + batch_size]]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(model.device)
            base_logits = model(**enc).logits.float()
            with edit_context():
                edit_logits = model(**enc).logits.float()
            base_logp = F.log_softmax(base_logits, dim=-1)
            edit_logp = F.log_softmax(edit_logits, dim=-1)
            token_kl = F.kl_div(
                edit_logp, base_logp, reduction="none", log_target=True
            ).sum(dim=-1)
            mask = enc["attention_mask"].bool()
            total_kl += token_kl.masked_select(mask).sum().item()
            total_tokens += mask.sum().item()
    finally:
        tokenizer.padding_side = old_side
    if total_tokens == 0:
        raise ValueError("no prompt tokens available for KL")
    return total_kl / total_tokens


def completion_agreement(tokenizer, baseline: list[str], edited: list[str]) -> dict:
    """Exact-string rate and mean position-wise token agreement."""
    if len(baseline) != len(edited) or not baseline:
        raise ValueError("completion lists must have the same non-zero length")
    exact = 0
    positional = []
    for left, right in zip(baseline, edited):
        exact += left == right
        a = tokenizer.encode(left, add_special_tokens=False)
        b = tokenizer.encode(right, add_special_tokens=False)
        denom = max(len(a), len(b), 1)
        positional.append(sum(x == y for x, y in zip(a, b)) / denom)
    return {
        "exact_match_rate": exact / len(baseline),
        "mean_token_agreement": sum(positional) / len(positional),
    }


def _has_repetition(tokens: list[int], n: int = 4) -> bool:
    if len(tokens) < 2 * n:
        return False
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    low_diversity_loop = (
        max(counts.values(), default=0) >= 10
        and len(set(tokens)) / len(tokens) < 0.35
    )
    consecutive_loop = any(
        tokens[i : i + width] == tokens[i + width : i + 2 * width]
        == tokens[i + 2 * width : i + 3 * width]
        for width in range(n, min(33, len(tokens) // 3 + 1))
        for i in range(len(tokens) - 3 * width + 1)
    )
    return low_diversity_loop or consecutive_loop


def completion_quality(tokenizer, completions: list[str]) -> dict:
    """Cheap deterministic flags for empty, repetitive, or corrupted outputs."""
    if not completions:
        raise ValueError("completions must be non-empty")
    flags = []
    lengths = []
    unique_ratios = []
    for text in completions:
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
        unique_ratios.append(len(set(ids)) / max(len(ids), 1))
        is_empty = not text.strip()
        has_bad_unicode = "\ufffd" in text or any(
            ord(ch) < 32 and ch not in "\n\r\t" for ch in text
        )
        is_repetitive = _has_repetition(ids)
        flags.append((is_empty, has_bad_unicode, is_repetitive))
    n = len(completions)
    return {
        "empty_rate": sum(x[0] for x in flags) / n,
        "garbled_rate": sum(x[1] for x in flags) / n,
        "repetition_rate": sum(x[2] for x in flags) / n,
        "adverse_rate": sum(any(x) for x in flags) / n,
        "mean_tokens": sum(lengths) / n,
        "mean_unique_token_ratio": sum(unique_ratios) / n,
    }
