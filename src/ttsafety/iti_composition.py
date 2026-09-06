"""Reversible BLADE + legacy ITI composition, matching the existing OOD baseline.

The legacy generation policy skips the whole prefill (including the first answer
logits). Its all-position PPL is a separate stress metric, not decode-policy PPL.
"""
from contextlib import contextmanager
import hashlib

import torch

from .hooks import get_decoder_layers
from .weight_edit import iter_residual_writers
from .weight_prune import _resolve_modules, rank_weight_indices, selection_from_ranking


def qwen_no_thinking(tokenizer, prompt):
    # No silent fallback: provenance must actually enforce the requested mode.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def tensor_dict_hash(tensors):
    h = hashlib.sha256()
    for key in sorted(tensors, key=str):
        t = tensors[key].detach().cpu().contiguous()
        h.update(str(key).encode())
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def strict_selection(scores, rho):
    # Keep the existing OOD BLADE ranking's max_fraction=.05 and cap=.10.
    selection = selection_from_ranking(rank_weight_indices(scores, max(.05, rho)), rho)
    requested = max(1, round(rho * sum(x.numel() for x in scores.values())))
    if sum(x.numel() for x in selection.values()) != requested:
        raise ValueError("BLADE mask has fewer elements than requested")
    for name, idx in selection.items():
        chosen = scores[name].flatten()[idx]
        if not (torch.isfinite(chosen) & (chosen > 0)).all():
            raise ValueError(f"Infeasible BLADE rho={rho}: nonpositive/nonfinite selection in {name}")
    return selection


@contextmanager
def scaled_selection(model, selection, factor):
    modules = _resolve_modules(model, list(selection))
    backups = {}
    try:
        with torch.no_grad():
            for name, idx in selection.items():
                weight = modules[name].weight
                ids = idx.to(weight.device)
                backups[name] = (ids, weight.view(-1)[ids].clone())
                weight.view(-1)[ids] = backups[name][1] * factor
        yield
    finally:
        with torch.no_grad():
            for name, (idx, original) in backups.items():
                modules[name].weight.view(-1)[idx] = original
                if not torch.equal(modules[name].weight.view(-1)[idx], original):
                    raise RuntimeError(f"Weight restoration failed: {name}")


@contextmanager
def iti_hook(model, add, *, policy="legacy_decode_only"):
    if policy not in ("legacy_decode_only", "all_positions"):
        raise ValueError(policy)
    handles = []

    def make(vec):
        def hook(_module, args):
            x = args[0]
            if policy == "legacy_decode_only" and x.shape[1] > 1:
                return None
            return (x + vec.to(device=x.device, dtype=x.dtype),) + tuple(args[1:])
        return hook

    try:
        for layer, vec in add.items():
            handles.append(get_decoder_layers(model)[layer].self_attn.o_proj.register_forward_pre_hook(make(vec)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def collect_writer_means(model, tok, prompts, layers, batch_size=8):
    writers = dict(iter_residual_writers(model, layers, "both"))
    sums = {k: torch.zeros(m.in_features, device=model.device) for k, m in writers.items()}
    state = {}

    def make(name):
        def hook(_module, args):
            x = args[0].float()
            state["batch"][name] = x[torch.arange(x.shape[0], device=x.device), state["last"]].sum(0)
        return hook

    handles = [m.register_forward_pre_hook(make(n)) for n, m in writers.items()]
    try:
        for start in range(0, len(prompts), batch_size):
            texts = [qwen_no_thinking(tok, p) for p in prompts[start:start + batch_size]]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
                      add_special_tokens=False).to(model.device)
            state.update(last=enc["attention_mask"].sum(1) - 1, batch={})
            model(**enc, use_cache=False)
            for name in sums:
                sums[name] += state["batch"][name]
    finally:
        for handle in handles:
            handle.remove()
    return {k: (x / len(prompts)).cpu() for k, x in sums.items()}


@torch.no_grad()
def collect_head_acts(model, tok, prompts, batch_size=8):
    blocks = get_decoder_layers(model)
    nh = model.config.num_attention_heads
    hd = getattr(model.config, "head_dim", model.config.hidden_size // nh)
    acc = {i: [] for i in range(len(blocks))}
    state = {}

    def make(i):
        def hook(_module, args):
            x = args[0].float()
            last = x[torch.arange(x.shape[0], device=x.device), state["last"]]
            acc[i].append(last.reshape(x.shape[0], nh, hd).cpu())
        return hook

    handles = [b.self_attn.o_proj.register_forward_pre_hook(make(i)) for i, b in enumerate(blocks)]
    try:
        for start in range(0, len(prompts), batch_size):
            texts = [qwen_no_thinking(tok, p) for p in prompts[start:start + batch_size]]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
                      add_special_tokens=False).to(model.device)
            state["last"] = enc["attention_mask"].sum(1) - 1
            model(**enc, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {i: torch.cat(parts) for i, parts in acc.items()}


def fit_legacy_iti(acts, labels, k=48, fixed_heads=None):
    labels = torch.tensor(labels)
    nh, hd = next(iter(acts.values())).shape[1:]
    # Match old class means, whose batch sum was computed on GPU before CPU accumulation.
    def class_mean(x):
        total = torch.zeros_like(x[0])
        for i in range(0, len(x), 8):
            total += x[i:i + 8].sum(0)
        return total / len(x)
    diffs = {}
    for i, x in acts.items():
        diff = class_mean(x[labels == 1]) - class_mean(x[labels == 0])
        diffs.update({(i, h): diff[h] for h in range(nh)})
    ranked = sorted(diffs, key=lambda key: -diffs[key].norm().item())[:k]
    heads = [tuple(x) for x in fixed_heads] if fixed_heads is not None else ranked
    if len(set(heads)) != k:
        raise ValueError("Expected exactly K distinct ITI heads")
    dirs, sigmas = {}, {}
    add = {i: torch.zeros(nh * hd) for i in acts}
    for key in heads:
        i, h = key
        norm = diffs[key].norm()
        if norm < 1e-6 or not torch.isfinite(norm):
            raise ValueError(f"Degenerate ITI direction {key}")
        dirs[key] = diffs[key] / norm
        # Historical sigma uses population std on uncertain+certain in this order.
        x = torch.cat([acts[i][labels == 1, h], acts[i][labels == 0, h]])
        projected = x @ dirs[key]
        s1 = s2 = 0.0
        for j in range(0, len(projected), 8):
            v = projected[j:j + 8]
            s1 += float(v.sum()); s2 += float((v * v).sum())
        sigmas[key] = max(s2 / len(x) - (s1 / len(x)) ** 2, 1e-8) ** 0.5
        add[i].reshape(nh, hd)[h] = sigmas[key] * dirs[key]
    return {"heads": heads, "directions": dirs, "sigmas": sigmas, "add_unit": add,
            "recomputed_ranking": ranked, "k": k, "nh": nh, "hd": hd,
            "historical_head_overlap": len(set(heads) & set(ranked)),
            "add_hash": tensor_dict_hash(add)}


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens=128):
    previous = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok([qwen_no_thinking(tok, p) for p in prompts], padding=True,
                  return_tensors="pt", add_special_tokens=False).to(model.device)
        # Legacy policy is intentionally defined by length>1; the selected OOD prompts
        # must all have multi-token prefills so it cannot steer their first answer logits.
        assert (enc["attention_mask"].sum(1) > 1).all()
        output = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tok.pad_token_id, use_cache=True)
        ids = output[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        rows = []
        for text, seq in zip(texts, ids.tolist()):
            end = next((i + 1 for i, t in enumerate(seq) if t in eos), len(seq))
            tokens = seq[:end]
            repeated = len(tokens) >= 16 and len(set(tokens[-32:])) <= 3
            rows.append({"generation": text, "n_tokens": len(tokens),
                         "truncated": len(tokens) >= max_new_tokens and not any(t in eos for t in tokens),
                         "degenerate_tail": repeated, "empty": not text.strip()})
        return rows
    finally:
        tok.padding_side = previous
