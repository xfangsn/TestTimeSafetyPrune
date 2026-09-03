"""Faithful activation-steering baselines for STRENGTHENING refusal.

Methods and their residual sites (residual-site correctness matters, see reviews):
  - CAA   (Rimsky et al., 2312.06681): span-mean(refusal)-span-mean(compliance) from
           matched caa_pairs, read at resid_post; added at resid_post, assistant-span
           positions (paired-response adaptation of CAA).
  - Arditi (2406.11717): mean(harmful)-mean(harmless) at the last post-instruction token,
           read at resid_post; RAW vector added at resid_post (same site), all positions.
  - ActAdd (Turner 2308.10248): single natural-language contrast PAIR, positionwise
           resid_pre difference (right-padded, masked to real-token overlap), added at
           resid_pre of the target's first content tokens (prompt-time only).
  - ITI   (Li 2306.03341): per-head mass-mean shift theta_h (unit) scaled by sigma_h at
           the pre-o_proj head slice; heads selected by out-of-fold logistic-probe balanced
           accuracy (instruction-grouped CV); theta_h/sigma_h refit on all pairs after ranking.

Causal-position rule: the LAST prompt/prefill position is always steered so the first free
token is affected. Prefill forwards have T>1; cached-decode forwards have T==1.

The driver sets STATE before each forward-group:
  STATE["left_pad"] : LongTensor [B] of left-pad counts (generation uses left padding), or None.
"""
from contextlib import contextmanager

import torch

from .hooks import get_decoder_layers, capture_span_mean, capture_last_token
from .models import chat_wrap

STATE: dict = {"left_pad": None}


# ============================ direction / probe extraction ============================

@torch.no_grad()
def caa_direction(model, tok, pairs, wrap=chat_wrap, batch_size: int = 16):
    """CAA span-mean direction: mean_examples[ mean_tokens(refusal) - mean_tokens(compliance) ].
    Primary recipe. Returns {layer: (H,) fp32 CPU}."""
    prompts = [wrap(tok, p["instruction"]) for p in pairs]
    ref = capture_span_mean(model, tok, prompts, [p["refusal"] for p in pairs], batch_size=batch_size)
    com = capture_span_mean(model, tok, prompts, [p["compliance"] for p in pairs], batch_size=batch_size)
    return {l: (ref[l].mean(0) - com[l].mean(0)) for l in ref}


@torch.no_grad()
def _full_resid(model, tok, text: str, add_special_tokens: bool):
    """resid_pre(l) for every layer l for a single raw text = resid_post(l-1); layer 0 uses
    the embedding output. Returns (input_ids, {layer: (T, H) fp32 CPU})."""
    blocks = get_decoder_layers(model)
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
    store: dict[int, torch.Tensor] = {}

    def mk(idx):
        def hook(_m, _a, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            # resid_pre(idx+1) = resid_post(idx)
            store[idx + 1] = h[0].detach().to("cpu", torch.float32)
        return hook

    handles = [blocks[i].register_forward_hook(mk(i)) for i in range(len(blocks))]
    # capture embedding output as resid_pre(0)
    emb = model.get_input_embeddings()
    def emb_hook(_m, _a, out):
        store[0] = out[0].detach().to("cpu", torch.float32)
    handles.append(emb.register_forward_hook(emb_hook))
    try:
        model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return enc["input_ids"][0].cpu(), store


@torch.no_grad()
def actadd_direction(model, tok, p_plus: str, p_minus: str, add_special_tokens: bool = False):
    """ActAdd positionwise resid_pre difference for a single contrast pair.
    Right-pads the shorter side; the returned per-layer delta has length = min(len+, len-)
    real overlap (positions beyond the shorter side are dropped, not fabricated).
    Returns {layer: (span, H) fp32 CPU}, span = min token length of the two phrases."""
    ids_p, res_p = _full_resid(model, tok, p_plus, add_special_tokens)
    ids_m, res_m = _full_resid(model, tok, p_minus, add_special_tokens)
    span = min(ids_p.numel(), ids_m.numel())  # real-token overlap only (no pad deltas)
    return {l: (res_p[l][:span] - res_m[l][:span]) for l in res_p}


@torch.no_grad()
def _oproj_input_last(model, tok, prompts, responses, batch_size: int = 16,
                      eot: str = "<|eot_id|>"):
    """Per-head pre-o_proj activation at the LAST completion token, all layers.
    Returns {layer: (N, n_heads*head_dim) fp32 CPU}. Response span end computed per
    example from tokenization (common-prefix rule, same as capture_span_mean)."""
    blocks = get_decoder_layers(model)
    device = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full = [p + r + eot for p, r in zip(prompts, responses)]
    last_idx = []
    for p, f in zip(prompts, full):
        f_ids = tok(f, add_special_tokens=False)["input_ids"]
        last_idx.append(len(f_ids) - 1)  # last completion (before pad); right padding used
    acc = {i: [] for i in range(len(blocks))}
    state = {}

    def mk(idx):
        def hook(_m, args):
            x = args[0]  # (B, T, n_heads*head_dim) input to o_proj
            rows = torch.arange(x.shape[0], device=x.device)
            acc[idx].append(x[rows, state["last"]].detach().to("cpu", torch.float32))
        return hook

    handles = [blocks[i].self_attn.o_proj.register_forward_pre_hook(mk(i))
               for i in range(len(blocks))]
    try:
        for s in range(0, len(full), batch_size):
            batch = full[s:s + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, padding_side="right",
                      add_special_tokens=False).to(device)
            state["last"] = torch.tensor(last_idx[s:s + batch_size], device=device)
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {i: torch.cat(acc[i], 0) for i in acc}


def iti_fit(model, tok, pairs, wrap, n_heads: int, head_dim: int, n_folds: int = 4,
            seed_perm=None):
    """Fit ITI: rank all (layer,head) query slices by out-of-fold balanced accuracy of a
    per-head logistic probe (refusal vs compliance response), instruction-grouped CV.
    Returns a list of dicts sorted best-first:
      {layer, head, acc, theta (head_dim, unit), sigma (float)} with theta/sigma refit on ALL pairs.
    seed_perm: optional precomputed permutation of range(len(pairs)) for deterministic folds."""
    from sklearn.linear_model import LogisticRegression

    prompts = [wrap(tok, p["instruction"]) for p in pairs]
    refusal = [p["refusal"] for p in pairs]
    compliance = [p["compliance"] for p in pairs]
    act_ref = _oproj_input_last(model, tok, prompts, refusal)   # {l: (N, HD)}
    act_com = _oproj_input_last(model, tok, prompts, compliance)
    n = len(pairs)
    L = len(act_ref)
    perm = list(range(n)) if seed_perm is None else list(seed_perm)
    folds = [perm[i::n_folds] for i in range(n_folds)]  # instruction-grouped by index

    results = []
    for l in range(L):
        ref = act_ref[l].view(n, n_heads, head_dim)
        com = act_com[l].view(n, n_heads, head_dim)
        for hd in range(n_heads):
            X = torch.cat([ref[:, hd], com[:, hd]], 0).numpy()       # (2N, head_dim)
            y = torch.cat([torch.ones(n), torch.zeros(n)]).numpy()
            # out-of-fold balanced accuracy; example i's two versions share fold(i)
            correct_pos = correct_neg = 0
            for fold in folds:
                test = set(fold)
                tr = [i for i in range(n) if i not in test]
                te = list(fold)
                if not te or not tr:
                    continue
                idx_tr = tr + [n + i for i in tr]
                idx_te_pos = te
                idx_te_neg = [n + i for i in te]
                clf = LogisticRegression(max_iter=200, C=1.0)
                clf.fit(X[idx_tr], y[idx_tr])
                pp = clf.predict(X[idx_te_pos])
                pn = clf.predict(X[idx_te_neg])
                correct_pos += (pp == 1).sum()
                correct_neg += (pn == 0).sum()
            bal = 0.5 * (correct_pos / n + correct_neg / n)
            # mass-mean shift refit on ALL pairs
            theta = (ref[:, hd].mean(0) - com[:, hd].mean(0))
            theta = theta / theta.norm().clamp_min(1e-8)
            allact = torch.cat([ref[:, hd], com[:, hd]], 0)
            sigma = (allact @ theta).std().item()
            results.append({"layer": l, "head": hd, "acc": float(bal),
                            "theta": theta, "sigma": float(sigma)})
    results.sort(key=lambda r: r["acc"], reverse=True)
    return results


# ============================ interventions (context managers) ============================

def _add_positions(h, delta, positions, is_prefill, k):
    """In-place add a single vector `delta` (H,) broadcast over selected positions of h (B,T,H)."""
    d = delta.to(device=h.device, dtype=h.dtype)
    if positions == "all":
        h.add_(d)
    elif positions == "after_prefix":          # CAA ppl: token 0 is the prefix
        h[:, 1:, :].add_(d)
    elif positions == "assistant_suffix":      # CAA generation: prefill=last k, decode=all
        if is_prefill:
            if k > 0:                          # NB: h[:, -0:] would be ALL rows, not none
                h[:, -k:, :].add_(d)
        else:
            h.add_(d)
    else:
        raise ValueError(positions)


@contextmanager
def resid_add(model, layer: int, vec: torch.Tensor, coef: float, positions: str, k: int = 0):
    """Add coef*vec at resid_post(layer) with the given position rule (CAA / Arditi)."""
    if coef == 0.0:
        yield
        return
    delta = coef * vec

    def hook(_m, _a, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        is_prefill = h.shape[1] > 1
        _add_positions(h, delta, positions, is_prefill, k)
        return out

    handle = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def actadd_apply(model, layer: int, seq_delta: torch.Tensor, coef: float, offset: int = 0):
    """Add coef*seq_delta (span,H) at resid_pre(layer) = resid_post(layer-1), aligned at the
    first content token of each row, prompt-time (prefill) only. Uses STATE['left_pad']."""
    if coef == 0.0 or layer == 0:
        yield
        return
    span = seq_delta.shape[0]

    def hook(_m, _a, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        if h.shape[1] <= 1:            # decode step: ActAdd is prompt-time only
            return out
        d = (coef * seq_delta).to(device=h.device, dtype=h.dtype)   # (span, H)
        lp = STATE.get("left_pad")
        B, T, _ = h.shape
        if lp is None:
            lp = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            start = int(lp[b]) + offset
            end = min(start + span, T)
            n = end - start
            if n > 0:
                h[b, start:end, :].add_(d[:n])
        return out

    handle = get_decoder_layers(model)[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def iti_apply(model, heads: list[dict], alpha: float, head_dim: int):
    """Add alpha*sigma_h*theta_h to each selected head's pre-o_proj slice, all positions."""
    if alpha == 0.0 or not heads:
        yield
        return
    by_layer: dict[int, list[dict]] = {}
    for hrec in heads:
        by_layer.setdefault(hrec["layer"], []).append(hrec)
    handles = []
    blocks = get_decoder_layers(model)

    def mk(layer_heads):
        def hook(_m, args):
            x = args[0]                      # (B, T, n_heads*head_dim)
            for hrec in layer_heads:
                sl = slice(hrec["head"] * head_dim, (hrec["head"] + 1) * head_dim)
                d = (alpha * hrec["sigma"] * hrec["theta"]).to(device=x.device, dtype=x.dtype)
                x[:, :, sl].add_(d)
            return (x,) + args[1:] if len(args) > 1 else (x,)
        return hook

    for layer, lheads in by_layer.items():
        handles.append(blocks[layer].self_attn.o_proj.register_forward_pre_hook(mk(lheads)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
