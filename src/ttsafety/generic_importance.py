"""BLADE-G generic-importance term Q_ij: the expected squared perturbation that zeroing W_ij would
cause AT THE DOWNSTREAM RMSNORM READER, estimated on generic text (C4). Adds a per-weight capability
cost to the BLADE score so pruning can avoid weights that matter for general computation.

Reader mapping (Llama-3.2, verified): layers[l].self_attn.o_proj -> layers[l].post_attention_layernorm;
layers[l].mlp.down_proj -> layers[l+1].input_layernorm (or model.norm for the last layer).

Estimators (increasing fidelity):
  g0        : Q_ij = W_ij^2 * E[x_j^2]                          (Wanda-style, no output geometry)
  g1scalar  : Q_ij = W_ij^2 * gamma_i^2 * E[x_j^2 / r_t^2]      (keeps per-token norm covariance)
where x is the writer's INPUT, r_t = RMS of the reader-norm's input, gamma the reader-norm weight.
All fp32. Keys match ttsafety.sycophancy.score_edges (iter_residual_writers naming).
"""
from __future__ import annotations

import torch

from .weight_edit import iter_residual_writers
from .hooks import get_decoder_layers


def _final_norm(model):
    for path in ("model.norm", "model.language_model.norm", "norm"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot locate final norm")


def _reader_of(model, name):
    """Return (reader_module, gamma) for a residual-writer name like 'layers.5.self_attn.o_proj'.
    Uses get_decoder_layers so it works across archs (Llama/Qwen/Phi/Gemma). Architecture-aware:
      * Llama-style pre-norm blocks: o_proj -> that layer's post_attention_layernorm; down_proj ->
        next layer's input_layernorm (final norm for last layer) -- the first norm to re-read the
        residual after the writer's additive contribution.
      * Gemma-style branch-output norms: the writer's output is normalized BEFORE the residual add,
        so the first reader is on the branch: o_proj -> post_attention_layernorm (same name, branch
        semantics); down_proj -> that layer's post_feedforward_layernorm.
    gamma is the reader-norm's effective gain: Gemma RMSNorm scales by (1 + weight), others by weight."""
    layers = get_decoder_layers(model)
    l = int(name.split(".")[1])
    layer = layers[l]
    is_gemma_block = hasattr(layer, "post_feedforward_layernorm")   # Gemma-2/3 branch-output norms
    if name.endswith("o_proj"):
        rd = layer.post_attention_layernorm
    elif is_gemma_block:
        rd = layer.post_feedforward_layernorm
    else:  # Llama-style mlp.down_proj
        rd = layers[l + 1].input_layernorm if l + 1 < len(layers) else _final_norm(model)
    gamma = rd.weight.detach().float()
    if "gemma" in type(rd).__name__.lower():
        gamma = gamma + 1.0   # Gemma RMSNorm applies (1 + weight) as its gain
    return rd, gamma


def _rms_eps(norm):
    return float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))


@torch.no_grad()
def collect_c4_generic_importance(model, tokenizer, layers, components, *, text,
                                  seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=262144):
    """Return {writer_name: Q (out,in) fp32 CPU}. Packs `text` into non-overlapping seqlen windows
    (no padding -> every token real). mode in {g0, g1scalar}."""
    dev = next(model.parameters()).device
    writers = dict(iter_residual_writers(model, layers, components))
    # map each reader module id -> list of (writer_name, gamma) that read from it
    reader_for = {n: _reader_of(model, n) for n in writers}
    readers = {}   # id(module) -> (module, [writer_names])
    for n, (rd, _g) in reader_for.items():
        readers.setdefault(id(rd), [rd, []])[1].append(n)
    gamma = {n: reader_for[n][1].to(dev) for n in writers}
    eps = {n: _rms_eps(reader_for[n][0]) for n in writers}

    acc = {n: torch.zeros(writers[n].in_features, dtype=torch.float64, device=dev) for n in writers}
    xbuf = {}   # writer_name -> current-batch input (T*, in)

    def mk_writer(n):
        def hook(_m, args):
            xbuf[n] = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        return hook

    def mk_reader(rd_module, wnames):
        def hook(_m, args):
            h = args[0].detach().reshape(-1, args[0].shape[-1]).float()     # (T*, hidden)
            inv_r2 = 1.0 / (h.pow(2).mean(-1) + eps[wnames[0]])            # (T*,) ; eps ~ same
            for n in wnames:
                x = xbuf.pop(n, None)
                if x is None:
                    continue
                if mode == "g0":
                    acc[n] += x.pow(2).double().sum(0)
                else:  # g1scalar: weight each token's x^2 by 1/r_t^2 (gamma applied at the end)
                    acc[n] += (x.pow(2) * inv_r2[:, None]).double().sum(0)
        return hook

    handles = [writers[n].register_forward_pre_hook(mk_writer(n)) for n in writers]
    handles += [rd.register_forward_pre_hook(mk_reader(rd, wn)) for rd, wn in readers.values()]

    # pack text into non-overlapping seqlen windows
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n_win = min(len(ids) // seqlen, max_tokens // seqlen)
    ids = ids[: n_win * seqlen].view(n_win, seqlen)
    n_tok = 0
    try:
        for s in range(0, n_win, batch_size):
            batch = ids[s:s + batch_size].to(dev)
            model(input_ids=batch, use_cache=False)
            n_tok += batch.numel()
            xbuf.clear()
    finally:
        for hnd in handles:
            hnd.remove()

    Q = {}
    for n, mod in writers.items():
        W2 = mod.weight.detach().float().pow(2)                 # (out,in)
        a = (acc[n] / max(1, n_tok)).float()                   # (in,)
        if mode == "g0":
            Q[n] = (W2 * a[None, :]).cpu()
        else:
            Q[n] = (W2 * gamma[n].pow(2)[:, None] * a[None, :]).cpu()
    return Q, {"n_tokens": n_tok, "n_windows": n_win, "seqlen": seqlen, "mode": mode}
