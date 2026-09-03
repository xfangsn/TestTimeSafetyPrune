"""N7a: per-injection-layer local attribution for ALL layers 1..27 + actdiff.

Extends N6a (scripts/neuron_attr_local.py): every layer uses its OWN flipped
subset (N6a's non-main layers reused layer 9's subset — N7a removes that
caveat). Generation caches are shared with N6a: alpha=-2 generations live in
data/gen_alpha2_inject{l}.jsonl and are reused when present. Layers whose
flipped subset has <30 samples are low-confidence; for those, an alpha=-4
variant (fresh generations, same pipeline) is computed via --alpha4.

actdiff (no injection): instruction-token-span mean of MLP post-SwiGLU
activations on harmful_train (256) vs harmless (320); per-neuron two-sample
t-statistic |m_h - m_h0| / sqrt(v_h/n_h + v_h0/n_h0). All layers captured in
one forward per prompt.

Chunks (each saves partials immediately, safe to resume):
  python scripts/neuron_attr_alllayers.py --layers 1,2,3
  python scripts/neuron_attr_alllayers.py --layers 16,17,18 --alpha4
  python scripts/neuron_attr_alllayers.py --actdiff
  python scripts/neuron_attr_alllayers.py --finalize    # CPU-only merge/report
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neuron_attr_local import concentration, spearman  # noqa: E402
from ttsafety.data import load_jsonl  # noqa: E402
from ttsafety.hooks import capture_neurons  # noqa: E402
from ttsafety.generate import generate_texts  # noqa: E402
from ttsafety.judge import is_refusal  # noqa: E402
from ttsafety.models import chat_wrap, env_info, load_model  # noqa: E402
from ttsafety.steer import steer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_TAG = "llama32_3b_instruct"
LAYERS_ALL = list(range(1, 28))  # l=0 has no upstream block output to inject
N_NEURONS = 8192
LOW_CONF_MIN = 30
MAX_NEW_TOKENS = 128
EPS = 1e-12


def gen_path(layer, alpha):
    tag = "alpha2" if alpha == -2.0 else f"alpha{int(abs(alpha))}"
    return DATA_DIR / f"gen_{tag}_inject{layer}.jsonl"


def partial_path(layer, alpha=-2.0):
    if alpha == -2.0:
        return DATA_DIR / f"n7_attr_L{layer}.pt"
    return DATA_DIR / f"n7_attr_L{layer}_a{int(abs(alpha))}.pt"


def actdiff_path():
    return DATA_DIR / "n7_actdiff.pt"


def response_span_start(tokenizer, prompt, full):
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    m = 0
    for a, b in zip(p_ids, f_ids):
        if a != b:
            break
        m += 1
    if m >= len(f_ids):
        raise ValueError("empty response span")
    return m


def layer_response_mean(model, tokenizer, prompt, completion, layer,
                        inject_vec=None, alpha=-2.0):
    """Mean post-SwiGLU activations at `layer` over the response span."""
    full = prompt + completion + "<|eot_id|>"
    start = response_span_start(tokenizer, prompt, full)
    enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)

    def _fwd():
        caps = capture_neurons(model, enc["input_ids"], enc["attention_mask"],
                               layers=[layer])
        return caps["mlp"][layer][0, start:].mean(dim=0)

    if inject_vec is None:
        return _fwd()
    with steer(model, inject_vec, layer=layer - 1, alpha=alpha):
        return _fwd()


def ensure_gens(model, tokenizer, layer, alpha, vec, train, flip_records):
    path = gen_path(layer, alpha)
    if path.exists():
        print(f"reusing existing generations {path.name}")
        return {r["idx"]: r for r in load_jsonl(path)}
    print(f"generating at alpha={alpha}, injection layer {layer} "
          f"(hook block {layer - 1} output) ...")
    instructions = [r["instruction"] for r in train]
    with steer(model, vec, layer=layer - 1, alpha=alpha):
        outs = generate_texts(model, tokenizer, instructions, MAX_NEW_TOKENS)
    gens = {}
    for idx, (rec, out) in enumerate(zip(train, outs)):
        flipped = flip_records[idx]["refused_baseline"] and not is_refusal(out)
        gens[idx] = {"idx": idx, "instruction": rec["instruction"],
                     "output": out, "flipped": flipped}
    with path.open("w", encoding="utf-8") as f:
        for idx in sorted(gens):
            f.write(json.dumps(gens[idx], ensure_ascii=False) + "\n")
    return gens


def run_layer(model, tokenizer, directions, train, flip_records, layer, alpha):
    vec = directions[layer - 1]
    gens = ensure_gens(model, tokenizer, layer, alpha, vec, train, flip_records)
    flip_rate = sum(r["flipped"] for r in gens.values()) / len(gens)
    subset = [r for r in gens.values() if r["flipped"]]
    print(f"layer {layer} alpha={alpha}: flip rate {flip_rate:.3f}, "
          f"subset n={len(subset)}"
          + (" [LOW CONFIDENCE]" if len(subset) < LOW_CONF_MIN else ""))

    d1 = torch.zeros(N_NEURONS)
    d2 = torch.zeros(N_NEURONS)
    for n, rec in enumerate(subset):
        prompt = chat_wrap(tokenizer, rec["instruction"])
        x1 = layer_response_mean(model, tokenizer, prompt, rec["output"], layer)
        x2 = layer_response_mean(model, tokenizer, prompt, rec["output"], layer,
                                 inject_vec=vec, alpha=alpha)
        p1 = torch.softmax(x1, dim=0)
        p2 = torch.softmax(x2, dim=0)
        d1 += (p1 - p2).abs()
        d2 += torch.log(p1.clamp(min=EPS))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(subset)}")
    torch.save({"layer": layer, "alpha": alpha, "D1": d1, "D2": d2,
                "n_subset": len(subset), "flip_rate": flip_rate,
                "subset_from": "own",
                "low_confidence": len(subset) < LOW_CONF_MIN},
               partial_path(layer, alpha))
    print(f"saved {partial_path(layer, alpha)}")


def instruction_span(tokenizer, prompt, instruction):
    """(start, end) of the instruction tokens inside the wrapped prompt.

    Falls back to the full prompt span when exact subsequence matching fails
    (tokenizer boundary effects); caller counts fallbacks.
    """
    f_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    i_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    for s in range(0, len(f_ids) - len(i_ids) + 1):
        if f_ids[s:s + len(i_ids)] == i_ids:
            return s, s + len(i_ids), False
    return 0, len(f_ids), True


def run_actdiff(model, tokenizer, train, harmless):
    layers = list(range(28))
    sums = {g: torch.zeros(len(layers), N_NEURONS, dtype=torch.float64)
            for g in ("h", "h0")}
    sqs = {g: torch.zeros(len(layers), N_NEURONS, dtype=torch.float64)
           for g in ("h", "h0")}
    ns = {"h": len(train), "h0": len(harmless)}
    n_fallback = 0
    for group, records in (("h", train), ("h0", harmless)):
        for n, rec in enumerate(records):
            inst = rec["instruction"]
            prompt = chat_wrap(tokenizer, inst)
            s, e, fb = instruction_span(tokenizer, prompt, inst)
            n_fallback += fb
            enc = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=False)
            caps = capture_neurons(model, enc["input_ids"],
                                   enc["attention_mask"], layers=layers)
            for layer in layers:
                m = caps["mlp"][layer][0, s:e].mean(dim=0).to(torch.float64)
                sums[group][layer] += m
                sqs[group][layer] += m * m
            del caps
            if (n + 1) % 100 == 0:
                print(f"  actdiff {group}: {n + 1}/{len(records)}")
    mean = {g: sums[g] / ns[g] for g in sums}
    var = {g: (sqs[g] - sums[g] ** 2 / ns[g]) / (ns[g] - 1) for g in sums}
    t = ((mean["h"] - mean["h0"]).abs()
         / (var["h"] / ns["h"] + var["h0"] / ns["h0"]).sqrt().clamp(min=EPS))
    torch.save({"layers": layers, "t_stat": t,  # (28, 8192)
                "n_harmful": ns["h"], "n_harmless": ns["h0"],
                "span": "instruction", "n_span_fallback": n_fallback},
               actdiff_path())
    print(f"saved {actdiff_path()} (span fallbacks: {n_fallback})")


def finalize():
    per_layer = {}
    missing = []
    for layer in LAYERS_ALL:
        p = partial_path(layer)
        if not p.exists():
            missing.append(layer)
            continue
        m = torch.load(p, weights_only=True)
        rho = (spearman(m["D1"].numpy(), m["D2"].numpy())
               if m["n_subset"] > 0 else None)
        entry = {"flip_rate": m["flip_rate"], "n_subset": m["n_subset"],
                 "low_confidence": m["low_confidence"],
                 "spearman_D1_D2": rho,
                 "D1": concentration(m["D1"].numpy()),
                 "D2": concentration(m["D2"].numpy())}
        p4 = partial_path(layer, alpha=-4.0)
        if p4.exists():
            m4 = torch.load(p4, weights_only=True)
            rho4 = (spearman(m4["D1"].numpy(), m4["D2"].numpy())
                    if m4["n_subset"] > 0 else None)
            entry["alpha4"] = {
                "flip_rate": m4["flip_rate"], "n_subset": m4["n_subset"],
                "spearman_D1_D2": rho4,
                "D1": concentration(m4["D1"].numpy()),
                "D2": concentration(m4["D2"].numpy())}
            m["D1_alpha4"], m["D2_alpha4"] = m4["D1"], m4["D2"]
        per_layer[layer] = (m, entry)
    if missing:
        print(f"WARNING: missing layers {missing} — partial finalize")

    act = torch.load(actdiff_path(), weights_only=True) \
        if actdiff_path().exists() else None

    # merged tensor artifact for N7b global top-k selection
    merged = {"layers": LAYERS_ALL,
              "D1": torch.stack([per_layer[l][0]["D1"] for l in per_layer])
              if not missing else None,
              "D2": torch.stack([per_layer[l][0]["D2"] for l in per_layer])
              if not missing else None,
              "low_confidence": [l for l in per_layer
                                 if per_layer[l][1]["low_confidence"]],
              "alpha4": {str(l): {"D1": per_layer[l][0]["D1_alpha4"],
                                  "D2": per_layer[l][0]["D2_alpha4"]}
                         for l in per_layer if "D1_alpha4" in per_layer[l][0]},
              "actdiff": act["t_stat"] if act else None}
    if not missing and act is not None:
        torch.save(merged, DATA_DIR / "neuron_importance_local_alllayers.pt")
        print(f"saved {DATA_DIR / 'neuron_importance_local_alllayers.pt'}")

    report = {"config": {"model": MODEL_TAG, "layers": LAYERS_ALL,
                         "alpha": -2.0, "alpha4_low_conf": -4.0,
                         "low_conf_min": LOW_CONF_MIN,
                         "max_new_tokens": MAX_NEW_TOKENS,
                         "subset": "own per layer",
                         "missing_layers": missing},
              "env": env_info(),
              "per_layer": {str(l): e for l, (m, e) in sorted(per_layer.items())}}
    if act is not None:
        t = act["t_stat"].numpy()
        layer_score = t.max(axis=1)  # strongest neuron per layer
        order = np.argsort(layer_score)[::-1]
        report["actdiff"] = {
            "n_harmful": act["n_harmful"], "n_harmless": act["n_harmless"],
            "span": act["span"], "n_span_fallback": act["n_span_fallback"],
            "top_layers_by_max_t": [
                {"layer": int(l), "max_t": float(layer_score[l])}
                for l in order[:8]],
            "per_layer_max_t": {str(l): float(layer_score[l])
                                for l in range(len(layer_score))},
            "concentration": {str(l): concentration(t[l])
                              for l in range(1, 28)},
        }
    out = RESULTS_DIR / "n7_attribution.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")

    for l, (m, e) in sorted(per_layer.items()):
        rho = f"{e['spearman_D1_D2']:+.3f}" if e["spearman_D1_D2"] is not None \
            else "  n/a"
        line = (f"L{l}: flip {e['flip_rate']:.3f} n={e['n_subset']}"
                f"{' LOW' if e['low_confidence'] else ''} rho "
                f"{rho} D1top1% {e['D1']['top1pct_share']:.3f}"
                f" D2top1% {e['D2']['top1pct_share']:.3f}")
        if "alpha4" in e:
            line += (f" | a4: flip {e['alpha4']['flip_rate']:.3f} "
                     f"n={e['alpha4']['n_subset']}")
        print(line)

    # flip-rate curve + D1/D2 top-1% share across layers
    if per_layer:
        ls = sorted(per_layer)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(ls, [per_layer[l][1]["flip_rate"] for l in ls], "o-",
                     label="alpha=-2")
        a4 = [l for l in ls if "alpha4" in per_layer[l][1]]
        if a4:
            axes[0].plot(a4, [per_layer[l][1]["alpha4"]["flip_rate"]
                              for l in a4], "s--", label="alpha=-4")
        axes[0].set_xlabel("injection layer")
        axes[0].set_ylabel("flip rate (harmful_train)")
        axes[0].legend()
        axes[1].plot(ls, [per_layer[l][1]["D1"]["top1pct_share"] for l in ls],
                     "o-", label="D1 top1%")
        axes[1].plot(ls, [per_layer[l][1]["D2"]["top1pct_share"] for l in ls],
                     "s-", label="D2 top1%")
        axes[1].set_xlabel("injection layer")
        axes[1].set_ylabel("top-1% importance share")
        axes[1].legend()
        fig.suptitle("N7a: per-injection-layer flip rate and concentration")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "n7_attribution.png", dpi=150)
        print(f"saved {RESULTS_DIR / 'n7_attribution.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None)
    ap.add_argument("--alpha4", action="store_true",
                    help="run the alpha=-4 variant for the given layers")
    ap.add_argument("--actdiff", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
        return
    model, tokenizer = load_model()
    train = load_jsonl(DATA_DIR / "harmful_train.jsonl")
    if args.actdiff:
        harmless = load_jsonl(DATA_DIR / "harmless.jsonl")
        run_actdiff(model, tokenizer, train, harmless)
        return
    layers = [int(x) for x in args.layers.split(",")]
    directions = torch.load(
        DATA_DIR / "directions" / f"refusal_{MODEL_TAG}.pt", weights_only=True)
    flip_records = {r["idx"]: r for r in load_jsonl(DATA_DIR / "flip_alphas.jsonl")}
    alpha = -4.0 if args.alpha4 else -2.0
    for layer in layers:
        run_layer(model, tokenizer, directions, train, flip_records, layer,
                  alpha)


if __name__ == "__main__":
    main()
