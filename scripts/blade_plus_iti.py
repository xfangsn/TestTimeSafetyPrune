"""Run ONLY the two BLADE+ITI variants requested on 2026-09-06.

This is a legacy-compatible exploratory extension, not the redesigned prospective
study: historical prompts, fit split, norm-ranked heads, decode-only generation.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from ttsafety.eval import load_c4_text, teacher_forced_ppl
from ttsafety.behaviors import solo_layer_pool, bestfirst_layers
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.hooks import capture_last_token, get_decoder_layers
from ttsafety.iti_composition import (
    collect_head_acts, collect_writer_means, fit_legacy_iti, generate_batch,
    iti_hook, qwen_no_thinking, scaled_selection, strict_selection, tensor_dict_hash,
)
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.weight_prune import _resolve_modules


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_split(rows):
    """Exactly historical split_3way membership (not a new leakage claim)."""
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    rng = random.Random(0)
    train, select, evaluate = [], [], []
    for group in by_family.values():
        entities = sorted({x["entity"] for x in group})
        rng.shuffle(entities)
        n1 = max(1, int(.45 * len(entities)))
        n2 = max(1, int(.25 * len(entities)))
        a, b = set(entities[:n1]), set(entities[n1:n1 + n2])
        for x in group:
            (train if x["entity"] in a else select if x["entity"] in b else evaluate).append(x)
    return train, select, evaluate


def median_positive(tensors):
    values = torch.cat([x.flatten() for x in tensors.values()]).float()
    positive = values[values > 0]
    if not positive.numel():
        raise ValueError("Empty positive score distribution")
    return positive.median().item()


@torch.no_grad()
def smoke(model, tok, selection, iti, prompt):
    mods = _resolve_modules(model, list(selection))
    before = {n: mods[n].weight.view(-1)[idx.to(model.device)].clone() for n, idx in selection.items()}
    handles = sum(len(m._forward_pre_hooks) for m in model.modules())
    enc = tok(qwen_no_thinking(tok, prompt), return_tensors="pt", add_special_tokens=False).to(model.device)
    raw = model(**enc, use_cache=False).logits[:, -1].float()
    with scaled_selection(model, selection, 1.0), iti_hook(model, {i: v * 0 for i, v in iti["add_unit"].items()}):
        identity = model(**enc, use_cache=False).logits[:, -1].float()
    torch.testing.assert_close(raw, identity, rtol=0, atol=0)
    try:
        with scaled_selection(model, selection, 2.0):
            for n, idx in selection.items():
                assert torch.equal(mods[n].weight.view(-1)[idx.to(model.device)], before[n] * 2)
            with iti_hook(model, {i: v * 2 for i, v in iti["add_unit"].items()}):
                assert torch.isfinite(model(**enc, use_cache=False).logits[:, -1]).all()
                raise RuntimeError("intentional restoration smoke")
    except RuntimeError as err:
        if str(err) != "intentional restoration smoke":
            raise
    for n, idx in selection.items():
        assert torch.equal(mods[n].weight.view(-1)[idx.to(model.device)], before[n])
    assert sum(len(m._forward_pre_hooks) for m in model.modules()) == handles
    with scaled_selection(model, selection, 2.0):
        plain = model(**enc, use_cache=False).logits[:, -1].float()
        with iti_hook(model, {i: v * 2 for i, v in iti["add_unit"].items()}):
            skipped = model(**enc, use_cache=False).logits[:, -1].float()
        torch.testing.assert_close(plain, skipped, rtol=0, atol=0)
        # Use a short generated prefix only as a numerical smoke; no OOD baseline rerun.
        completion = generate_batch(model, tok, [prompt], max_new_tokens=2)
        with iti_hook(model, {i: v * 0 for i, v in iti["add_unit"].items()}):
            zero = generate_batch(model, tok, [prompt], max_new_tokens=2)
        assert completion == zero
    return {"identity_exact": True, "exception_restoration_exact": True,
            "zero_iti_generation_exact": True, "legacy_prefill_skipped_exact": True,
            "hooks_restored": True}


def fit(args, inputs):
    start = time.monotonic()
    model, tok = load_model(inputs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    layers = list(range(len(get_decoder_layers(model))))
    rows = inputs["contrast_rows"]
    train, select, _ = legacy_split(rows)
    unc = [r["question"] for r in train if r["label"] == 1]
    cert = [r["question"] for r in train if r["label"] == 0]
    wrapped = qwen_no_thinking(tok, "What is the capital of France?")
    assert "</think>" in wrapped, "Template did not close the thinking block"
    print(f"fit FULL BLADE model={inputs['model']} layers={layers} train={len(train)} select={len(select)}", flush=True)
    au = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in unc], batch_size=16)
    ac = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in cert], batch_size=16)
    directions = {i: au[i].mean(0) - ac[i].mean(0) for i in layers}
    del au, ac
    mu_u = collect_writer_means(model, tok, unc, layers)
    mu_c = collect_writer_means(model, tok, cert, layers)
    c4 = load_c4_text()
    print("collecting g1scalar Q, 65536 tokens, no_grad", flush=True)
    q, qmeta = collect_c4_generic_importance(model, tok, layers, "both", text=c4,
                                            seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    raw_scores = score_edges(model, directions, mu_u, mu_c, layers, "both")
    lam = median_positive(raw_scores) / median_positive(q)
    del raw_scores
    gc.collect()
    # Use the existing BLADE ELS implementation and its unchanged hyperparameters.
    # Only the FINAL mask rho and amplification factor are the user overrides.
    els_trace = []
    current_layers = []

    def sfn(m, d, a, b, ls, comp):
        current_layers[:] = ls
        raw = score_edges_g(m, d, a, b, ls, comp, Q=q, lam=lam, abstain=True)
        # Exact score_fn_for behavior in the existing OOD BLADE implementation.
        # The final mask is additionally audited below, without changing its ranking.
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in raw.items()}

    def ppl_now():
        p = teacher_forced_ppl(model, tok, c4, max_tokens=5000)
        els_trace.append({"layers": list(current_layers), "ppl": p})
        atomic_json(args.run_dir / "els_progress.json", els_trace)
        return p

    # Same uncertainty lexical proxy as blade_epistemic_els.is_unc.
    markers = inputs["uncertainty_markers"]
    prompts_select = [x["question"] for x in select if x["label"] == 1]

    def measure():
        outputs = []
        for start in range(0, len(prompts_select), 16):
            outputs.extend(generate_batch(model, tok, prompts_select[start:start + 16], max_new_tokens=128))
        rate = sum(any(m in x["generation"].lower() for m in markers) for x in outputs) / len(outputs)
        p = ppl_now()
        els_trace[-1]["uncertainty_rate"] = rate
        atomic_json(args.run_dir / "els_progress.json", els_trace)
        print(f"ELS cand={list(current_layers)} rate={rate:.4f} ppl={p:.4f}", flush=True)
        return rate, p

    base_ppl = ppl_now()
    base_sel = measure()[0]
    print("running complete solo-pool + best-first ELS (screen/test .005, beta .05, eps .005)", flush=True)
    pool = solo_layer_pool(model, directions, mu_u, mu_c, layers, "both", ppl_now, base_ppl,
                           screen_frac=.005, beta=.05, score_fn=sfn)
    L_star = bestfirst_layers(model, directions, mu_u, mu_c, pool, "both", measure,
                              base_sel, base_ppl, beta=.05, eps=.005, test_frac=.005, score_fn=sfn)
    if not L_star:
        raise ValueError("Full BLADE ELS returned no layers; no historical-layer fallback allowed")
    print(f"FULL BLADE ELS complete: L*={L_star}", flush=True)
    scores = score_edges_g(model, directions, mu_u, mu_c, L_star, "both",
                          Q=q, lam=lam, abstain=True)
    selection = strict_selection(scores, inputs["rho"])
    mask_meta = {"lambda": lam, "lambda_effective": lam * abs(inputs["blade_alpha"] - 1),
                 "n_edges": sum(x.numel() for x in selection.values()),
                 "eligible_edges": sum(x.numel() for x in scores.values()),
                 "positive_per_matrix": {k: int((torch.isfinite(v) & (v > 0)).sum()) for k, v in scores.items()},
                 "mask_hash": tensor_dict_hash(selection), "Q": qmeta}
    mask_meta["fraction_all_parameters"] = mask_meta["n_edges"] / sum(p.numel() for p in model.parameters())
    print(f"mask {mask_meta['n_edges']} edges lambda={lam:.5g}", flush=True)
    del q, scores, mu_u, mu_c, directions
    gc.collect(); torch.cuda.empty_cache()
    # Uncertain followed by certain matches the old class-batched calibration.
    iti_rows = [x for x in rows if x["label"] == 1] + [x for x in rows if x["label"] == 0]
    prompts = [r["question"] for r in iti_rows]
    labels = [r["label"] for r in iti_rows]
    print("reconstructing base-model ITI (historical head IDs fixed)", flush=True)
    acts = collect_head_acts(model, tok, prompts)
    base = fit_legacy_iti(acts, labels, k=48, fixed_heads=inputs["historical_iti_heads"])
    del acts
    print(f"base ranking overlap {base['historical_head_overlap']}/48", flush=True)
    check = smoke(model, tok, selection, base, prompts[0])
    print("refitting ITI under BLADE rho=.01 alpha=2 (reselect heads + refit direction/sigma)", flush=True)
    with scaled_selection(model, selection, inputs["blade_alpha"]):
        acts = collect_head_acts(model, tok, prompts)
        refit = fit_legacy_iti(acts, labels, k=48)
    del acts
    check_refit = smoke(model, tok, selection, refit, prompts[0])
    same = sorted(set(base["heads"]) & set(refit["heads"]))
    diagnostics = {"head_overlap": len(same), "common_head_cosines": {
        str(k): float(F.cosine_similarity(base["directions"][k], refit["directions"][k], dim=0)) for k in same},
        "common_head_sigma_ratios": {str(k): refit["sigmas"][k] / base["sigmas"][k] for k in same}}
    artifact = {"selection": selection, "base_iti": base, "refit_iti": refit,
                "inputs_hash": file_hash(args.inputs), "model": inputs["model"],
                "model_commit": getattr(model.config, "_commit_hash", None),
                "template_hash": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                "thinking": False, "rho": inputs["rho"], "blade_alpha": inputs["blade_alpha"],
                "L_star": L_star, "mask": mask_meta, "diagnostics": diagnostics,
                "els": {"pool": pool, "screen_frac": .005, "test_frac": .005,
                        "beta": .05, "eps": .005, "base_select_rate": base_sel,
                        "base_ppl_c4": base_ppl, "trace": els_trace,
                        "source": "full existing solo_layer_pool + bestfirst_layers, unchanged defaults"},
                "smoke_base": check, "smoke_refit": check_refit, "env": env_info(),
                "c4_text_hash": hashlib.sha256(c4.encode()).hexdigest(),
                "elapsed_seconds": time.monotonic() - start,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    tmp = args.run_dir / "artifact.pt.tmp"
    torch.save(artifact, tmp)
    tmp.replace(args.run_dir / "artifact.pt")
    atomic_json(args.run_dir / "fit.json", {k: v for k, v in artifact.items()
                if k not in ("selection", "base_iti", "refit_iti")})
    print(f"FIT COMPLETE {artifact['elapsed_seconds']:.1f}s", flush=True)


@torch.no_grad()
def legacy_ppl(model, tok, text):
    # Exact historical windows; only a stress metric, clearly distinct from generation policy.
    ids = tok(text, return_tensors="pt")["input_ids"][0, :5000]
    windows = ids[:(len(ids) // 1024) * 1024].reshape(-1, 1024)
    if not len(windows):
        raise ValueError("No legacy PPL windows")
    nlls = []
    for window in windows:
        x = window[None].to(model.device)
        nlls.append(float(model(input_ids=x, labels=x, use_cache=False).loss))
    return {"ppl": math.exp(sum(nlls) / len(nlls)), "window_nll": nlls,
            "scored_tokens": len(windows) * 1023,
            "window_ids_hash": tensor_dict_hash({0: windows}), "policy": "all_positions_legacy_stress"}


def generate(args, inputs):
    artifact = torch.load(args.run_dir / "artifact.pt", map_location="cpu", weights_only=False)
    assert artifact["inputs_hash"] == file_hash(args.inputs)
    model, tok = load_model(inputs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert hashlib.sha256(tok.chat_template.encode()).hexdigest() == artifact["template_hash"]
    assert getattr(model.config, "_commit_hash", None) == artifact["model_commit"]
    iti = artifact["base_iti"] if args.variant == "transfer" else artifact["refit_iti"]
    c4 = load_c4_text()
    assert hashlib.sha256(c4.encode()).hexdigest() == artifact["c4_text_hash"]
    for dose in inputs["iti_doses"]:
        name = f"{args.variant}_c{dose:g}"
        outpath = args.run_dir / f"{name}.json"
        out = json.loads(outpath.read_text()) if outpath.exists() else {
            "condition": name, "variant": args.variant, "iti_dose": dose,
            "rho": inputs["rho"], "blade_alpha": inputs["blade_alpha"],
            "L_star": artifact["L_star"], "thinking": False, "env": env_info(),
            "generation_policy": "legacy_decode_only_skip_entire_prefill",
            "iti_head_rule": "historical_fixed_ids" if args.variant == "transfer" else "legacy_mean_diff_norm_top48",
            "artifact_hash": file_hash(args.run_dir / "artifact.pt"),
            "inputs_hash": artifact["inputs_hash"], "n_expected": len(inputs["items"]),
            "items": [], "status": "running"}
        assert out["artifact_hash"] == file_hash(args.run_dir / "artifact.pt")
        if out["status"] == "complete":
            continue
        begin = time.monotonic()
        add = {i: dose * v for i, v in iti["add_unit"].items()}
        print(f"START {name} n={len(inputs['items'])} resume={len(out['items'])}", flush=True)
        with scaled_selection(model, artifact["selection"], inputs["blade_alpha"]):
            with iti_hook(model, add):
                for start in range(len(out["items"]), len(inputs["items"]), args.batch_size):
                    items = inputs["items"][start:start + args.batch_size]
                    rows = generate_batch(model, tok, [x["question"] for x in items])
                    out["items"].extend({**x, **g} for x, g in zip(items, rows))
                    atomic_json(outpath, out)
                    print(f"{name} {len(out['items'])}/{len(inputs['items'])}", flush=True)
            with iti_hook(model, add, policy="all_positions"):
                out["legacy_stress_ppl"] = legacy_ppl(model, tok, c4)
        out["legacy_reference_ppl"] = inputs["legacy_reference_ppl"]
        out["legacy_stress_ppl"]["delta_vs_historical_base"] = out["legacy_stress_ppl"]["ppl"] / inputs["legacy_reference_ppl"] - 1
        out["legacy_stress_ppl"]["same_run_ELS_base_ppl"] = artifact["els"]["base_ppl_c4"]
        out["legacy_stress_ppl"]["delta_vs_same_run_ELS_base"] = out["legacy_stress_ppl"]["ppl"] / artifact["els"]["base_ppl_c4"] - 1
        out["comparison_caveat"] = "Historical baseline on different software/GPU; PPL policy differs from generation; exploratory only."
        out.update(status="complete", current_attempt_seconds=time.monotonic() - begin,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated())
        atomic_json(outpath, out)
        print(f"DONE {name} ppl={out['legacy_stress_ppl']['ppl']:.3f} elapsed={time.monotonic()-begin:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "generate"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["transfer", "refit"], default="transfer")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(args.inputs.read_text())
    assert inputs["rho"] == 0.01 and inputs["blade_alpha"] == 2.0
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    try:
        (fit if args.stage == "fit" else generate)(args, inputs)
    except Exception as err:
        atomic_json(args.run_dir / f"failure_{args.stage}_{args.variant}.json",
                    {"type": type(err).__name__, "error": str(err), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
