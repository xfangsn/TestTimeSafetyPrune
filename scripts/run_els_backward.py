#!/usr/bin/env python3
"""Resumable GPU runner for the scheme-3 forward/backward ELS experiment.

Large tensors and every checkpoint live below --run-root.  The four commands are
intended for Slurm dependencies: fit -> pool -> search (repeatable slices).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


SETTINGS = dict(model="Qwen/Qwen3-8B", components="both", rho_sweep=[.005,.01],
                generation_batch_size=16, max_new_tokens=128,
                ppl_max_tokens=5000, ppl_window=1024, ppl_batch_size=8,
                q_mode="g1scalar", q_max_tokens=65536, q_seqlen=2048,
                q_batch_size=2, max_requests=672)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def digest(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False).encode())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def git_info(root: Path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def common(args):
    import torch
    from blade_plus_iti import file_hash as _unused, legacy_split, median_positive
    from ttsafety.eval import load_c4_text, teacher_forced_ppl
    from ttsafety.hooks import get_decoder_layers
    from ttsafety.iti_composition import generate_batch, qwen_no_thinking, tensor_dict_hash
    from ttsafety.models import env_info, load_model
    from ttsafety.sycophancy import score_edges_g

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fit/pool/search")
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    inputs = json.loads(args.inputs.read_text())
    model, tok = load_model(inputs.get("model", SETTINGS["model"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert "</think>" in qwen_no_thinking(tok, "What is the capital of France?")
    artifact_path = args.run_root / "fit" / "artifact.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    expected = json.loads((artifact_path.with_suffix(".json")).read_text())
    if file_hash(artifact_path) != expected["artifact_sha256"]:
        raise RuntimeError("fit artifact hash mismatch")
    if file_hash(args.inputs) != artifact["inputs_hash"]:
        raise RuntimeError("inputs differ from the immutable fit artifact")
    source_now = git_info(Path(__file__).resolve().parents[1])
    if artifact.get("source") != source_now or source_now["dirty"]:
        raise RuntimeError("source checkout differs from the clean fit artifact")
    if getattr(model.config, "_commit_hash", None) != artifact["model_commit"]:
        raise RuntimeError("model commit differs from fit artifact")
    template_hash = digest_bytes(tok.chat_template.encode())
    if template_hash != artifact["template_hash"]:
        raise RuntimeError("tokenizer template differs from fit artifact")
    directions = {int(k): v.to(model.device) for k, v in artifact["directions"].items()}
    mu_a = {k: v.to(model.device) for k, v in artifact["mu_a"].items()}
    mu_b = {k: v.to(model.device) for k, v in artifact["mu_b"].items()}
    q = {k: v.to(model.device) for k, v in artifact["q"].items()}
    c4 = load_c4_text()
    if digest_bytes(c4.encode()) != artifact["c4_hash"]:
        raise RuntimeError("C4 text differs from fit artifact")
    _, select, _ = legacy_split(inputs["contrast_rows"])
    prompts = [r["question"] for r in select if r["label"] == 1]
    markers = inputs["uncertainty_markers"]

    def score(m, d, a, b, ls, comp):
        raw = score_edges_g(m, d, a, b, ls, comp, Q=q, lam=artifact["lambda"], abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in raw.items()}

    def ppl():
        return teacher_forced_ppl(model, tok, c4, max_tokens=5000, window=1024, batch_size=8)

    def generate():
        out = []
        for i in range(0, len(prompts), 16):
            out.extend(generate_batch(model, tok, prompts[i:i + 16], max_new_tokens=128))
        positives = sum(any(x in row["generation"].lower() for x in markers) for row in out)
        return out, positives

    return locals()


def fit(args):
    import torch
    from blade_plus_iti import legacy_split, median_positive
    from ttsafety.eval import load_c4_text, teacher_forced_ppl
    from ttsafety.generic_importance import collect_c4_generic_importance
    from ttsafety.hooks import capture_last_token, get_decoder_layers
    from ttsafety.iti_composition import (collect_writer_means, generate_batch,
                                          qwen_no_thinking, tensor_dict_hash)
    from ttsafety.models import env_info, load_model
    from ttsafety.sycophancy import score_edges

    out = args.run_root / "fit"
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "artifact.pt"
    if artifact_path.exists():
        meta = json.loads(artifact_path.with_suffix(".json").read_text())
        if file_hash(artifact_path) != meta["artifact_sha256"]:
            raise RuntimeError("existing artifact failed hash validation")
        stale = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if stale.get("version") != 2 or "base" not in stale or stale.get("settings") != SETTINGS:
            raise RuntimeError("existing artifact uses an incompatible schema; use a fresh run root")
        print("fit artifact already complete", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    inputs = json.loads(args.inputs.read_text())
    model, tok = load_model(inputs.get("model", SETTINGS["model"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert "</think>" in qwen_no_thinking(tok, "What is the capital of France?")
    layers = list(range(len(get_decoder_layers(model))))
    train, select, _ = legacy_split(inputs["contrast_rows"])
    unc = [r["question"] for r in train if r["label"] == 1]
    cert = [r["question"] for r in train if r["label"] == 0]
    # Match benchmark_els_bounds.py exactly: direction is the difference of
    # last prompt-token activations after the no-thinking chat template.
    au = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in unc], batch_size=16)
    ac = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in cert], batch_size=16)
    directions = {i: au[i].mean(0) - ac[i].mean(0) for i in layers}
    del au, ac
    mu_a = collect_writer_means(model, tok, unc, layers)
    mu_b = collect_writer_means(model, tok, cert, layers)
    c4 = load_c4_text()
    q, qmeta = collect_c4_generic_importance(model, tok, layers, "both", text=c4,
                                              seqlen=2048, batch_size=2,
                                              mode="g1scalar", max_tokens=65536)
    raw = score_edges(model, directions, mu_a, mu_b, layers, "both")
    lam = median_positive(raw) / median_positive(q)
    c4_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=5000, window=1024, batch_size=8)
    select_prompts = [r["question"] for r in select if r["label"] == 1]
    base_outputs = []
    for i in range(0, len(select_prompts), 16):
        base_outputs.extend(generate_batch(model, tok, select_prompts[i:i+16], max_new_tokens=128))
    markers = inputs["uncertainty_markers"]
    base_positives = sum(any(x in row["generation"].lower() for x in markers) for row in base_outputs)
    artifact = {"version": 2, "settings": SETTINGS, "inputs_hash": file_hash(args.inputs),
                "model": inputs.get("model", SETTINGS["model"]),
                "model_commit": getattr(model.config, "_commit_hash", None),
                "template_hash": digest_bytes(tok.chat_template.encode()),
                "c4_hash": digest_bytes(c4.encode()), "layers": layers, "lambda": lam,
                "base":{"ppl":c4_ppl,"metric":base_positives/len(select_prompts),
                        "positives":base_positives,"outputs":base_outputs},
                "source":git_info(Path(__file__).resolve().parents[1]),
                "fit_gpu":torch.cuda.get_device_name(),
                "directions": {k: v.cpu() for k, v in directions.items()},
                "mu_a": {k: v.cpu() for k, v in mu_a.items()},
                "mu_b": {k: v.cpu() for k, v in mu_b.items()},
                "q": {k: v.cpu() for k, v in q.items()}, "q_meta": qmeta}
    tmp = artifact_path.with_suffix(".tmp")
    torch.save(artifact, tmp)
    tmp.replace(artifact_path)
    meta = {"status": "complete", "artifact_sha256": file_hash(artifact_path),
            "tensor_hashes": {"directions": tensor_dict_hash(artifact["directions"]),
                              "mu_a": tensor_dict_hash(artifact["mu_a"]),
                              "mu_b": tensor_dict_hash(artifact["mu_b"]),
                              "q": tensor_dict_hash(artifact["q"])},
            "env": env_info(), "host": socket.gethostname()}
    atomic_json(artifact_path.with_suffix(".json"), meta)
    artifact_path.chmod(0o444)
    print(f"saved immutable fit artifact {artifact_path}", flush=True)


def pool(args):
    from ttsafety.behaviors import solo_layer_pool
    from ttsafety.els_cache import ELSCandidateCache
    from ttsafety.weight_prune import pruned_weights, selection_from_ranking

    ctx = common(args); model = ctx["model"]; artifact = ctx["artifact"]
    out = args.run_root / f"rho-{args.rho:g}_beta-{args.beta:g}" / "pool.json"
    if out.exists() and json.loads(out.read_text()).get("status") == "complete":
        print("pool already complete", flush=True); return
    base_ppl = artifact["base"]["ppl"]
    cache = ELSCandidateCache(ctx["score"])
    current = []
    def rank(m, d, a, b, ls, comp, max_fraction):
        current[:] = ls
        return cache.rank(m, d, a, b, ls, comp, max_fraction)
    def measure_ppl(): return ctx["ppl"]()
    chosen = solo_layer_pool(model, ctx["directions"], ctx["mu_a"], ctx["mu_b"],
        artifact["layers"], "both", measure_ppl, base_ppl, screen_frac=args.rho,
        beta=args.beta, ranking_fn=rank)
    atomic_json(out, {"status":"complete", "rho":args.rho, "beta":args.beta, "base_ppl":base_ppl,
                      "pool":chosen, "artifact_sha256":file_hash(args.run_root / "fit/artifact.pt")})
    print(f"pool beta={args.beta:g}: {chosen}", flush=True)


class SliceExpired(RuntimeError): pass


def search(args):
    import torch
    from ttsafety.els_backward import backward_layers, EvaluationBudgetExceeded
    from ttsafety.els_cache import ELSCandidateCache
    from ttsafety.weight_prune import pruned_weights, selection_from_ranking

    slice_start = time.monotonic()
    run_dir = args.run_root / f"rho-{args.rho:g}_beta-{args.beta:g}_eps-{args.eps:g}" / args.direction
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "complete":
        print("search already complete", flush=True); return
    ctx = common(args); model = ctx["model"]
    pool_meta = json.loads((args.run_root / f"rho-{args.rho:g}_beta-{args.beta:g}/pool.json").read_text())
    pool_layers = pool_meta["pool"]
    run_dir.mkdir(parents=True, exist_ok=True)
    # Per-arm cache avoids concurrent writers. This deliberately gives up
    # cross-epsilon metric sharing in exchange for crash-safe JSON checkpoints.
    cache_dir = run_dir / "candidate-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    requests_path = run_dir / "requested_candidates.json"
    requested = set(json.loads(requests_path.read_text())) if requests_path.exists() else set()
    ranking_cache = ELSCandidateCache(ctx["score"])
    positive_by_layer = {}
    artifact_hash = file_hash(args.run_root / "fit/artifact.pt")
    deadline = slice_start + args.slice_minutes * 60
    counters = {"logical_requests":0, "new_requests":0, "cache_hits":0}

    def candidate_path(layers):
        fit_meta=json.loads((args.run_root/"fit/artifact.json").read_text())
        key = digest({"artifact":artifact_hash,"fit_env":digest(fit_meta.get("env")),
                      "source_commit":git_info(Path(__file__).resolve().parents[1])["commit"],
                      "gpu":torch.cuda.get_device_name(),"layers":list(layers),"rho":args.rho,
                      "ppl": [5000,1024,8],"generation":[128,16],
                      "mask_policy":"zero_clamped_global", "prompt_hash":digest(ctx["prompts"])})
        return cache_dir / (key + ".json")

    def evaluate(layers, incumbent=None, need_metric=True):
        layers = tuple(sorted(layers)); counters["logical_requests"] += 1
        path = candidate_path(layers)
        candidate_id = path.stem
        row = json.loads(path.read_text()) if path.exists() else None
        if row is not None and row.get("status") != "complete":
            raise RuntimeError(f"invalid candidate cache row {path}")
        if row and (not need_metric or row.get("metric") is not None):
            counters["cache_hits"] += 1
            return row
        if candidate_id not in requested and len(requested) >= SETTINGS["max_requests"]:
            raise EvaluationBudgetExceeded("672 unique candidate evaluations exhausted")
        if time.monotonic() >= deadline:
            raise EvaluationBudgetExceeded("wall-clock slice exhausted")
        if candidate_id not in requested:
            requested.add(candidate_id)
            atomic_json(requests_path, sorted(requested))
            counters["new_requests"] += 1
        ranking = ranking_cache.rank(model, ctx["directions"], ctx["mu_a"], ctx["mu_b"],
                                     layers, "both", max(args.rho, .01)) if layers else {}
        selection = selection_from_ranking(ranking, args.rho) if layers else {}
        # Scores are finite, zero-clamped, and selected from one descending
        # global prefix. Count cached positive candidates tensor-wise; avoid
        # materializing a Python map containing millions of scalar indices.
        for layer in layers:
            if layer not in positive_by_layer:
                positive_by_layer[layer] = sum(
                    int((entry.values > 0).sum())
                    for entry in ranking_cache._layers[layer].values())
        cached_positive=sum(positive_by_layer[layer] for layer in layers)
        from ttsafety.weight_prune import _resolve_modules
        modules = _resolve_modules(model, list(selection))
        before = {n: modules[n].weight.view(-1)[idx.to(model.device)].clone()
                  for n, idx in selection.items()}
        with pruned_weights(model, selection):
            ppl_value = row["ppl"] if row is not None else ctx["ppl"]()
            outputs = None if row is None else row.get("outputs")
            positives = None if row is None else row.get("positives")
            metric = None if row is None else row.get("metric")
            feasible = (ppl_value - pool_meta["base_ppl"]) / pool_meta["base_ppl"] <= args.beta
            if need_metric and feasible and metric is None:
                outputs, positives = ctx["generate"](); metric = positives / len(ctx["prompts"])
        for n, idx in selection.items():
            if not torch.equal(modules[n].weight.view(-1)[idx.to(model.device)], before[n]):
                raise RuntimeError(f"weight restoration failed for {n}")
        requested_edges=max(1,round(args.rho*ranking["total_pool_weights"])) if layers else 0
        actual_edges=sum(v.numel() for v in selection.values())
        row = {"status":"complete", "layers":list(layers), "ppl":ppl_value,
               "metric":metric, "positives":positives, "n_total":len(ctx["prompts"]),
               "outputs":outputs, "feasible":feasible,
               "mask_hash": ctx["tensor_dict_hash"](selection) if selection else None,
               "n_edges":actual_edges,"requested_n_edges":requested_edges,
               "edge_shortfall":requested_edges-actual_edges,
               "restoration_exact":True,
               "positive_selected":min(actual_edges,cached_positive),
               "nonpositive_selected":actual_edges-min(actual_edges,cached_positive)}
        atomic_json(path, row)
        return row

    base = ctx["artifact"]["base"]
    if base["ppl"] != pool_meta["base_ppl"]:
        raise RuntimeError("pool base PPL differs from immutable fit base")
    status="complete"; result=None
    try:
        if args.direction == "backward":
            result = backward_layers(pool_layers, evaluate, base_metric=base["metric"],
                base_ppl=base["ppl"], beta=args.beta, eps=args.eps)
            status = result.status
            selected = list(result.layers)
            trace = [vars(x) if hasattr(x, "__dict__") else x for x in result.steps]
        else:
            state_path=run_dir/"state.json"
            state=json.loads(state_path.read_text()) if state_path.exists() else {"selected":[],"metric":base["metric"],"trace":[]}
            selected=list(state["selected"])
            while True:
                best=None
                for layer in pool_layers:
                    if layer in selected: continue
                    row=evaluate(tuple(selected+[layer]), state["metric"], True)
                    if row["metric"] is not None and row["metric"] < state["metric"]-args.eps:
                        if best is None or (row["metric"], layer) < (best[0], best[1]): best=(row["metric"],layer,row)
                if best is None: break
                state["metric"]=best[0]; selected.append(best[1]); state["selected"]=selected
                state["trace"].append({"layers":list(selected),"metric":best[0],"ppl":best[2]["ppl"]})
                atomic_json(state_path,state)
            trace=state["trace"]
    except EvaluationBudgetExceeded as err:
        status="truncated"; selected=(state.get("selected",[]) if args.direction=="forward" else [])
        trace=(state.get("trace",[]) if args.direction=="forward" else [])
        result_error=str(err)
    elapsed=time.monotonic()-slice_start
    prior=[]
    if summary_path.exists(): prior=json.loads(summary_path.read_text()).get("slice_seconds",[])
    summary={"status":status,"direction":args.direction,"rho":args.rho,"beta":args.beta,"eps":args.eps,
             "pool":pool_layers,"L_star":selected,"trace":trace,"counters":counters,
             "artifact_sha256":artifact_hash,"slice":args.slice,"settings":SETTINGS,
             "error":locals().get("result_error"),"slice_seconds":prior+[elapsed],
             "cumulative_search_seconds":sum(prior)+elapsed,
             "slurm_requested_seconds_per_slice":4800,
             "slurm_cumulative_requested_seconds":4800*len(prior+[elapsed]),
             "unique_candidate_requests":len(requested),
             "cache_scope":"per search arm; no cross-epsilon sharing"}
    atomic_json(summary_path,summary)
    # Final generation is a separate dependency stage so all jobs can be
    # submitted up front and so one endpoint's generation cannot consume a
    # search slice's wall budget.


def final_eval(args):
    """Generate the frozen 70x4 exploratory set for a completed endpoint."""
    import csv
    import torch
    from blade_refusal_amplify import scaled_weights
    from ttsafety.eval import teacher_forced_ppl
    from ttsafety.iti_composition import generate_batch, strict_selection, tensor_dict_hash
    ctx=common(args)
    run_dir=args.run_root / f"rho-{args.rho:g}_beta-{args.beta:g}_eps-{args.eps:g}" / args.direction
    search_summary_path=run_dir/"summary.json"
    rho, alpha = args.rho, (2.5 if args.rho == .005 else 2.)
    path=run_dir/f"final_rho-{rho:g}_alpha-{alpha:g}.json"
    if path.exists() and json.loads(path.read_text()).get("status") == "complete": return
    if not search_summary_path.exists():
        atomic_json(path,{"status":"skipped","reason":"search summary missing"}); return
    summary=json.loads(search_summary_path.read_text())
    if summary.get("status") != "complete":
        atomic_json(path,{"status":"skipped","reason":f"search status {summary.get('status')}",
                          "search_status":summary.get("status")}); return
    selected=summary["L_star"]
    if args.direction == "backward":
        matched_k_eval(args, ctx, run_dir, selected)
    items=json.loads(args.inputs.read_text())["items"]
    from collections import Counter
    cells=Counter((x["dataset"],x["gold"]) for x in items)
    expected={("falseqa","false_premise"):70,("falseqa","true_premise"):70,
              ("selfaware","unanswerable"):70,("selfaware","answerable"):70}
    if len(items)!=280 or cells != expected or len({x["id"] for x in items}) != 280:
        raise RuntimeError(f"invalid frozen OOD items: n={len(items)} cells={dict(cells)}")
    scores=ctx["score"](ctx["model"],ctx["directions"],ctx["mu_a"],ctx["mu_b"],selected,"both") if selected else {}
    # This is the fixed-ranking convention used by the existing rho/alpha
    # sweep: lambda is fit once and does not change with amplification alpha.
    selection=strict_selection(scores,rho) if selected else {}
    prompts=[x["question"] for x in items]
    with scaled_weights(ctx["model"],selection,alpha):
        outputs=[]
        for i in range(0,len(prompts),16): outputs.extend(generate_batch(ctx["model"],ctx["tok"],prompts[i:i+16],max_new_tokens=128))
        ppl=ctx["ppl"]()
    atomic_json(path,{"status":"complete","items":[{**x,**o} for x,o in zip(items,outputs)],
        "rho":rho,"alpha":alpha,"score_policy":"fixed_reference_lambda_across_alpha",
        "lambda_used":ctx["artifact"]["lambda"],
        "n_edges":sum(v.numel() for v in selection.values()),
        "mask_hash":tensor_dict_hash(selection) if selection else None,"ppl":ppl})


def matched_k_eval(args, ctx, run_dir, backward_layers):
    """Evaluate B removal with the exact rho_test edge count of matched F."""
    from ttsafety.iti_composition import strict_selection, tensor_dict_hash
    from ttsafety.weight_prune import pruned_weights

    path = run_dir / "matched_k.json"
    if path.exists() and json.loads(path.read_text()).get("status") == "complete":
        return
    f_summary_path = (args.run_root / f"rho-{args.rho:g}_beta-{args.beta:g}_eps-{args.eps:g}"
                      / "forward" / "summary.json")
    if not f_summary_path.exists():
        atomic_json(path, {"status":"unavailable", "reason":"matched F summary missing"})
        return
    f_summary = json.loads(f_summary_path.read_text())
    if f_summary.get("status") != "complete":
        atomic_json(path, {"status":"unavailable",
                          "reason":f"matched F status {f_summary.get('status')}",
                          "forward_status":f_summary.get("status")})
        return
    f_layers = f_summary["L_star"]
    if not f_layers:
        # F's empty endpoint has exactly zero edited edges.  It is a valid K=0
        # reference and equals the supplied base model, without a mask call.
        atomic_json(path, {"status":"complete", "K":0, "forward_layers":[],
                          "backward_layers":backward_layers,
                          "metric":ctx["artifact"]["base"]["metric"],
                          "ppl":ctx["artifact"]["base"]["ppl"],
                          "n_edges":0, "mask_hash":None})
        return
    try:
        f_scores = ctx["score"](ctx["model"],ctx["directions"],ctx["mu_a"],ctx["mu_b"],f_layers,"both")
        f_selection = strict_selection(f_scores, args.rho)
        k = sum(v.numel() for v in f_selection.values())
    except (ValueError, RuntimeError) as err:
        atomic_json(path,{"status":"infeasible","reason":"F rho_test mask is not strictly positive",
                          "detail":str(err),"forward_layers":f_layers})
        return
    del f_scores, f_selection
    if not backward_layers:
        atomic_json(path,{"status":"infeasible","reason":"B endpoint empty but F K is positive",
                          "K":k,"forward_layers":f_layers,"backward_layers":[]})
        return
    try:
        b_scores = ctx["score"](ctx["model"],ctx["directions"],ctx["mu_a"],ctx["mu_b"],backward_layers,"both")
        b_total = sum(v.numel() for v in b_scores.values())
        if k > round(.10 * b_total):
            raise ValueError("K exceeds the frozen 10% per-matrix candidate cap ceiling")
        b_selection = strict_selection(b_scores, k / b_total)
        if sum(v.numel() for v in b_selection.values()) != k:
            raise ValueError("ranking did not supply exactly K edges")
    except (ValueError, RuntimeError) as err:
        atomic_json(path,{"status":"infeasible","reason":"B cannot supply strict-positive K mask",
                          "detail":str(err),"K":k,"forward_layers":f_layers,
                          "backward_layers":backward_layers})
        return
    with pruned_weights(ctx["model"],b_selection):
        outputs, positives = ctx["generate"]()
        ppl = ctx["ppl"]()
    atomic_json(path,{"status":"complete","K":k,"forward_layers":f_layers,
                      "backward_layers":backward_layers,"metric":positives/len(ctx["prompts"]),
                      "positives":positives,"n_total":len(ctx["prompts"]),"ppl":ppl,
                      "outputs":outputs,"n_edges":sum(v.numel() for v in b_selection.values()),
                      "mask_hash":tensor_dict_hash(b_selection),"rho_equivalent":k/b_total,
                      "mask_policy":"global top-K, strict-positive, per-matrix cap .10"})


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=["fit","pool","search","final"])
    p.add_argument("--inputs",type=Path,required=True); p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--rho",type=float,choices=[.005,.01]); p.add_argument("--beta",type=float,choices=[.05,.10]); p.add_argument("--eps",type=float,choices=[.005,.025])
    p.add_argument("--direction",choices=["forward","backward"]); p.add_argument("--slice",type=int,default=0)
    p.add_argument("--slice-minutes",type=float,default=78); return p.parse_args()


if __name__ == "__main__":
    a=parse_args()
    if a.command=="fit": fit(a)
    elif a.command=="pool":
        if a.beta is None or a.rho is None: raise SystemExit("--rho and --beta required")
        pool(a)
    elif a.command=="search":
        if None in (a.rho,a.beta,a.eps,a.direction): raise SystemExit("--rho, --beta, --eps, --direction required")
        search(a)
    else:
        if None in (a.rho,a.beta,a.eps,a.direction): raise SystemExit("--rho, --beta, --eps, --direction required")
        final_eval(a)
