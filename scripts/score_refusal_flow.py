"""C0-C2: prepare, collect, and rank gradient-free CRFP weight scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.models import env_info, load_model
from ttsafety.refusal_flow import (
    assert_gradient_free,
    collect_harmless_writer_moments,
    collect_paired_writer_moments,
    crfp_matrix_score,
)
from ttsafety.weight_edit import iter_residual_writers
from ttsafety.weight_prune import rank_weight_indices

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCORES = DATA / "weight_scores"
SPLITS = DATA / "splits"
PAIR_PATH = DATA / "caa_pairs.jsonl"
SPLIT_PATH = SPLITS / "crfp_caa_split_seed0.json"
PAIRED_PATH = SCORES / "crfp_activation_stats.pt"
HARMLESS_PATH = SCORES / "crfp_harmless_moments.pt"
SCORE_PATH = SCORES / "crfp.pt"
RANKING_PATH = SCORES / "ranking_crfp.pt"
DIRECTION_PATH = DATA / "directions" / "refusal_llama32_3b_instruct.pt"
LAYERS = list(range(7, 19))
COMPONENTS = "both"
BETA = 1.0
ELIGIBILITY_FRACTION = 0.10
MAX_FRACTION = 0.01


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def pair_digest(pair: dict) -> str:
    payload = json.dumps(pair, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_split() -> dict:
    pairs = load_jsonl(PAIR_PATH)
    ordered = sorted((pair_digest(pair), index) for index, pair in enumerate(pairs))
    cut = int(0.80 * len(ordered))
    value = {
        "method": "sha256_sorted_80_20",
        "source": str(PAIR_PATH.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(PAIR_PATH.read_bytes()).hexdigest(),
        "n_total": len(pairs),
        "score": [digest for digest, _ in ordered[:cut]],
        "calibration": [digest for digest, _ in ordered[cut:]],
    }
    if SPLIT_PATH.exists():
        old = json.loads(SPLIT_PATH.read_text())
        if old != value:
            raise RuntimeError("existing CRFP split does not match current CAA pairs")
    else:
        atomic_json(SPLIT_PATH, value)
    print(json.dumps({
        "split": str(SPLIT_PATH.relative_to(ROOT)),
        "score": len(value["score"]),
        "calibration": len(value["calibration"]),
    }, indent=2))
    return value


def split_pairs(role: str) -> list[dict]:
    manifest = prepare_split()
    allowed = set(manifest[role])
    pairs = [pair for pair in load_jsonl(PAIR_PATH) if pair_digest(pair) in allowed]
    if len(pairs) != len(allowed):
        raise RuntimeError(f"could not resolve all {role} pair hashes")
    return pairs


def load_gradient_free_model():
    model, tokenizer = load_model()
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    assert_gradient_free(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def provenance(started: float) -> dict:
    out = {
        "env": env_info(),
        "wall_seconds": time.perf_counter() - started,
        "gradient_free_verified": True,
    }
    if torch.cuda.is_available():
        out.update({
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
    return out


def run_paired(model, tokenizer) -> None:
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pairs = split_pairs("score")
    moments = collect_paired_writer_moments(
        model, tokenizer, pairs, LAYERS, COMPONENTS, batch_pairs=4
    )
    metadata = {
        "score": "paired_response_prediction_writer_input_moments",
        "n_pairs": len(pairs),
        "layers": LAYERS,
        "components": COMPONENTS,
        "include_eot": True,
        "causal_shift": True,
        **provenance(started),
    }
    SCORES.mkdir(parents=True, exist_ok=True)
    torch.save({"moments": moments, "metadata": metadata}, PAIRED_PATH)
    print(f"saved {PAIRED_PATH.relative_to(ROOT)}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


def run_harmless(model, tokenizer) -> None:
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    instructions = [
        row["instruction"] for row in load_jsonl(DATA / "harmless.jsonl")
    ]
    moments = collect_harmless_writer_moments(
        model, tokenizer, instructions, LAYERS, COMPONENTS, batch_size=8
    )
    metadata = {
        "score": "harmless_writer_input_second_moment",
        "n_instructions": len(instructions),
        "layers": LAYERS,
        "components": COMPONENTS,
        **provenance(started),
    }
    SCORES.mkdir(parents=True, exist_ok=True)
    torch.save({"moments": moments, "metadata": metadata}, HARMLESS_PATH)
    print(f"saved {HARMLESS_PATH.relative_to(ROOT)}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


def run_rank(model) -> None:
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    paired_payload = torch.load(PAIRED_PATH, map_location="cpu", weights_only=False)
    harmless_payload = torch.load(
        HARMLESS_PATH, map_location="cpu", weights_only=False
    )
    directions = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=True)
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    scores = {}
    diagnostics = {}
    for name, module in writers.items():
        layer = int(name.split(".")[1])
        print(f"CRFP scoring {name} ...", flush=True)
        score, info = crfp_matrix_score(
            module.weight,
            directions[layer],
            paired_payload["moments"][name],
            harmless_payload["moments"][name],
            beta=BETA,
            eligibility_fraction=ELIGIBILITY_FRACTION,
        )
        scores[name] = score
        diagnostics[name] = info
    assert_gradient_free(model)
    metadata = {
        "score": "crfp_lcb_tempered_wanda",
        "beta": BETA,
        "alpha": 0.5,
        "tau": "per_matrix_median_harmless_cost",
        "eligibility_fraction_of_positive_benefit": ELIGIBILITY_FRACTION,
        "layers": LAYERS,
        "components": COMPONENTS,
        "direction": str(DIRECTION_PATH.relative_to(ROOT)),
        "direction_sign": "as_stored_harmful_minus_harmless",
        "paired_metadata": paired_payload["metadata"],
        "harmless_metadata": harmless_payload["metadata"],
        **provenance(started),
    }
    torch.save(
        {"scores": scores, "diagnostics": diagnostics, "metadata": metadata},
        SCORE_PATH,
    )
    atomic_json(
        SCORE_PATH.with_suffix(".json"),
        {"metadata": metadata, "matrices": diagnostics},
    )
    print("building CRFP global ranking ...", flush=True)
    ranking = rank_weight_indices(
        scores,
        MAX_FRACTION,
        largest=True,
        per_matrix_cap=0.10,
    )
    ranking["score_name"] = "crfp"
    ranking["gradient_free_verified"] = True
    torch.save(ranking, RANKING_PATH)
    print(f"saved {SCORE_PATH.relative_to(ROOT)}", flush=True)
    print(f"saved {RANKING_PATH.relative_to(ROOT)}", flush=True)
    print(json.dumps({
        "total_pool_weights": ranking["total_pool_weights"],
        "ranking_entries": int(ranking["flat_indices"].numel()),
        "metadata": metadata,
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "paired", "harmless", "rank", "all")
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_split()
        return
    prepare_split()
    model, tokenizer = load_gradient_free_model()
    if args.stage in ("paired", "all"):
        run_paired(model, tokenizer)
    if args.stage in ("harmless", "all"):
        run_harmless(model, tokenizer)
    if args.stage in ("rank", "all"):
        if not PAIRED_PATH.exists() or not HARMLESS_PATH.exists():
            raise SystemExit("paired and harmless moment files are required before rank")
        run_rank(model)


if __name__ == "__main__":
    main()
