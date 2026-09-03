"""Score Wei et al. (2026, arXiv:2604.09544) signed safety SNIP weights.

Signed variant of the Wei et al. (2024) absolute SNIP in score_wei_snip.py:
I(W_ij) = mean_x [ W_ij * dL(x)/dW_ij ] with L = response-token NLL
(prompt masked, EOT included) — critically WITHOUT the absolute value
(their Eq. 1-2). Weights with NEGATIVE signed score facilitate generation
of the safety-behavior target; for this refusal-targeting adaptation the
target is refusal-response NLL on data/caa_pairs.jsonl
(response_field="refusal", 247 pairs), so pruning the most-negative scores
should reduce refusal.

Only the safety split is scored here; the utility exclusion for the sweep
reuses the cached absolute utility SNIP (data/weight_scores/wei_utility_snip.pt),
which is the correct form for that purpose.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.models import env_info, load_model
from ttsafety.weight_edit import iter_residual_writers
from score_refusal_weights import response_batch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "weight_scores"
LAYERS = list(range(7, 19))
COMPONENTS = "both"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safety_rows() -> tuple[list[tuple[str, str]], dict]:
    path = DATA / "caa_pairs.jsonl"
    rows = [
        (item["instruction"], item["refusal"])
        for item in load_jsonl(path)
    ]
    return rows, {
        "source": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "response_field": "refusal",
    }


def atomic_torch_save(value, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def score_safety_signed(model, tokenizer, max_samples: int | None) -> None:
    rows, provenance = safety_rows()
    if max_samples is not None:
        rows = rows[:max_samples]
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    model.requires_grad_(False)
    for module in writers.values():
        module.weight.requires_grad_(True)
    accumulators = {
        name: torch.zeros_like(module.weight, dtype=torch.float32)
        for name, module in writers.items()
    }

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        model.zero_grad(set_to_none=True)
        # HF-style response-token mean cross entropy, including the EOT token.
        nll = -response_batch(model, tokenizer, [row]).squeeze(0)
        nll.backward()
        with torch.no_grad():
            for name, module in writers.items():
                if module.weight.grad is None:
                    raise RuntimeError(f"missing gradient for {name}")
                accumulators[name].add_(
                    module.weight.detach().float()
                    * module.weight.grad.detach().float()
                )  # signed: NO .abs_() (Wei et al. 2026, Eq. 1-2)
        if index % 10 == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            print(
                f"Wei signed SNIP safety {index}/{len(rows)} "
                f"({elapsed / index:.2f}s/example)",
                flush=True,
            )

    elapsed = time.monotonic() - started
    scores = {
        name: (accumulator / len(rows)).cpu()
        for name, accumulator in accumulators.items()
    }
    metadata = {
        "method": "wei2026_signed_snip",
        "split": "safety",
        "formula": "mean_x(W * grad_W(mean_response_token_nll(x)))  [signed]",
        "aggregation": "mean_of_per_example_signed_scores",
        "loss": "mean response-token NLL including EOT; prompt tokens masked",
        "arxiv": "2604.09544",
        "n_examples": len(rows),
        "max_samples": max_samples,
        "layers": LAYERS,
        "components": COMPONENTS,
        "target_pool": "residual writers only (controlled adaptation)",
        "data": provenance,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "env": env_info(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_samples is None else f"_smoke{max_samples}"
    path = OUT_DIR / f"wei_safety_signed_snip{suffix}.pt"
    atomic_torch_save({"scores": scores, "metadata": metadata}, path)

    # sanity: signed vs cached absolute version + sign distribution
    abs_path = OUT_DIR / "wei_safety_snip.pt"
    sign_summary = {}
    abs_check = None
    if abs_path.exists() and max_samples is None:
        abs_scores = torch.load(
            abs_path, map_location="cpu", weights_only=False
        )["scores"]
        diffs = []
        for name, score in scores.items():
            a = abs_scores[name].float()
            diffs.append(float((score.abs() - a).abs().max()))
        abs_check = {"max_abs_diff_vs_abs_version": max(diffs)}
    for name, score in scores.items():
        sign_summary[name] = {
            "shape": list(score.shape),
            "negative_fraction": float((score < 0).float().mean()),
            "mean": float(score.mean()),
            "min": float(score.min()),
            "max": float(score.max()),
        }
    neg_fracs = [v["negative_fraction"] for v in sign_summary.values()]
    print(f"negative-score fraction: min {min(neg_fracs):.4f} "
          f"max {max(neg_fracs):.4f}", flush=True)
    if abs_check:
        print(f"signed vs abs version: {abs_check}", flush=True)
    summary = {
        "path": str(path.relative_to(ROOT)),
        "metadata": metadata,
        "sanity": abs_check,
        "matrices": sign_summary,
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}", flush=True)
    model.zero_grad(set_to_none=True)
    del scores, accumulators
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples", type=int,
        help="debug-only prefix length; omit for the reported experiment",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    model, tokenizer = load_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    score_safety_signed(model, tokenizer, args.max_samples)


if __name__ == "__main__":
    main()
