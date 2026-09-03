"""Score Wei et al. (2024) safety/utility weights with absolute SNIP.

The paper defines I(W, x) = |W * grad_W L(x)| and averages this quantity
over examples.  The absolute value is therefore taken before the dataset
average; this is intentionally different from the signed Taylor baseline in
``score_refusal_weights.py``.
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


def utility_rows() -> tuple[list[tuple[str, str]], dict]:
    prompts_path = DATA / "harmless.jsonl"
    responses_path = DATA / "cache" / "weight_edit_harmless_base.json"
    prompts = load_jsonl(prompts_path)
    responses = json.loads(responses_path.read_text())
    if len(prompts) != len(responses):
        raise ValueError("harmless prompts and cached responses do not align")
    rows = [
        (item["instruction"], response)
        for item, response in zip(prompts, responses)
    ]
    return rows, {
        "prompt_source": str(prompts_path.relative_to(ROOT)),
        "prompt_sha256": file_sha256(prompts_path),
        "response_source": str(responses_path.relative_to(ROOT)),
        "response_sha256": file_sha256(responses_path),
        "responses": "deterministic cached completions from the unedited model",
    }


def atomic_torch_save(value, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def score_split(model, tokenizer, split: str, max_samples: int | None) -> None:
    rows, provenance = safety_rows() if split == "safety" else utility_rows()
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
                    (module.weight.detach().float()
                     * module.weight.grad.detach().float()).abs_()
                )
        if index % 10 == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            print(
                f"Wei SNIP {split} {index}/{len(rows)} "
                f"({elapsed / index:.2f}s/example)",
                flush=True,
            )

    elapsed = time.monotonic() - started
    scores = {
        name: (accumulator / len(rows)).cpu()
        for name, accumulator in accumulators.items()
    }
    metadata = {
        "method": "wei2024_absolute_snip",
        "split": split,
        "formula": "mean_x(abs(W * grad_W(mean_response_token_nll(x))))",
        "aggregation": "mean_of_per_example_absolute_scores",
        "loss": "mean response-token NLL including EOT; prompt tokens masked",
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
    path = OUT_DIR / f"wei_{split}_snip{suffix}.pt"
    atomic_torch_save({"scores": scores, "metadata": metadata}, path)
    summary = {
        "path": str(path.relative_to(ROOT)),
        "metadata": metadata,
        "matrices": {
            name: {
                "shape": list(score.shape),
                "nonzero_fraction": float((score > 0).float().mean()),
                "mean": float(score.mean()),
                "max": float(score.max()),
            }
            for name, score in scores.items()
        },
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}", flush=True)
    model.zero_grad(set_to_none=True)
    del scores, accumulators
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("safety", "utility", "both"), default="both"
    )
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
    splits = ("safety", "utility") if args.split == "both" else (args.split,)
    for split in splits:
        score_split(model, tokenizer, split, args.max_samples)


if __name__ == "__main__":
    main()
