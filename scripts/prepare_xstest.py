"""Download XSTest v2 (over-refusal benchmark) and dump to data/.

Held-out benign benchmark for measuring OVER-refusal (fraction of SAFE prompts wrongly
refused). XSTest v2 = 250 safe prompts + 200 unsafe contrasts; the over-refusal denominator
is the 250 safe prompts only (Rottger et al., NAACL 2024, arXiv:2308.01263).

Source: natolambert/xstest-v2-copy (HF), split "gpt4". Safe = type not starting with
"contrast_". We pin the dataset commit revision and record it in the meta file.
"""
import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
REPO = "natolambert/xstest-v2-copy"
REVISION = "main"  # recorded resolved commit hash in meta below
SAFE_OUT = ROOT / "data" / "xstest_safe.jsonl"
UNSAFE_OUT = ROOT / "data" / "xstest_unsafe.jsonl"
META = ROOT / "data" / "xstest.meta.json"


def main() -> None:
    from huggingface_hub import HfApi
    resolved_sha = HfApi().dataset_info(REPO, revision=REVISION).sha
    ds = load_dataset(REPO, split="gpt4", revision=resolved_sha)
    safe = [r for r in ds if not r["type"].startswith("contrast_")]
    unsafe = [r for r in ds if r["type"].startswith("contrast_")]
    assert len(safe) == 250 and len(unsafe) == 200, (len(safe), len(unsafe))

    def dump(path, rows):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "type": r["type"],
                                    "instruction": r["prompt"]}, ensure_ascii=False) + "\n")

    dump(SAFE_OUT, safe)
    dump(UNSAFE_OUT, unsafe)
    # content hash of the safe prompts for provenance
    safe_hash = hashlib.sha256(
        "\n".join(r["prompt"] for r in safe).encode("utf-8")).hexdigest()
    meta = {
        "repo": REPO, "split": "gpt4", "revision": REVISION, "resolved_commit": resolved_sha,
        "n_safe": len(safe), "n_unsafe": len(unsafe),
        "over_refusal_denominator": 250,
        "safe_prompts_sha256": safe_hash,
        "dataset_fingerprint": ds._fingerprint,
        "note": "over-refusal = refused / 250 safe prompts; unsafe used only if a contrast is needed",
    }
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
