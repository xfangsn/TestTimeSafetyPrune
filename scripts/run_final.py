"""M3: final held-out test evaluation at the selected (layer, alpha)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.generate import generate_texts
from ttsafety.judge import refusal_rate
from ttsafety.models import env_info, load_model
from ttsafety.steer import steer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_TAG = "llama32_3b_instruct"
MAX_NEW_TOKENS = 128
N_SAMPLES = 10


def main():
    sweep = json.loads((RESULTS_DIR / "sweep_steer.json").read_text())
    sel = sweep.get("selection")
    if sel is None:
        raise SystemExit("no selection in sweep_steer.json — run "
                         "`uv run python scripts/sweep_steer.py --finalize` first")
    layer, alpha = sel["layer"], sel["alpha"]
    print(f"Selected cell: layer {layer}, alpha {alpha} "
          f"(flagged={sel['constraint_flag']})")

    model, tokenizer = load_model()
    directions = torch.load(
        DATA_DIR / "directions" / f"refusal_{MODEL_TAG}.pt", weights_only=True
    )
    harmful_test = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmful_test.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmless.jsonl")]
    wiki_text = load_wikitext_text()

    print(f"Generating baseline on {len(harmful_test)} harmful_test ...")
    base_out = generate_texts(model, tokenizer, harmful_test, MAX_NEW_TOKENS)
    print("Generating steered ...")
    with steer(model, directions[layer], layer=layer, alpha=alpha):
        steered_out = generate_texts(model, tokenizer, harmful_test, MAX_NEW_TOKENS)
        harmless_refusal = refusal_rate(
            generate_texts(model, tokenizer, harmless, MAX_NEW_TOKENS))
        ppl_steered = teacher_forced_ppl(model, tokenizer, wiki_text)
    ppl_base = teacher_forced_ppl(model, tokenizer, wiki_text)

    base_refusal = refusal_rate(base_out)
    steered_refusal = refusal_rate(steered_out)
    report = {
        "config": {
            "model": MODEL_TAG,
            "selected": {"layer": layer, "alpha": alpha,
                         "constraint_flag": sel["constraint_flag"]},
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_harmful_test": len(harmful_test),
        },
        "env": env_info(),
        "metrics": {
            "test_refusal_baseline": base_refusal,
            "test_refusal_steered": steered_refusal,
            "test_compliance_baseline": 1 - base_refusal,
            "test_compliance_steered": 1 - steered_refusal,
            "harmless_refusal_steered": harmless_refusal,
            "wikitext_ppl_baseline": ppl_base,
            "wikitext_ppl_steered": ppl_steered,
            "ppl_delta_pct": (ppl_steered - ppl_base) / ppl_base * 100,
        },
        "samples": [
            {"instruction": s, "baseline": b, "steered": t}
            for s, b, t in zip(harmful_test[:N_SAMPLES],
                               base_out[:N_SAMPLES], steered_out[:N_SAMPLES])
        ],
    }
    out = RESULTS_DIR / "final_test.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {out}")

    samples_path = DATA_DIR / "samples_final.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for s, b, t in zip(harmful_test, base_out, steered_out):
            f.write(json.dumps({"instruction": s, "baseline": b, "steered": t},
                               ensure_ascii=False) + "\n")
    print(f"All generations saved to {samples_path}")

    m = report["metrics"]
    print(f"test compliance: {m['test_compliance_baseline']:.3f} -> "
          f"{m['test_compliance_steered']:.3f}")
    print(f"harmless refusal (steered): {harmless_refusal:.3f}")
    print(f"wikitext ppl: {ppl_base:.2f} -> {ppl_steered:.2f} "
          f"({m['ppl_delta_pct']:+.2f}%)")

    # compact bar chart for the report
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = ["harmful_test refusal", "harmless refusal"]
    base_vals = [base_refusal, sweep["baseline"]["harmless_refusal"]]
    steer_vals = [steered_refusal, harmless_refusal]
    w = 0.35
    ax.bar([i - w / 2 for i in range(2)], base_vals, w, label="baseline")
    ax.bar([i + w / 2 for i in range(2)], steer_vals, w,
           label=f"steered L{layer} a={alpha}")
    ax.set_xticks(range(2))
    ax.set_xticklabels(xs)
    ax.set_ylabel("refusal rate")
    ax.set_title("Final test-set result")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_curves.png", dpi=150)
    print(f"Bar chart saved to {RESULTS_DIR / 'final_curves.png'}")


if __name__ == "__main__":
    main()
