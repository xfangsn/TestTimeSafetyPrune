"""Calibrate is_over_refusal against XSTest gpt4 gold labels.

XSTest rows carry a human `final_label` (1_full_compliance / 2_full_refusal /
3_partial_refusal ...) on a reference gpt4 `completion`. We treat labels starting with
"1" as compliance (gold negative) and "2"/"3" as refusal (gold positive), then report the
judge's false-positive rate on SAFE compliant answers (the number that actually matters for
an over-refusal metric) plus overall precision/recall.
"""
import json
from pathlib import Path

from datasets import load_dataset

from ttsafety.judge import is_over_refusal, is_refusal


def gold_is_refusal(label: str) -> bool:
    return label.startswith("2") or label.startswith("3")


def main() -> None:
    ds = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
    safe = [r for r in ds if not r["type"].startswith("contrast_")]
    for name, judge in [("is_over_refusal", is_over_refusal), ("is_refusal (generic)", is_refusal)]:
        tp = fp = tn = fn = 0
        for r in safe:
            pred = judge(r["completion"])
            gold = gold_is_refusal(r["final_label"])
            tp += pred and gold
            fp += pred and not gold
            tn += (not pred) and (not gold)
            fn += (not pred) and gold
        n_comp = sum(not gold_is_refusal(r["final_label"]) for r in safe)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{name:22} | safe FP rate {fp}/{n_comp}={fp/n_comp:.1%} | "
              f"precision {prec:.2f} recall {rec:.2f} | TP{tp} FP{fp} TN{tn} FN{fn}")


if __name__ == "__main__":
    main()
