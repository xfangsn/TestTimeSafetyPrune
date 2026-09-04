"""Re-score existing calibration results with the fixed verifier (no re-generation): recompute
correctness per item from saved pred/gold, then recompute Brier/ECE/AUROC/risk-coverage +
accuracy, and print base-vs-remove-vs-controls. Usage: python scripts/rescore_calibration.py 14b math"""
import json
import sys
from pathlib import Path

from eval_calibration import correct, brier, ece, auroc, risk_coverage_auc

TAG = sys.argv[1] if len(sys.argv) > 1 else "14b"
BENCH = sys.argv[2] if len(sys.argv) > 2 else "math"
R = Path("results")
MODES = ["base", "remove", "random", "shuffle"]

print(f"Qwen3-{TAG} {BENCH} uncertainty calibration — RESCORED (fixed verifier)")
print("%-9s %6s %7s %7s %7s %7s %6s" % ("mode", "acc", "Brier", "ECE", "AUROC", "RC-AUC", "n"))
for m in MODES:
    f = R / f"calib_{TAG}_{BENCH}_{m}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    items = d["items"]
    corr = [correct(it["pred"], it["gold"], BENCH) for it in items]
    conf = [it["vconf"] for it in items]
    acc = sum(corr) / len(corr)
    print("%-9s %6.3f %7.3f %7.3f %7.3f %7.3f %6d"
          % (m, acc, brier(conf, corr), ece(conf, corr), auroc(conf, corr),
             risk_coverage_auc(conf, corr), len(items)))
    # write back the corrected scores
    for it, c in zip(items, corr):
        it["correct"] = c
    d["accuracy"] = acc
    d["verbalized"] = {"brier": brier(conf, corr), "ece": ece(conf, corr),
                       "auroc": auroc(conf, corr), "risk_cov_auc": risk_coverage_auc(conf, corr)}
    d["rescored"] = True
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1))
