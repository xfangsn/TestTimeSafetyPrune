"""Budget-sensitivity of the finding: at each method's frozen structure, the best OOD refusal
whose (benign over-refusal, wiki Δppl) fall within a range of budgets. Shows whether "BLADE
amplify gives no eligible strengthening" is an artifact of the exact +5pp benign budget."""
import json
from pathlib import Path

R = Path("results")
D = json.loads((R / "blade_steering_baselines.json").read_text())
B = json.loads((R / "blade_amplify_ood200.json").read_text())
base_ood = D["base"]["ood_refusal"]
base_benign = D["base"]["benign"] * 100
PPL_BUDGET = D["config"]["BETA"] * 100  # keep ppl budget fixed at 5%

methods = {k: v["report"] for k, v in D["methods"].items()}
methods["BLADE amplify"] = [r for r in B["report"] if r.get("factor", 2) > 1.0]

print(f"base OOD refusal {base_ood:.3f} | base benign {base_benign:.1f}% | ppl budget {PPL_BUDGET:g}%\n")
print(f"{'benign budget':>14} | " + " | ".join(f"{m:>14}" for m in methods))
for pp in [2, 5, 10, 15, 20, 100]:
    limit = base_benign + pp
    cells = []
    for m, rows in methods.items():
        elig = [r for r in rows if r["benign_full"] * 100 <= limit and r["wiki_relppl"] * 100 <= PPL_BUDGET]
        best = max((r["ood_refusal"] for r in elig), default=base_ood)
        cells.append(f"{best:>14.3f}")
    print(f"  base+{pp:>3}pp    | " + " | ".join(cells))
print("\n(each cell = best OOD refusal achievable at that method's frozen structure within the "
      "benign budget AND <=5% ppl; base means no eligible strengthening point)")
