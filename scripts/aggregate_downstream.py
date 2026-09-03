"""W3: aggregate downstream results -> JSON + figure (CPU only).

Reads results/downstream/{config}.json partials plus refusal numbers from
results/sweep_weight_prune.json and results/sweep_wei_snip_set_difference.json,
and writes results/downstream_comparison.json + results/downstream_comparison.png
(mean acc vs n_pruned for edge/wei/random0; refusal vs mean-acc trade-off).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOWN = RESULTS / "downstream"
TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "boolq")

EDGE = ["edge_s0.0001", "edge_s0.0005", "edge_s0.001"]
WEI = ["wei_p0.0001_q0.0001", "wei_p0.0005_q0.0005", "wei_p0.001_q0.001"]
WEI_EQUAL_REFUSAL = "wei_p0.01_q0.01"
RANDOM = ["random0_s0.0001", "random0_s0.0005", "random0_s0.001"]
EXTRA = ["ratio_s0.0001"]

SWEEP_EDGE = RESULTS / "sweep_weight_prune.json"
SWEEP_WEI = RESULTS / "sweep_wei_snip_set_difference.json"


def refusal_map():
    out = {}
    edge = json.loads(SWEEP_EDGE.read_text())["cells"]
    wei = json.loads(SWEEP_WEI.read_text())["cells"]
    for k, c in edge.items():
        if c.get("status") == "complete":
            out[k] = {"harmful_val_refusal": c["harmful_refusal"],
                      "n_pruned": c["n_pruned"]}
    for k, c in wei.items():
        if c.get("status") == "complete":
            out[k] = {"harmful_val_refusal": c["harmful_refusal"],
                      "n_pruned": c["n_pruned"]}
    return out


def main():
    refs = refusal_map()
    configs = ["base"] + EDGE + WEI + [WEI_EQUAL_REFUSAL] + RANDOM + EXTRA
    table = {}
    for name in configs:
        p = DOWN / f"{name}.json"
        if not p.exists():
            print(f"WARNING: missing {p}")
            continue
        r = json.loads(p.read_text())
        entry = {
            "n_pruned": r["n_pruned"],
            "mean_acc": r["mean_acc"],
            "mean_acc_norm": r["mean_acc_norm"],
            "wikitext_ppl_10k": r["wikitext_ppl_10k"],
            "ppl_delta_pct_10k": r["ppl_delta_pct_10k"],
            "tasks": {t: {"acc": r["tasks"][t]["acc"],
                          "acc_norm": r["tasks"][t]["acc_norm"],
                          "n": r["tasks"][t]["n"]} for t in TASKS},
        }
        if name in refs:
            entry["harmful_val_refusal"] = refs[name]["harmful_val_refusal"]
            assert refs[name]["n_pruned"] == r["n_pruned"], (
                f"n_pruned mismatch for {name}: sweep "
                f"{refs[name]['n_pruned']} vs downstream {r['n_pruned']}")
        table[name] = entry

    # matched-n verdicts (>1pp mean-acc difference = meaningful, plan section 5)
    def diff(a, b):
        return table[a]["mean_acc"] - table[b]["mean_acc"]

    matched = []
    for e, w, rnd in zip(EDGE, WEI, RANDOM):
        matched.append({
            "tier": e.split("_s")[1],
            "edge": e, "wei": w, "random": rnd,
            "edge_mean_acc": table[e]["mean_acc"],
            "wei_mean_acc": table[w]["mean_acc"],
            "random_mean_acc": table[rnd]["mean_acc"],
            "edge_minus_wei_pp": diff(e, w) * 100,
            "edge_mean_acc_norm": table[e]["mean_acc_norm"],
            "wei_mean_acc_norm": table[w]["mean_acc_norm"],
        })
    equal_refusal = {
        "edge": "edge_s0.0005", "wei": WEI_EQUAL_REFUSAL,
        "edge_refusal": table["edge_s0.0005"].get("harmful_val_refusal"),
        "wei_refusal": table[WEI_EQUAL_REFUSAL].get("harmful_val_refusal"),
        "edge_mean_acc": table["edge_s0.0005"]["mean_acc"],
        "wei_mean_acc": table[WEI_EQUAL_REFUSAL]["mean_acc"],
        "edge_n_pruned": table["edge_s0.0005"]["n_pruned"],
        "wei_n_pruned": table[WEI_EQUAL_REFUSAL]["n_pruned"],
        "edge_minus_wei_pp": diff("edge_s0.0005", WEI_EQUAL_REFUSAL) * 100,
    }
    n_tiers_edge_better = sum(1 for m in matched
                              if m["edge_minus_wei_pp"] > 1.0)
    verdict = {
        "matched_tiers": matched,
        "equal_refusal_reduction": equal_refusal,
        "edge_better_by_gt1pp_in_n_tiers": n_tiers_edge_better,
        "claim_superior": n_tiers_edge_better == 3,
    }
    out = {"configs": table, "analysis": verdict,
           "notes": [
               "n_pruned matched by nearest neighbor (wei set-difference size "
               "is determined by p,q); refusal numbers from the sweep JSONs "
               "(harmful_val, n=64)",
               "base wikitext ppl 10k = "
               f"{table['base']['wikitext_ppl_10k']:.4f}",
           ]}
    (RESULTS / "downstream_comparison.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(verdict, indent=2))

    # figure: mean acc vs n_pruned + refusal vs mean acc
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    series = [("edge", EDGE, "o"), ("wei", WEI + [WEI_EQUAL_REFUSAL], "s"),
              ("random0", RANDOM, "^")]
    for label, names, marker in series:
        xs = [table[n]["n_pruned"] for n in names]
        axes[0].plot(xs, [table[n]["mean_acc"] for n in names],
                     marker=marker, label=label)
        axes[1].plot([table[n]["mean_acc"] for n in names],
                     [refs[n]["harmful_val_refusal"] for n in names],
                     marker=marker, label=label)
    base_acc = table["base"]["mean_acc"]
    axes[0].axhline(base_acc, ls=":", color="gray", label="base (unpruned)")
    axes[1].axhline(base_acc, ls=":", color="gray")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n_pruned (log scale)")
    axes[0].set_ylabel("mean acc (6 tasks)")
    axes[0].set_title("downstream utility vs pruning amount")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("mean acc (6 tasks)")
    axes[1].set_ylabel("harmful_val refusal (lower = more safety broken)")
    axes[1].set_title("refusal vs utility trade-off")
    axes[1].legend(fontsize=8)
    fig.suptitle("Downstream comparison: signed actdiff edge vs Wei SNIP "
                 "set-difference")
    fig.tight_layout()
    fig.savefig(RESULTS / "downstream_comparison.png", dpi=150)
    print(f"saved {RESULTS / 'downstream_comparison.json'} + .png")


if __name__ == "__main__":
    main()
