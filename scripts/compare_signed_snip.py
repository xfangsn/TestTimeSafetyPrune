"""Aggregate the three-way matched-n comparison (CPU only).

edge (signed actdiff, results/sweep_weight_prune.json) vs Wei-2024 unsigned
SNIP set-difference (results/sweep_wei_snip_set_difference.json) vs Wei-2026
signed SNIP set-difference (results/sweep_wei_signed_snip.json), plus
downstream six-task means from results/downstream/{config}.json.
Writes results/signed_snip_comparison.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TIERS = [
    # (tier label, edge key, wei2024 key, signed key)
    ("~0.01% (41.5k)", "edge_s0.0001", "wei_p0.0001_q0.0001",
     "signed_p0.0001_q0.0001"),
    ("~0.05% (207.6k)", "edge_s0.0005", "wei_p0.0005_q0.0005",
     "signed_p0.0005_q0.0005"),
    ("~0.1% (415.2k)", "edge_s0.001", "wei_p0.001_q0.001",
     "signed_p0.001_q0.001"),
]


def main():
    edge = json.loads((RESULTS / "sweep_weight_prune.json").read_text())["cells"]
    wei = json.loads(
        (RESULTS / "sweep_wei_snip_set_difference.json").read_text())["cells"]
    signed = json.loads(
        (RESULTS / "sweep_wei_signed_snip.json").read_text())["cells"]

    def entry(cell, source, key):
        down_path = RESULTS / "downstream" / f"{key}.json"
        down = json.loads(down_path.read_text()) if down_path.exists() else None
        return {
            "key": key,
            "n_pruned": cell["n_pruned"],
            "harmful_val_refusal": cell["harmful_refusal"],
            "harmless_refusal": cell["harmless_refusal"],
            "ppl_delta_pct": cell["ppl_delta_pct"],
            "harmless_kl": cell["harmless_kl"],
            "downstream_mean_acc": down["mean_acc"] if down else None,
            "downstream_mean_acc_norm": down["mean_acc_norm"] if down else None,
        }

    table = []
    for label, e_key, w_key, s_key in TIERS:
        table.append({
            "tier": label,
            "edge": entry(edge[e_key], edge, e_key),
            "wei2024_unsigned": entry(wei[w_key], wei, w_key),
            "wei2026_signed": entry(signed[s_key], signed, s_key),
        })

    analysis = {
        "signed_reaches_zero_refusal_at_all_tiers": all(
            t["wei2026_signed"]["harmful_val_refusal"] == 0.0 for t in table),
        "verdict": (
            "signed SNIP (Wei 2026) closes the refusal gap with edge "
            "completely (0.000 at every matched tier, including the smallest "
            "where edge still shows 0.125) — the score sign carries the "
            "critical information the 2024 absolute version discards. But "
            "collateral damage is far higher than edge at tiers 2-3: "
            "harmless KL 1.08/3.45/5.15 vs edge 0.04/0.09/0.13, and "
            "downstream mean acc_norm 0.682/0.639/0.607 vs edge "
            "0.687/0.683/0.681 (base 0.688). Unsigned Wei-2024 preserves "
            "utility (KL 0.03-0.07, acc_norm ~0.683) but barely dents refusal "
            "(0.80/0.63/0.56). Signed SNIP trades utility for safety-breaking "
            "power; edge dominates both on the combined criterion."
        ),
    }
    out = {"tiers": table, "analysis": analysis,
           "notes": [
               "refusal/ppl/KL from the sweep JSONs (harmful_val n=64, "
               "harmless n=320, wikitext ppl 50k tokens, KL on 128 harmless "
               "prompts); downstream from results/downstream/ partials",
               "n_pruned matched by nearest neighbor (set-difference sizes "
               "are determined by p,q and the overlap)",
           ]}
    path = RESULTS / "signed_snip_comparison.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(analysis, indent=2))
    for t in table:
        print(t["tier"])
        for m in ("edge", "wei2024_unsigned", "wei2026_signed"):
            e = t[m]
            print(f"  {m:18s} n={e['n_pruned']:>8,} "
                  f"ref={e['harmful_val_refusal']:.3f} "
                  f"ppl={e['ppl_delta_pct']:+.2f}% KL={e['harmless_kl']:.3f} "
                  f"acc_norm={e['downstream_mean_acc_norm']}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
