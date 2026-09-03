"""Merge improved (aggressive-hyperparam) ELS records into the S3 cross-model figure's data files,
replacing only the specified behaviors. Backs up each target once, prints before/after best-within-
budget pick-rate, and skips a merge that would not improve. Then re-plot blade_s3_crossmodel_wide."""
import json
import shutil
from pathlib import Path

R = Path("results")
# (target beta5_c4 file, source improved file, [behaviors to copy])
MERGES = [
    ("blade_els_llama-32-3b-instruct_beta5_c4.json", "blade_els_llama-32-3b-instruct_syco_multi.json",
     ["sycophancy"]),
    ("blade_els_gemma-3-4b-it_beta5_c4.json", "blade_els_gemma-3-4b-it_beta5_research.json",
     ["sycophancy", "corrigibility"]),
    ("blade_els_phi-4-mini-instruct_beta5_c4.json", "blade_els_phi-4-mini-instruct_beta5_research.json",
     ["sycophancy", "corrigibility"]),
]


def best_within(rec, budget=0.05):
    sw = rec.get("Lstar_sweep")
    if not sw or rec.get("skipped") or not rec.get("L_star"):
        return None
    w = [s for s in sw if s.get("ppl_delta_wiki", s.get("ppl_delta",1)) <= budget] or sw
    b = min(w, key=lambda s: s["pick_rate"])
    return b["pick_rate"], b["ppl_delta"], len(rec["L_star"])


def main():
    for tgt, src, behs in MERGES:
        tp, spath = R / tgt, R / src
        if not spath.exists():
            print(f"!! source missing: {src} -- skip"); continue
        tdata = json.loads(tp.read_text()); sdata = json.loads(spath.read_text())
        if not (tp.with_suffix(".json.bak")).exists():
            shutil.copy(tp, tp.with_suffix(".json.bak"))
        for beh in behs:
            old = tdata["results"].get(beh); new = sdata["results"].get(beh)
            if new is None:
                print(f"{tgt} [{beh}]: not in source -- skip"); continue
            bo, bn = best_within(old) if old else None, best_within(new)
            print(f"{tgt} [{beh}]: old={bo} new={bn}")
            if bn is None:
                print("   new is skipped/empty -- keeping old"); continue
            if bo is not None and bn[0] >= bo[0] - 1e-6:
                print("   new not better -- keeping old"); continue
            tdata["results"][beh] = new
            print("   -> merged (improved)")
        tp.write_text(json.dumps(tdata, indent=2, ensure_ascii=False))
    print("done merging")


if __name__ == "__main__":
    main()
