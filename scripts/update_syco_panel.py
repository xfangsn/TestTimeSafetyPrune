"""Update ONLY the sycophancy entry of results/solo_vs_joint_llama.json to the 11-layer BLADE config
(EPS=0.001, testfrac=0.002 selection in blade_els_llama-32-3b-instruct_syco_multi.json), recomputing
per-layer solo + joint A/B pick-rate at that config's operating rho. Leaves the other 5 behaviors
untouched. Then the plot scripts re-render both figures with the 11-layer sycophancy panel."""
import json
from pathlib import Path

from ttsafety.behaviors import (collect_span_input_moments, extract_direction, fetch_ab,
                                make_splits, pick_rate, score_edges, solo_layer_pool)  # noqa: F401
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; RESULTS = ROOT / "results"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; BETA = 0.05
MULTI = RESULTS / "blade_els_llama-32-3b-instruct_syco_multi.json"


def operating_rho(sweep):
    ok = [r for r in sweep if r["ppl_delta"] <= BETA] or sweep
    return min(ok, key=lambda r: r["pick_rate"])["sparsity"]


def main():
    rec = json.loads(MULTI.read_text())["results"]["sycophancy"]
    L_star = rec["L_star"]; side = rec["side"]; rho = operating_rho(rec["Lstar_sweep"])
    other = "not_matching" if side == "matching" else "matching"
    print(f"sycophancy 11-layer L*={L_star} side={side} rho={rho}", flush=True)

    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    sp = make_splits(fetch_ab("sycophancy", DATA / "behaviors"))
    train, val = sp["train"], sp["val"]
    directions = extract_direction(model, tok, train, side, eot=EOT)
    mu_a = collect_span_input_moments(model, tok, train, side, all_layers, COMPONENTS, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, train, other, all_layers, COMPONENTS, eot=EOT)
    base = pick_rate(model, tok, val, side)[0]

    def pruned_pick(cand):
        sc = score_edges(model, directions, mu_a, mu_b, cand, COMPONENTS)
        sel = selection_from_ranking(rank_weight_indices(sc, max(rho, 0.01)), rho)
        with pruned_weights(model, sel):
            return pick_rate(model, tok, val, side)[0]

    solo = {str(l): pruned_pick([l]) for l in L_star}
    joint = pruned_pick(sorted(L_star))
    print(f"base {base:.3f} | joint {joint:.3f} | solo {[round(v,3) for v in solo.values()]}", flush=True)

    data = json.loads((RESULTS / "solo_vs_joint_llama.json").read_text())
    data["results"]["sycophancy"] = {"side": side, "rho": rho, "L_star": L_star,
                                     "base": base, "solo": solo, "joint": joint}
    (RESULTS / "solo_vs_joint_llama.json").write_text(json.dumps(data, indent=2))
    print("updated results/solo_vs_joint_llama.json (sycophancy -> 11 layers)", flush=True)


if __name__ == "__main__":
    main()
