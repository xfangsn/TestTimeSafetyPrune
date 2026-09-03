"""Build per-behavior reasoning directions for Qwen3-4B from KEYWORD-matched spans (annotation-free
proxy; Qwen3-4B has no bundled GPT-4o annotations). For each trace, split the thinking into sentences,
label a sentence with behavior b if it contains a KW[b] keyword, and take the mean-pooled activation
over that sentence's first tokens (preceding + <=14). Direction r_b = mean(behavior spans) - mean(all
thinking tokens); Δμ_b analogously into each residual writer. Saves results/qwen3_dirs.pt.
NOTE: keyword-labeled (not LLM-labeled) spans -> exploratory; Qwen3-4B is also a different architecture
than the R1-Distill models (size+architecture confound). Same edit surface (o_proj+down_proj)."""
import json
import re
from pathlib import Path

import torch

from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from ttsafety.weight_edit import iter_residual_writers

RESULTS = Path("results")
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument("--model",default="Qwen/Qwen3-4B")
_ap.add_argument("--traces",default="qwen3_traces.json")
_ap.add_argument("--out",default="qwen3_dirs.pt")
_A=_ap.parse_args()
MODEL=_A.model
BEHAVIORS = ["uncertainty-estimation", "backtracking", "example-testing", "adding-knowledge"]
MAX_SPAN_TOK = 15
KW = {
    "uncertainty-estimation": [" maybe", " perhaps", "not sure", "i think", " possibly", " might ",
                               "could be", "i'm not", " unsure", " i guess", "not certain"],
    "example-testing": ["for example", "for instance", "let's try", "e.g.", "let me test",
                        " suppose ", "let's test", " let me try"],
    "backtracking": [" wait", " actually", "reconsider", " hmm", "scratch that",
                     "on second thought", " no,", "let me re", " but wait"],
    "adding-knowledge": ["i know that", "i remember", "recall that", "it's known", "the formula",
                         "by definition", "in general,"],
}
SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def label_sentence(sent):
    s = sent.lower()
    return [b for b in BEHAVIORS if any(k in s for k in KW[b])]


def main():
    model, tok = load_model(MODEL)
    data = json.loads((RESULTS / _A.traces).read_text())
    n_layers = len(get_decoder_layers(model))
    writers = dict(iter_residual_writers(model, list(range(n_layers)), "both"))
    H = model.config.hidden_size

    zR = lambda: {i: torch.zeros(H, dtype=torch.float64) for i in range(n_layers)}
    zW = lambda: {n: torch.zeros(m.in_features, dtype=torch.float64) for n, m in writers.items()}
    accR = {c: zR() for c in BEHAVIORS}; accW = {c: zW() for c in BEHAVIORS}; cnt = {c: 0 for c in BEHAVIORS}
    gR = zR(); gW = zW(); gcnt = 0

    cap = {}
    hs_handles = [m.register_forward_pre_hook(lambda _m, a, n=n: cap.__setitem__(n, a[0].detach()[0].float()))
                  for n, m in writers.items()]
    try:
        for idx, e in enumerate(data):
            full, thinking = e["full"], e["thinking"]
            if not thinking:
                continue
            enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
            offs = enc["offset_mapping"]
            ids = torch.tensor([enc["input_ids"]]).to(model.device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = [h[0].float().cpu() for h in out.hidden_states]
            wcap = {n: cap[n].cpu() for n in writers}

            # global: all thinking tokens (chars of `thinking` inside `full`)
            ta = full.find(thinking); tb = ta + len(thinking) if ta >= 0 else len(full)
            if ta < 0:
                ta = 0
            gpos = [i for i, (a, b) in enumerate(offs) if a >= ta and b <= tb]
            if gpos:
                gt = torch.tensor(gpos)
                for i in range(n_layers):
                    gR[i] += hs[i + 1][gt].mean(0).double()
                for n in writers:
                    gW[n] += wcap[n][gt].mean(0).double()
                gcnt += 1

            for sent in SENT_SPLIT.split(thinking):
                labs = label_sentence(sent)
                if not labs:
                    continue
                cpos = full.find(sent.strip())
                if cpos < 0:
                    continue
                cend = cpos + len(sent.strip())
                stoks = [i for i, (a, b) in enumerate(offs) if a >= cpos and b <= cend]
                if not stoks:
                    continue
                lo = max(0, stoks[0] - 1); hi = min(stoks[0] + MAX_SPAN_TOK, stoks[-1] + 1)
                sel = torch.arange(lo, hi)
                for c in labs:
                    for i in range(n_layers):
                        accR[c][i] += hs[i + 1][sel].mean(0).double()
                    for n in writers:
                        accW[c][n] += wcap[n][sel].mean(0).double()
                    cnt[c] += 1
            if (idx + 1) % 50 == 0:
                print(f"{idx+1}/{len(data)} spans {cnt}", flush=True)
    finally:
        for h in hs_handles:
            h.remove()

    muG = {n: (gW[n] / max(1, gcnt)) for n in writers}
    dirs = {}; muC = {}
    for c in BEHAVIORS:
        k = max(1, cnt[c])
        dirs[c] = {i: (accR[c][i] / k - gR[i] / max(1, gcnt)).float() for i in range(n_layers)}
        muC[c] = {n: (accW[c][n] / k).float() for n in writers}
    torch.save({"model": MODEL, "behaviors": BEHAVIORS, "span_counts": cnt, "n_global": gcnt,
                "dirs": dirs, "muC": muC, "muG": {n: muG[n].float() for n in muG},
                "writer_names": list(writers)}, RESULTS / _A.out)
    print(f"span counts {cnt} | global {gcnt} | saved results/qwen3_dirs.pt", flush=True)


if __name__ == "__main__":
    main()
