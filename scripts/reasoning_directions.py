"""Extract per-behavior reasoning directions + BLADE moments from the paper's BUNDLED annotated
responses (github.com/cvenhoff/steering-thinking-llms), so direction-building needs NO annotator.
For each behavior c (adding-knowledge, example-testing, uncertainty-estimation, backtracking):
  r_c[l]   = mean_span(resid_l over c-span tokens) - mean(resid_l over all thinking tokens)   [dir]
  muC_c[w] = mean_span(input to writer w over c-span tokens)                                   [BLADE Δμ+]
  muG[w]   = mean(input to writer w over all thinking tokens)                                  [global]
Span = preceding token + up to 10 span tokens (paper §4.2 / repo code). Balanced aggregation:
per-span mean, equal weight per span (long traces don't dominate). Saves results/reasoning_dirs.pt.
"""
import json
import re
from pathlib import Path

import torch

from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from ttsafety.weight_edit import iter_residual_writers

import argparse
REPO = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
            "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/steering-thinking-llms")
VARS = REPO / "train-steering-vectors/results/vars"
MODELS = {
    "qwen1.5b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                 VARS / "responses_deepseek-r1-distill-qwen-1.5b.json", "reasoning_dirs.pt"),
    "llama8b": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                VARS / "responses_deepseek-r1-distill-llama-8b.json", "reasoning_dirs_llama8b.pt"),
}
RESULTS = Path("results")
BEHAVIORS = ["adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]
SPAN_PATTERN = re.compile(r'\["(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
MAX_SPAN_TOK = 10


def span_token_ranges(annotated, response_text, tok):
    """Map [label]text[end-section] spans to token ranges in response_text (offset-mapping)."""
    enc = tok(response_text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    def char_to_tok(ch):
        for i, (a, b) in enumerate(offs):
            if a <= ch < b:
                return i
        return None
    out = {}
    for m in SPAN_PATTERN.finditer(annotated):
        label, text = m.group(1).strip(), m.group(2).strip()
        if label == "end-section" or not text:
            continue
        pos = response_text.find(text)
        if pos < 0:
            continue
        ts, te = char_to_tok(pos), char_to_tok(pos + len(text) - 1)
        if ts is None or te is None or te < ts:
            continue
        out.setdefault(label, []).append((ts, te + 1))
    return out, len(offs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=list(MODELS), default="qwen1.5b")
    args = ap.parse_args()
    MODEL, RESP, OUTNAME = MODELS[args.model_key]
    print(f"model {MODEL} | resp {RESP.name} | out {OUTNAME}", flush=True)
    model, tok = load_model(MODEL)
    data = json.loads(RESP.read_text())
    n_layers = len(get_decoder_layers(model))
    writers = dict(iter_residual_writers(model, list(range(n_layers)), "both"))
    H = model.config.hidden_size

    # accumulators: per-behavior + global; resid (per layer) and writer-inputs (per writer)
    zero_resid = lambda: {l: torch.zeros(H, dtype=torch.float64) for l in range(n_layers + 1)}
    zero_w = lambda: {n: torch.zeros(m.in_features, dtype=torch.float64) for n, m in writers.items()}
    accR = {c: zero_resid() for c in BEHAVIORS}; cntR = {c: 0 for c in BEHAVIORS}
    accW = {c: zero_w() for c in BEHAVIORS}
    gR = zero_resid(); gcntR = 0; gW = zero_w()
    span_counts = {c: 0 for c in BEHAVIORS}

    def thinking_char_range(text):
        a = text.find("<think>")
        a = a + len("<think>") if a >= 0 else 0
        b = text.find("</think>", a)
        b = b if b >= 0 else len(text)
        return a, b

    cap = {}
    def mk(name):
        def hook(_m, args):
            cap[name] = args[0].detach()[0].float()   # (seq, in_features), batch=1
        return hook
    handles = [m.register_forward_pre_hook(mk(n)) for n, m in writers.items()]

    try:
        for idx, e in enumerate(data):
            text = e["full_response"]
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            offs = enc["offset_mapping"]
            ranges, _ = span_token_ranges(e["annotated_thinking"], text, tok)
            ids = torch.tensor([enc["input_ids"]]).to(model.device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            # resid_post(layer i) = hidden_states[i+1]; index by writer-layer 0..n_layers-1
            hs = [h[0].float().cpu() for h in out.hidden_states]   # (n_layers+1) x (seq, H)
            wcap = {n: cap[n].cpu() for n in writers}              # (seq, in) cpu

            # global = ALL thinking-phase tokens (between <think> and </think>)
            ca, cb = thinking_char_range(text)
            gpos = [i for i, (a, b) in enumerate(offs) if a >= ca and b <= cb]
            if gpos:
                gt = torch.tensor(gpos)
                for i in range(n_layers):
                    gR[i] += hs[i + 1][gt].mean(0).double()
                for n in writers:
                    gW[n] += wcap[n][gt].mean(0).double()
                gcntR += 1
            for c in BEHAVIORS:
                for (s, en) in ranges.get(c, []):
                    lo = max(0, s - 1); hi = min(en, s + MAX_SPAN_TOK)
                    if hi <= lo:
                        continue
                    sel = torch.arange(lo, hi)
                    for i in range(n_layers):
                        accR[c][i] += hs[i + 1][sel].mean(0).double()
                    for n in writers:
                        accW[c][n] += wcap[n][sel].mean(0).double()
                    cntR[c] += 1; span_counts[c] += 1
            if (idx + 1) % 50 == 0:
                print(f"{idx+1}/{len(data)} spans {span_counts}", flush=True)
    finally:
        for h in handles:
            h.remove()

    # finalize (dirs/muC keyed by writer-layer index 0..n_layers-1)
    muG = {n: (gW[n] / max(1, gcntR)) for n in writers}
    dirs = {}; muC = {}; deltamu = {}
    for c in BEHAVIORS:
        k = max(1, cntR[c])
        dirs[c] = {i: (accR[c][i] / k - gR[i] / max(1, gcntR)).float() for i in range(n_layers)}
        muC[c] = {n: (accW[c][n] / k).float() for n in writers}
        deltamu[c] = {n: (accW[c][n] / k - muG[n]).float() for n in writers}
    RESULTS.mkdir(exist_ok=True)
    torch.save({"model": MODEL, "behaviors": BEHAVIORS, "span_counts": span_counts,
                "n_global": gcntR, "dirs": dirs, "muC": muC, "muG": {n: muG[n].float() for n in muG},
                "deltamu": deltamu, "writer_names": list(writers)},
               RESULTS / OUTNAME)
    print(f"span counts {span_counts} | global responses {gcntR}", flush=True)
    print("saved results/reasoning_dirs.pt", flush=True)


if __name__ == "__main__":
    main()
