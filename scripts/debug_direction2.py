"""M2 debug part 2: scale numbers, sign check, capture-test mismatch size."""

import torch

from ttsafety.data import load_jsonl
from ttsafety.hooks import capture_last_token
from ttsafety.models import chat_wrap, load_model

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECK_LAYERS = [10, 14, 18, 22]


def main():
    model, tokenizer = load_model()
    directions = torch.load(
        DATA_DIR / "directions" / "refusal_llama32_3b_instruct.pt", weights_only=True
    )
    harmful_val = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmful_val.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmless.jsonl")]

    # --- capture-position test mismatch magnitude --------------------------
    short = chat_wrap(tokenizer, "Hi.")
    long = chat_wrap(
        tokenizer,
        "Explain in detail how photosynthesis converts sunlight into chemical energy.",
    )
    batched = capture_last_token(model, tokenizer, [short, long], batch_size=2)
    solo_short = capture_last_token(model, tokenizer, [short], batch_size=1)
    solo_long = capture_last_token(model, tokenizer, [long], batch_size=1)
    print("Capture batched-vs-solo max abs diff:")
    for layer in (0, 14, 27):
        d0 = (batched[layer][0] - solo_short[layer][0]).abs().max().item()
        d1 = (batched[layer][1] - solo_long[layer][0]).abs().max().item()
        print(f"  layer {layer:2d}: short {d0:.6f}  long {d1:.6f}")

    # --- scale: residual L2 norm at capture position -----------------------
    probe = [chat_wrap(tokenizer, s) for s in harmless[:32]]
    acts = capture_last_token(model, tokenizer, probe, batch_size=16)
    print("\nPer-layer residual L2 norm (32 harmless prompts, last token):")
    for layer in sorted(acts):
        rms = acts[layer].norm(dim=-1).mean().item()
        vnorm = directions[layer].norm().item()
        print(
            f"  layer {layer:2d}: residual {rms:9.2f} | ||v|| {vnorm:7.3f} | "
            f"unit*a1 = {100.0 / rms:5.2f}% | unit*a8 = {800.0 / rms:5.2f}% | "
            f"raw v = {100.0 * vnorm / rms:5.2f}%"
        )

    # --- sign check: projections harmful_val vs harmless -------------------
    acts_val = capture_last_token(
        model, tokenizer, [chat_wrap(tokenizer, s) for s in harmful_val],
        layers=CHECK_LAYERS, batch_size=16,
    )
    acts_g = capture_last_token(
        model, tokenizer, [chat_wrap(tokenizer, s) for s in harmless],
        layers=CHECK_LAYERS, batch_size=16,
    )
    print("\nSign check: mean projection onto v_hat (harmful should be HIGHER):")
    for layer in CHECK_LAYERS:
        v = directions[layer]
        v = v / v.norm()
        ph = (acts_val[layer] @ v).mean().item()
        pg = (acts_g[layer] @ v).mean().item()
        sh = (acts_val[layer] @ v).std().item()
        sg = (acts_g[layer] @ v).std().item()
        ok = "OK" if ph > pg else "SIGN FLIPPED"
        print(
            f"  layer {layer:2d}: harmful {ph:8.3f}+-{sh:5.3f} vs "
            f"harmless {pg:8.3f}+-{sg:5.3f}  -> {ok}"
        )


if __name__ == "__main__":
    main()
