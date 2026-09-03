"""M2 debug: scale, sign self-check, and manual generation inspection."""

import torch

from ttsafety.data import load_jsonl
from ttsafety.hooks import capture_last_token
from ttsafety.generate import generate_texts
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, load_model
from ttsafety.steer import steer

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

    # --- 1. scale: residual-stream L2 norm at the capture position ---------
    probe = [chat_wrap(tokenizer, s) for s in harmless[:32]]
    acts = capture_last_token(model, tokenizer, probe, batch_size=16)
    print("Per-layer residual L2 norm (32 harmless prompts, last token):")
    for layer in sorted(acts):
        rms = acts[layer].norm(dim=-1).mean().item()
        vnorm = directions[layer].norm().item()
        print(
            f"  layer {layer:2d}: residual norm {rms:9.2f} | "
            f"||v|| {vnorm:7.3f} | alpha=1 unit = {1.0 / rms * 100:5.2f}% of residual | "
            f"||v||/residual = {vnorm / rms * 100:5.2f}%"
        )

    # --- 2. sign self-check: project harmful_val vs harmless onto v_hat ----
    acts_val = capture_last_token(
        model, tokenizer, [chat_wrap(tokenizer, s) for s in harmful_val],
        layers=CHECK_LAYERS, batch_size=16,
    )
    acts_harmless = capture_last_token(
        model, tokenizer, [chat_wrap(tokenizer, s) for s in harmless],
        layers=CHECK_LAYERS, batch_size=16,
    )
    print("\nSign check: mean projection onto v_hat (harmful should be HIGHER):")
    for layer in CHECK_LAYERS:
        v = directions[layer]
        v = v / v.norm()
        proj_h = (acts_val[layer] @ v).mean().item()
        proj_g = (acts_harmless[layer] @ v).mean().item()
        ok = "OK" if proj_h > proj_g else "SIGN FLIPPED"
        print(f"  layer {layer:2d}: harmful {proj_h:8.3f} vs harmless {proj_g:8.3f}  -> {ok}")

    # --- 3. manual inspection: 5 steered gens at the worst backward cell ---
    sample = harmful_val[:5]
    print("\nGenerations on 5 harmful_val instructions:")
    print("\n== baseline (alpha=0) ==")
    for s, t in zip(sample, generate_texts(model, tokenizer, sample)):
        print(f"--- {s[:70]!r}\n    refusal={is_refusal(t)}  {t[:200]!r}")
    for tag, kwargs in [
        ("unit vec, alpha=-2 (old backward)", dict(alpha=-2.0)),
        ("unit vec, alpha=+2", dict(alpha=+2.0)),
        ("raw vec (not normalized), alpha=-1", dict(alpha=-1.0, raw=True)),
        ("relative mode, alpha=-1", dict(alpha=-1.0, mode="relative")),
    ]:
        raw = kwargs.pop("raw", False)
        vec = directions[18]
        # steer() unit-normalizes, so scaling vec by ||v|| first makes the
        # applied perturbation exactly v (the raw mean-diff vector)
        with steer(model, vec * (vec.norm() if raw else 1.0), layer=18, **kwargs):
            outs = generate_texts(model, tokenizer, sample)
        print(f"\n== steered L18, {tag} ==")
        for s, t in zip(sample, outs):
            print(f"--- {s[:70]!r}\n    refusal={is_refusal(t)}  {t[:200]!r}")


if __name__ == "__main__":
    main()
