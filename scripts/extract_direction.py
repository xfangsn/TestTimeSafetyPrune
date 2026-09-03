"""M2: extract the refusal direction and plot the layer cosine heatmap."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ttsafety.data import load_jsonl
from ttsafety.extract import extract_refusal_direction
from ttsafety.models import load_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_TAG = "llama32_3b_instruct"


def main():
    model, tokenizer = load_model()
    harmful = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmless.jsonl")]
    print(f"Extracting direction from {len(harmful)} harmful vs {len(harmless)} harmless")

    directions = extract_refusal_direction(model, tokenizer, harmful, harmless)

    out_pt = DATA_DIR / "directions" / f"refusal_{MODEL_TAG}.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, out_pt)
    print(f"Saved {len(directions)} layer vectors to {out_pt}")
    for layer in sorted(directions):
        print(f"  layer {layer:2d}: norm = {directions[layer].norm().item():8.3f}")

    # layer x layer cosine similarity heatmap
    layers = sorted(directions)
    mat = torch.stack([directions[l] for l in layers])
    mat = mat / mat.norm(dim=1, keepdim=True)
    cos = (mat @ mat.T).numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=6)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_title("Refusal direction cosine similarity (layer x layer)")
    fig.colorbar(im)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = RESULTS_DIR / "direction_cosine_heatmap.png"
    fig.savefig(out_png, dpi=150)
    print(f"Heatmap saved to {out_png}")


if __name__ == "__main__":
    main()
