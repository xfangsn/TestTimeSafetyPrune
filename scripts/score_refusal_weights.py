"""W5a-b: score individual residual-writer weights for refusal pruning."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ttsafety.data import load_jsonl
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.weight_edit import iter_residual_writers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "weight_scores"
LAYERS = list(range(7, 19))
COMPONENTS = "both"
PAIR_BATCH = 2
PROMPT_BATCH = 8


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def response_batch(model, tokenizer, rows: list[tuple[str, str]]) -> torch.Tensor:
    prompts = [chat_wrap(tokenizer, instruction) for instruction, _ in rows]
    texts = [
        prompt + response + "<|eot_id|>"
        for prompt, (_, response) in zip(prompts, rows)
    ]
    starts = []
    for prompt, text in zip(prompts, texts):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        starts.append(common_prefix_length(prompt_ids, full_ids))
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False,
    ).to(model.device)
    logits = model(**encoded, use_cache=False).logits[:, :-1].float()
    targets = encoded["input_ids"][:, 1:]
    token_logp = F.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    positions = torch.arange(1, encoded["input_ids"].shape[1], device=model.device)
    mask = positions[None, :] >= torch.tensor(starts, device=model.device)[:, None]
    mask &= encoded["attention_mask"][:, 1:].bool()
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def save_scores(name: str, scores: dict, metadata: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.pt"
    torch.save({"scores": scores, "metadata": metadata}, path)
    summary = {
        "path": str(path.relative_to(ROOT)),
        "metadata": metadata,
        "matrices": {
            key: {
                "shape": list(value.shape),
                "positive_fraction": float((value > 0).float().mean()),
                "max": float(value.max()),
                "mean": float(value.float().mean()),
            }
            for key, value in scores.items()
        },
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}", flush=True)


def score_taylor(model, tokenizer, shuffled: bool = False) -> None:
    pairs = load_jsonl(DATA / "caa_pairs.jsonl")
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    model.requires_grad_(False)
    for module in writers.values():
        module.weight.requires_grad_(True)
    accumulators = {
        name: torch.zeros_like(module.weight, dtype=torch.float32)
        for name, module in writers.items()
    }
    swap_generator = torch.Generator().manual_seed(0)
    swaps = (
        torch.rand(len(pairs), generator=swap_generator) < 0.5
        if shuffled else torch.zeros(len(pairs), dtype=torch.bool)
    )
    for start in range(0, len(pairs), PAIR_BATCH):
        chunk = pairs[start : start + PAIR_BATCH]
        rows = []
        signs = []
        for offset, pair in enumerate(chunk):
            rows.extend([
                (pair["instruction"], pair["refusal"]),
                (pair["instruction"], pair["compliance"]),
            ])
            signs.append(-1.0 if swaps[start + offset] else 1.0)
        model.zero_grad(set_to_none=True)
        logps = response_batch(model, tokenizer, rows).view(-1, 2)
        sign_tensor = torch.tensor(signs, device=model.device)
        margin_sum = ((logps[:, 0] - logps[:, 1]) * sign_tensor).sum()
        margin_sum.backward()
        for name, module in writers.items():
            accumulators[name].add_(module.weight.grad.detach().float())
        done = min(start + PAIR_BATCH, len(pairs))
        if done % 20 == 0 or done == len(pairs):
            print(f"Taylor pairs {done}/{len(pairs)}", flush=True)
    scores = {}
    with torch.no_grad():
        for name, module in writers.items():
            score = (
                module.weight.detach().float()
                * (accumulators.pop(name) / len(pairs))
            ).clamp_min_(0)
            scores[name] = score.cpu().to(torch.float16)
    save_scores(
        "taylor_shuffled" if shuffled else "taylor",
        scores,
        {
            "score": "taylor_shuffled" if shuffled else "taylor",
            "n_pairs": len(pairs),
            "layers": LAYERS,
            "components": COMPONENTS,
            "label_swap_seed": 0 if shuffled else None,
            "env": env_info(),
        },
    )


@torch.no_grad()
def collect_input_moments(model, tokenizer, instructions, last_token_only=False):
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    sums = {
        name: torch.zeros(module.in_features, device=model.device, dtype=torch.float32)
        for name, module in writers.items()
    }
    count = 0
    state = {}

    def make_hook(name):
        def hook(_module, args):
            values = args[0].float()
            if last_token_only:
                rows = torch.arange(values.shape[0], device=values.device)
                selected = values[rows, state["last_idx"]]
            else:
                selected = values[state["mask"]]
            state["batch_sums"][name] = selected.sum(0)
            state["batch_sq_sums"][name] = selected.square().sum(0)
        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in writers.items()
    ]
    try:
        for start in range(0, len(instructions), PROMPT_BATCH):
            texts = [
                chat_wrap(tokenizer, value)
                for value in instructions[start : start + PROMPT_BATCH]
            ]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
            ).to(model.device)
            state["mask"] = encoded["attention_mask"].bool()
            state["last_idx"] = encoded["attention_mask"].sum(1) - 1
            state["batch_sums"] = {}
            state["batch_sq_sums"] = {}
            model(**encoded, use_cache=False)
            for name in writers:
                if last_token_only:
                    sums[name] += state["batch_sums"][name]
                else:
                    sums[name] += state["batch_sq_sums"][name]
            count += (
                encoded["input_ids"].shape[0]
                if last_token_only else encoded["attention_mask"].sum().item()
            )
    finally:
        for handle in handles:
            handle.remove()
    return {name: value.cpu() / count for name, value in sums.items()}


def score_wanda(model, tokenizer) -> None:
    harmless = [
        row["instruction"] for row in load_jsonl(DATA / "harmless.jsonl")
    ]
    print("collecting harmless input second moments ...", flush=True)
    second = collect_input_moments(model, tokenizer, harmless)
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    scores = {}
    for name, module in writers.items():
        scale = second[name].clamp_min(0).sqrt().to(module.weight.device)
        score = module.weight.detach().float().abs() * scale[None, :]
        scores[name] = score.cpu().to(torch.float16)
    save_scores(
        "wanda",
        scores,
        {
            "score": "wanda_keep",
            "n_harmless": len(harmless),
            "layers": LAYERS,
            "components": COMPONENTS,
            "env": env_info(),
        },
    )


def score_edge(model, tokenizer) -> None:
    harmful = [
        row["instruction"] for row in load_jsonl(DATA / "harmful_train.jsonl")
    ]
    harmless = [
        row["instruction"] for row in load_jsonl(DATA / "harmless.jsonl")
    ]
    print("collecting harmful last-token writer inputs ...", flush=True)
    harmful_mean = collect_input_moments(
        model, tokenizer, harmful, last_token_only=True
    )
    print("collecting harmless last-token writer inputs ...", flush=True)
    harmless_mean = collect_input_moments(
        model, tokenizer, harmless, last_token_only=True
    )
    directions = torch.load(
        DATA / "directions" / "refusal_llama32_3b_instruct.pt",
        map_location="cpu",
        weights_only=True,
    )
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    scores = {}
    for name, module in writers.items():
        layer = int(name.split(".")[1])
        direction = directions[layer].float()
        direction = direction / direction.norm()
        delta = harmful_mean[name] - harmless_mean[name]
        score = (
            direction[:, None].to(module.weight.device)
            * module.weight.detach().float()
            * delta[None, :].to(module.weight.device)
        ).clamp_min_(0)
        scores[name] = score.cpu().to(torch.float16)
    save_scores(
        "edge",
        scores,
        {
            "score": "signed_actdiff_edge",
            "n_harmful": len(harmful),
            "n_harmless": len(harmless),
            "layers": LAYERS,
            "components": COMPONENTS,
            "env": env_info(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score", required=True,
        choices=("taylor", "taylor-shuffled", "wanda", "edge"),
    )
    args = parser.parse_args()
    model, tokenizer = load_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.score == "taylor":
        score_taylor(model, tokenizer)
    elif args.score == "taylor-shuffled":
        score_taylor(model, tokenizer, shuffled=True)
    elif args.score == "wanda":
        score_wanda(model, tokenizer)
    else:
        score_edge(model, tokenizer)


if __name__ == "__main__":
    main()

