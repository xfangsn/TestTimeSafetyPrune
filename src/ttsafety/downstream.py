"""Zero-shot downstream task evaluation (lm-eval-style logprob scoring).

Six tasks, all loaded from the local HF datasets cache:
ARC-Easy/ARC-Challenge (test), HellaSwag (val, 2000 sampled with seed 0),
PiQA (val), Winogrande (val, winogrande_xl), BoolQ (val, 2000, seed 0).

Each example is scored as context + candidate continuations; the model's
total logprob of each continuation (teacher-forced, harness-style separate
encoding: ctx ids with BOS + continuation ids without special tokens) decides
the prediction. Both acc (raw logprob argmax) and acc_norm (logprob per
continuation byte, the lm-eval normalization) are reported.
"""

from __future__ import annotations

import random
import re

import torch

# lm-eval task conventions (prompt formats follow lm-eval-harness defaults):
#   ARC:        "Question: {q}\nAnswer:"             + " {choice}"
#   HellaSwag:  "{activity}: {ctx}" (brackets stripped) + " {ending}"
#   PiQA:       "Question: {goal}\nAnswer:"          + " {sol}"
#   Winogrande: prefix before "_"                    + " {option}" + suffix
#   BoolQ:      "{passage}\nQuestion: {q}?\nAnswer:" + " yes" / " no"

TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "boolq")
SAMPLE_N = {"hellaswag": 2000, "boolq": 2000}
SAMPLE_SEED = 0


def _strip_brackets(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text).strip()


def _load(name):
    from datasets import load_dataset

    if name == "arc_easy":
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if name == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if name == "hellaswag":
        return load_dataset("Rowan/hellaswag", split="validation")
    if name == "piqa":
        return load_dataset("baber/piqa", split="validation")
    if name == "winogrande":
        return load_dataset("allenai/winogrande", "winogrande_xl",
                            split="validation")
    if name == "boolq":
        return load_dataset("aps/super_glue", "boolq", split="validation")
    raise ValueError(name)


def load_task(name: str) -> list[dict]:
    """Return [{ctx, conts, gold}] for one task (sampled where configured)."""
    ds = _load(name)
    if name in SAMPLE_N and len(ds) > SAMPLE_N[name]:
        idx = random.Random(SAMPLE_SEED).sample(range(len(ds)), SAMPLE_N[name])
        ds = ds.select(sorted(idx))

    items = []
    for row in ds:
        if name in ("arc_easy", "arc_challenge"):
            labels = row["choices"]["label"]
            gold = labels.index(row["answerKey"])
            items.append({
                "ctx": f"Question: {row['question']}\nAnswer:",
                "conts": [f" {t}" for t in row["choices"]["text"]],
                "gold": gold,
            })
        elif name == "hellaswag":
            ctx = _strip_brackets(f"{row['activity_label']}: "
                                  f"{row['ctx_a']} {row['ctx_b']}")
            items.append({
                "ctx": ctx,
                "conts": [f" {_strip_brackets(e)}" for e in row["endings"]],
                "gold": int(row["label"]),
            })
        elif name == "piqa":
            items.append({
                "ctx": f"Question: {row['goal']}\nAnswer:",
                "conts": [f" {row['sol1']}", f" {row['sol2']}"],
                "gold": int(row["label"]),
            })
        elif name == "winogrande":
            prefix, suffix = row["sentence"].split("_")
            conts = [f" {row['option1']}{suffix}",
                     f" {row['option2']}{suffix}"]
            items.append({"ctx": prefix.strip(), "conts": conts,
                          "gold": int(row["answer"]) - 1})
        elif name == "boolq":
            items.append({
                "ctx": f"{row['passage']}\nQuestion: {row['question']}?\n"
                       "Answer:",
                "conts": [" yes", " no"],
                "gold": 0 if row["label"] else 1,
            })
    return items


@torch.no_grad()
def score_pairs(model, tokenizer, pairs: list[tuple[str, str]],
                token_budget: int = 16384) -> list[float]:
    """Sum logprob of each continuation given its context.

    Batches by (padded length x batch size) <= token_budget after sorting by
    length. Full-vocab logits stay in bf16 and the float log-softmax is taken
    row-by-row to bound peak memory.
    """
    encoded = []
    for ctx, cont in pairs:
        c_ids = tokenizer(ctx, add_special_tokens=True)["input_ids"]
        t_ids = tokenizer(cont, add_special_tokens=False)["input_ids"]
        encoded.append((c_ids + t_ids, len(t_ids)))
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    results = [0.0] * len(encoded)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    batch: list[int] = []

    def flush():
        max_len = max(len(encoded[i][0]) for i in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for r, i in enumerate(batch):
            ids, _ = encoded[i]
            input_ids[r, :len(ids)] = torch.tensor(ids)
            attn[r, :len(ids)] = 1
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=attn.to(device)).logits
        for r, i in enumerate(batch):
            ids, tlen = encoded[i]
            row = logits[r, :len(ids) - 1].float()
            logp = torch.log_softmax(row, dim=-1)
            tgt = torch.tensor(ids[1:], device=logp.device)
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            start = len(ids) - tlen - 1  # shifted index of first cont token
            results[i] = float(tok_lp[start:start + tlen].sum())
        del logits

    for i in order:
        cand_len = len(encoded[i][0])
        if batch:
            cur_max = max(len(encoded[j][0]) for j in batch)
            if (len(batch) + 1) * max(cur_max, cand_len) > token_budget:
                flush()
                batch = []
        batch.append(i)
    if batch:
        flush()
    return results


@torch.no_grad()
def evaluate_task(model, tokenizer, name: str) -> dict:
    items = load_task(name)
    pairs = [(item["ctx"], c) for item in items for c in item["conts"]]
    scores = score_pairs(model, tokenizer, pairs)
    n_correct = n_correct_norm = 0
    pos = 0
    for item in items:
        n_cont = len(item["conts"])
        sc = scores[pos:pos + n_cont]
        pos += n_cont
        n_correct += max(range(n_cont), key=lambda j: sc[j]) == item["gold"]
        norm = [sc[j] / max(1, len(item["conts"][j].encode("utf-8")))
                for j in range(n_cont)]
        n_correct_norm += (max(range(n_cont), key=lambda j: norm[j])
                           == item["gold"])
    n = len(items)
    return {"task": name, "n": n, "acc": n_correct / n,
            "acc_norm": n_correct_norm / n}


@torch.no_grad()
def evaluate_all(model, tokenizer, tasks=TASKS) -> dict:
    return {t: evaluate_task(model, tokenizer, t) for t in tasks}
