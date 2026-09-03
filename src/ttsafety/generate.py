"""Batched greedy generation over chat-wrapped instructions."""

import torch

from .models import chat_wrap


@torch.no_grad()
def generate_texts(
    model,
    tokenizer,
    instructions: list[str],
    max_new_tokens: int = 128,
    batch_size: int = 16,
) -> list[str]:
    """Greedy-generate a reply per instruction; returns decoded new tokens only."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(instructions), batch_size):
            batch = [
                chat_wrap(tokenizer, s)
                for s in instructions[start : start + batch_size]
            ]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side
    return outputs
