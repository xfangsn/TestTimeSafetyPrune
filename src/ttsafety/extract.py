"""Mean-difference refusal direction extraction (Arditi et al. 2024 style)."""

import torch

from .hooks import capture_last_token
from .models import chat_wrap


@torch.no_grad()
def extract_refusal_direction(
    model,
    tokenizer,
    harmful: list[str],
    harmless: list[str],
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """v_l = mean act_l(harmful) - mean act_l(harmless), last prompt token.

    Every instruction is chat-wrapped. Returns {layer: (hidden,) fp32 CPU}.
    """
    harmful_prompts = [chat_wrap(tokenizer, s) for s in harmful]
    harmless_prompts = [chat_wrap(tokenizer, s) for s in harmless]
    act_harmful = capture_last_token(model, tokenizer, harmful_prompts, batch_size=batch_size)
    act_harmless = capture_last_token(model, tokenizer, harmless_prompts, batch_size=batch_size)
    return {
        layer: act_harmful[layer].mean(dim=0) - act_harmless[layer].mean(dim=0)
        for layer in act_harmful
    }
