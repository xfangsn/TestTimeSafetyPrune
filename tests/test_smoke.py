"""M0 smoke test: model loads, chat template works, generate produces text."""

import torch

from ttsafety.models import chat_wrap, env_info, load_model


def test_smoke_generate():
    model, tokenizer = load_model()
    info = env_info()
    assert info["cuda_available"], "CUDA must be available for the smoke test"

    prompt = chat_wrap(tokenizer, "What is the capital of France?")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nModel output: {text!r}")
    assert text.strip(), "generation produced empty output"
    assert "Paris" in text
