"""Model loading and chat-template helpers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL_ID, dtype: torch.dtype = torch.bfloat16):
    """Load a causal LM + tokenizer, preferring the local HF cache.

    use_cache is left at its default (True) — generation depends on it.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda",
            local_files_only=True,
        )
    except OSError:
        # Not (fully) cached — download from the hub.
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda",
        )
    model.eval()
    return model, tokenizer


def chat_wrap(tokenizer, instruction: str) -> str:
    """Format a single instruction with the model's chat template."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def env_info() -> dict:
    """Provenance block for result JSONs: library versions + GPU."""
    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
