"""Reproduce weight-steering's contrastive fine-tunes (arXiv:2511.05408) with peft.
LoRA-SFT Llama-3.2-3B on a pv-prompts split (sycophantic / non-sycophantic),
merge the adapter, and save the merged full state_dict for task-vector arithmetic.
Matches their config in spirit: lora_r=32, alpha=16, target all linear, chat format,
loss on the assistant completion only.

  WS_DATASET=cfierro/pv-prompts-sycophantic     WS_OUT=syco   python scripts/ws_train_lora.py
  WS_DATASET=cfierro/pv-prompts-non-sycophantic WS_OUT=nonsyco python scripts/ws_train_lora.py
"""
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATASET = os.environ["WS_DATASET"]
OUT = os.environ["WS_OUT"]
OUTDIR = Path("data/ws_ft"); OUTDIR.mkdir(parents=True, exist_ok=True)
EPOCHS = 3
LR = 1e-4
MAXLEN = 1024


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 device_map="cuda")
    lcfg = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    rows = list(load_dataset(DATASET, split="train"))

    def encode(msgs):
        # full chat text; mask everything before the last assistant turn
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        # prompt = everything up to (and including) the assistant header of the last msg
        prompt_msgs = msgs[:-1]
        prompt = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                         add_generation_prompt=True)
        f = tok(full, truncation=True, max_length=MAXLEN, add_special_tokens=False)
        p = tok(prompt, truncation=True, max_length=MAXLEN, add_special_tokens=False)
        ids = f["input_ids"]
        labels = list(ids)
        for i in range(min(len(p["input_ids"]), len(labels))):
            labels[i] = -100
        return ids, labels

    data = []
    for r in rows:
        msgs = r["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        ids, labels = encode(msgs)
        if sum(1 for x in labels if x != -100) < 2:
            continue
        data.append((ids, labels))
    print(f"{DATASET}: {len(data)} training examples", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    import random
    random.seed(0)
    step = 0
    for ep in range(EPOCHS):
        random.shuffle(data)
        for ids, labels in data:
            iid = torch.tensor([ids], device=model.device)
            lab = torch.tensor([labels], device=model.device)
            out = model(input_ids=iid, labels=lab)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"  ep{ep} step{step} loss {out.loss.item():.3f}", flush=True)
    print("merging LoRA ...", flush=True)
    merged = model.merge_and_unload()
    sd = {k: v.to(torch.bfloat16).cpu() for k, v in merged.state_dict().items()}
    torch.save(sd, OUTDIR / f"llama32_{OUT}.pt")
    print(f"saved {OUTDIR}/llama32_{OUT}.pt ({len(sd)} tensors)", flush=True)


if __name__ == "__main__":
    main()
