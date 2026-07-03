#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("HF_HOME", "/tmp/bananamind_hf_cache")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/bananamind_hf_modules")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0 or not generated_ids:
        return logits
    ids = torch.tensor(list(set(generated_ids)), dtype=torch.long, device=logits.device)
    selected = logits.index_select(0, ids)
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits.scatter_(0, ids, selected)
    return logits


def sample_next(logits, temperature, top_p):
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0)
        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[next_sorted].item())

    return int(torch.multinomial(probs, num_samples=1).item())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with the Project KV1 2-bit packed-cache model.")
    parser.add_argument("--model", default="./BananaMind-KV1-8M-2Bit-Experimental")
    parser.add_argument("--prompt", default="Project KV1 is")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--show-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    load_dtype = dtype if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=load_dtype,
    ).to(args.device)
    model.eval()

    enc = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[:, -args.max_input_tokens :].to(args.device)
    generated = input_ids[0].tolist()
    temperature = 0.0 if args.greedy else args.temperature

    autocast_enabled = args.device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        outputs = model(input_ids=input_ids, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        if args.show_cache and past:
            k_pack, k_min, k_scale, v_pack, v_min, v_scale = past[0]
            print(
                "cache:",
                f"k_pack={tuple(k_pack.shape)} {k_pack.dtype}",
                f"v_pack={tuple(v_pack.shape)} {v_pack.dtype}",
                f"meta={k_min.dtype}/{k_scale.dtype}",
            )

        for _ in range(args.max_new_tokens):
            next_logits = logits[0].float()
            next_logits = apply_repetition_penalty(next_logits, generated, args.repeat_penalty)
            next_id = sample_next(next_logits, temperature, args.top_p)
            generated.append(next_id)

            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break

            step_ids = torch.tensor([[next_id]], dtype=torch.long, device=args.device)
            outputs = model(input_ids=step_ids, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    print(tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False))


if __name__ == "__main__":
    main()
