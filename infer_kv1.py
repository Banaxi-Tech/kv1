#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modeling_kv1 import KV1ForCausalLM


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "checkpoints" / "BananaMind-KV1-v0.1-8M-Sample10BT" / "latest"


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


def cache_summary(past_key_values, kv_cache_bits):
    if not past_key_values:
        return "empty"

    first = past_key_values[0]
    if kv_cache_bits == 0:
        k, v = first
        return f"normal floating KV cache | k={tuple(k.shape)} {k.dtype} | v={tuple(v.shape)} {v.dtype}"

    if kv_cache_bits == 1:
        k, v = first
        return f"1-bit bool KV cache | k={tuple(k.shape)} {k.dtype} | v={tuple(v.shape)} {v.dtype}"

    if kv_cache_bits == 15:
        k_codes, k_scale, v_codes, v_scale = first
        return (
            "1.5-bit scaled-binary KV cache | "
            f"k_codes={tuple(k_codes.shape)} {k_codes.dtype}, "
            f"k_scale={k_scale.dtype}, "
            f"v_codes={tuple(v_codes.shape)} {v_codes.dtype}, "
            f"v_scale={v_scale.dtype}"
        )

    k_codes, k_min, k_scale, v_codes, v_min, v_scale = first
    return (
        "2-bit uint8 affine KV cache | "
        f"k_codes={tuple(k_codes.shape)} {k_codes.dtype}, "
        f"k_meta={k_min.dtype}/{k_scale.dtype}, "
        f"v_codes={tuple(v_codes.shape)} {v_codes.dtype}, "
        f"v_meta={v_min.dtype}/{v_scale.dtype}"
    )


def generate(model, tokenizer, prompt, args, device, dtype):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[:, -args.max_input_tokens :].to(device)
    generated = input_ids[0].tolist()

    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        if args.show_cache:
            print(cache_summary(past, model.config.kv_cache_bits))

        new_ids = []
        for _ in range(args.max_new_tokens):
            next_logits = logits[0].float()
            next_logits = apply_repetition_penalty(next_logits, generated, args.repeat_penalty)
            next_id = sample_next(next_logits, args.temperature, args.top_p)
            generated.append(next_id)
            new_ids.append(next_id)

            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break

            step_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(input_ids=step_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def load(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise RuntimeError(f"Missing model folder: {model_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = KV1ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer, device, dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Infer with BananaMind-KV1 using quantized KV cache.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, device, dtype = load(args)

    print(f"model:          {args.model}")
    print(f"device:         {device}")
    print(f"dtype:          {dtype if device == 'cuda' else 'fp32'}")
    print(f"kv_cache_bits:  {model.config.kv_cache_bits}")

    if args.interactive:
        while True:
            prompt = input("\nPrompt> ")
            if not prompt.strip():
                break
            t0 = time.time()
            text = generate(model, tokenizer, prompt, args, device, dtype)
            print(text)
            print(f"[{time.time() - t0:.2f}s]")
        return

    prompt = args.prompt or "The capital of France is"
    t0 = time.time()
    print(generate(model, tokenizer, prompt, args, device, dtype))
    print(f"\n[{time.time() - t0:.2f}s]")


if __name__ == "__main__":
    main()
