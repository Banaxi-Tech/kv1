#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modeling_kv1 import KV1ForCausalLM, apply_rope, make_causal_mask, repeat_kv


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "checkpoints" / "BananaMind-KV1-v0.1-8M-Sample10BT-2bit" / "latest"


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


def pack_1bit(codes):
    original_dim = codes.size(-1)
    pad = (-original_dim) % 8
    if pad:
        codes = F.pad(codes, (0, pad))
    chunks = codes.view(*codes.shape[:-1], -1, 8).to(torch.int16)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int16, device=codes.device)
    return (chunks * weights).sum(dim=-1).to(torch.uint8), original_dim


def unpack_1bit(packed, original_dim):
    parts = [((packed // (1 << bit)) % 2).to(torch.uint8) for bit in range(8)]
    return torch.stack(parts, dim=-1).flatten(-2)[..., :original_dim]


def pack_2bit(codes):
    original_dim = codes.size(-1)
    pad = (-original_dim) % 4
    if pad:
        codes = F.pad(codes, (0, pad))
    chunks = codes.view(*codes.shape[:-1], -1, 4).to(torch.int16)
    weights = torch.tensor([1, 4, 16, 64], dtype=torch.int16, device=codes.device)
    return (chunks * weights).sum(dim=-1).to(torch.uint8), original_dim


def unpack_2bit(packed, original_dim):
    parts = [((packed // (1 << (2 * shift))) % 4).to(torch.uint8) for shift in range(4)]
    return torch.stack(parts, dim=-1).flatten(-2)[..., :original_dim]


def quantize_binary_packed(x, threshold):
    codes = (x >= threshold).to(torch.uint8)
    return pack_1bit(codes)


def dequantize_binary_packed(packed, original_dim, dtype):
    return unpack_1bit(packed, original_dim).to(dtype=dtype)


def quantize_scaled_binary_packed(x, threshold, eps):
    codes = (x >= threshold).to(torch.uint8)
    scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    packed, original_dim = pack_1bit(codes)
    return packed, scale.to(torch.float16), original_dim


def dequantize_scaled_binary_packed(packed, scale, original_dim, dtype):
    codes = unpack_1bit(packed, original_dim).bool()
    signs = torch.where(
        codes,
        torch.ones((), dtype=dtype, device=packed.device),
        -torch.ones((), dtype=dtype, device=packed.device),
    )
    return signs * scale.to(dtype=dtype)


def quantize_affine_packed_2bit(x, eps):
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / 3).clamp_min(eps)
    codes = ((x - x_min) / scale).round().clamp(0, 3).to(torch.uint8)
    packed, original_dim = pack_2bit(codes)
    return packed, x_min.to(torch.float16), scale.to(torch.float16), original_dim


def dequantize_affine_packed_2bit(packed, x_min, scale, original_dim, dtype):
    codes = unpack_2bit(packed, original_dim).to(dtype=dtype)
    return codes * scale.to(dtype=dtype) + x_min.to(dtype=dtype)


def cache_seq_len(past_key_value):
    if past_key_value is None:
        return 0
    return past_key_value[0].size(-2)


def packed_attention(attn, x, past_key_value, kv_cache_bits):
    bsz, seq_len, _ = x.shape
    past_len = cache_seq_len(past_key_value)

    q = attn.q_proj(x).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

    q, k = apply_rope(q, k, attn.rope_theta, start_pos=past_len)

    if kv_cache_bits == 0:
        if past_key_value is None:
            k_attn = k
            v_attn = v
        else:
            past_k, past_v = past_key_value
            k_attn = torch.cat((past_k, k), dim=-2)
            v_attn = torch.cat((past_v, v), dim=-2)
        present = (k_attn, v_attn)
    elif kv_cache_bits == 1:
        new_k_pack, head_dim = quantize_binary_packed(k, attn.kv_quant_threshold)
        new_v_pack, _ = quantize_binary_packed(v, attn.kv_quant_threshold)
        if past_key_value is None:
            k_pack = new_k_pack
            v_pack = new_v_pack
        else:
            past_k_pack, past_v_pack, head_dim = past_key_value
            k_pack = torch.cat((past_k_pack, new_k_pack), dim=-2)
            v_pack = torch.cat((past_v_pack, new_v_pack), dim=-2)
        present = (k_pack, v_pack, head_dim)
        k_attn = dequantize_binary_packed(k_pack, head_dim, q.dtype)
        v_attn = dequantize_binary_packed(v_pack, head_dim, q.dtype)
    elif kv_cache_bits == 15:
        new_k_pack, new_k_scale, head_dim = quantize_scaled_binary_packed(
            k,
            threshold=attn.kv_quant_threshold,
            eps=attn.kv_quant_eps,
        )
        new_v_pack, new_v_scale, _ = quantize_scaled_binary_packed(
            v,
            threshold=attn.kv_quant_threshold,
            eps=attn.kv_quant_eps,
        )
        if past_key_value is None:
            k_pack, k_scale = new_k_pack, new_k_scale
            v_pack, v_scale = new_v_pack, new_v_scale
        else:
            past_k_pack, past_k_scale, past_v_pack, past_v_scale, head_dim = past_key_value
            k_pack = torch.cat((past_k_pack, new_k_pack), dim=-2)
            k_scale = torch.cat((past_k_scale, new_k_scale), dim=-2)
            v_pack = torch.cat((past_v_pack, new_v_pack), dim=-2)
            v_scale = torch.cat((past_v_scale, new_v_scale), dim=-2)
        present = (k_pack, k_scale, v_pack, v_scale, head_dim)
        k_attn = dequantize_scaled_binary_packed(k_pack, k_scale, head_dim, q.dtype)
        v_attn = dequantize_scaled_binary_packed(v_pack, v_scale, head_dim, q.dtype)
    elif kv_cache_bits == 2:
        new_k_pack, new_k_min, new_k_scale, head_dim = quantize_affine_packed_2bit(
            k,
            eps=attn.kv_quant_eps,
        )
        new_v_pack, new_v_min, new_v_scale, _ = quantize_affine_packed_2bit(
            v,
            eps=attn.kv_quant_eps,
        )
        if past_key_value is None:
            k_pack, k_min, k_scale = new_k_pack, new_k_min, new_k_scale
            v_pack, v_min, v_scale = new_v_pack, new_v_min, new_v_scale
        else:
            (
                past_k_pack,
                past_k_min,
                past_k_scale,
                past_v_pack,
                past_v_min,
                past_v_scale,
                head_dim,
            ) = past_key_value
            k_pack = torch.cat((past_k_pack, new_k_pack), dim=-2)
            k_min = torch.cat((past_k_min, new_k_min), dim=-2)
            k_scale = torch.cat((past_k_scale, new_k_scale), dim=-2)
            v_pack = torch.cat((past_v_pack, new_v_pack), dim=-2)
            v_min = torch.cat((past_v_min, new_v_min), dim=-2)
            v_scale = torch.cat((past_v_scale, new_v_scale), dim=-2)
        present = (k_pack, k_min, k_scale, v_pack, v_min, v_scale, head_dim)
        k_attn = dequantize_affine_packed_2bit(k_pack, k_min, k_scale, head_dim, q.dtype)
        v_attn = dequantize_affine_packed_2bit(v_pack, v_min, v_scale, head_dim, q.dtype)
    else:
        raise ValueError(f"Unsupported kv_cache_bits={kv_cache_bits}")

    repeats = attn.num_heads // attn.num_kv_heads
    k_attn = repeat_kv(k_attn, repeats)
    v_attn = repeat_kv(v_attn, repeats)

    key_len = k_attn.size(-2)
    if past_len == 0:
        attn_mask = None
        is_causal = seq_len > 1
    elif seq_len == 1:
        attn_mask = None
        is_causal = False
    else:
        attn_mask = make_causal_mask(seq_len, key_len, past_len, x.device)
        is_causal = False

    y = F.scaled_dot_product_attention(
        q,
        k_attn,
        v_attn,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    y = y.transpose(1, 2).contiguous().view(bsz, seq_len, attn.hidden_size)
    return attn.o_proj(y), present


def packed_forward(model, input_ids, past_key_values=None):
    if past_key_values is None:
        past_key_values = [None] * len(model.layers)
    elif len(past_key_values) != len(model.layers):
        raise ValueError("past_key_values length must match number of layers")

    x = model.embed_tokens(input_ids)
    presents = []
    for layer, past in zip(model.layers, past_key_values):
        attn_out, present = packed_attention(
            layer.self_attn,
            layer.input_layernorm(x),
            past,
            model.config.kv_cache_bits,
        )
        x = x + attn_out
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        presents.append(present)

    x = model.norm(x)
    logits = model.lm_head(x)
    return logits, tuple(presents)


def tensor_bytes(value):
    return value.numel() * value.element_size() if torch.is_tensor(value) else 0


def format_mib(num_bytes):
    return f"{num_bytes / (1024 ** 2):.3f} MiB"


def cache_summary(past_key_values, model):
    if not past_key_values:
        return "empty"

    kv_cache_bits = model.config.kv_cache_bits
    first = past_key_values[0]
    actual_bytes = sum(tensor_bytes(item) for layer in past_key_values for item in layer)

    if kv_cache_bits == 0:
        k, v = first
        baseline_bytes = actual_bytes
        return (
            "normal floating KV cache | "
            f"k={tuple(k.shape)} {k.dtype} | "
            f"v={tuple(v.shape)} {v.dtype} | "
            f"stored={format_mib(actual_bytes)}"
        )

    if kv_cache_bits == 1:
        k_pack, v_pack, head_dim = first
        bsz, kv_heads, seq_len, _ = k_pack.shape
        baseline_bytes = len(past_key_values) * 2 * bsz * kv_heads * seq_len * head_dim * 2
        shrink = baseline_bytes / max(1, actual_bytes)
        return (
            "packed 1-bit KV cache | "
            f"k_pack={tuple(k_pack.shape)} {k_pack.dtype} | "
            f"v_pack={tuple(v_pack.shape)} {v_pack.dtype} | "
            f"stored={format_mib(actual_bytes)} | "
            f"fp16_equiv={format_mib(baseline_bytes)} | "
            f"shrink={shrink:.2f}x"
        )

    if kv_cache_bits == 15:
        k_pack, k_scale, v_pack, v_scale, head_dim = first
        del v_scale
        bsz, kv_heads, seq_len, _ = k_pack.shape
        baseline_bytes = len(past_key_values) * 2 * bsz * kv_heads * seq_len * head_dim * 2
        shrink = baseline_bytes / max(1, actual_bytes)
        return (
            "packed 1.5-bit scaled-binary KV cache | "
            f"k_pack={tuple(k_pack.shape)} {k_pack.dtype}, "
            f"k_scale={k_scale.dtype}, "
            f"v_pack={tuple(v_pack.shape)} {v_pack.dtype}, "
            f"stored={format_mib(actual_bytes)} | "
            f"fp16_equiv={format_mib(baseline_bytes)} | "
            f"shrink={shrink:.2f}x"
        )

    k_pack, k_min, k_scale, v_pack, v_min, v_scale, head_dim = first
    del k_scale, v_min, v_scale
    bsz, kv_heads, seq_len, _ = k_pack.shape
    baseline_bytes = len(past_key_values) * 2 * bsz * kv_heads * seq_len * head_dim * 2
    shrink = baseline_bytes / max(1, actual_bytes)
    return (
        "packed 2-bit affine KV cache | "
        f"k_pack={tuple(k_pack.shape)} {k_pack.dtype} | "
        f"k_meta={k_min.dtype} | "
        f"v_pack={tuple(v_pack.shape)} {v_pack.dtype} | "
        f"stored={format_mib(actual_bytes)} | "
        f"fp16_equiv={format_mib(baseline_bytes)} | "
        f"shrink={shrink:.2f}x"
    )


def generate(model, tokenizer, prompt, args, device, dtype):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[:, -args.max_input_tokens :].to(device)
    generated = input_ids[0].tolist()

    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        logits, past = packed_forward(model, input_ids=input_ids, past_key_values=None)
        logits = logits[:, -1, :]

        if args.show_cache:
            print("prefill:", cache_summary(past, model))

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
            logits, past = packed_forward(model, input_ids=step_ids, past_key_values=past)
            logits = logits[:, -1, :]

        if args.show_cache:
            print("final:  ", cache_summary(past, model))

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
    parser = argparse.ArgumentParser(description="Infer with BananaMind-KV1 using bit-packed KV cache storage.")
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
