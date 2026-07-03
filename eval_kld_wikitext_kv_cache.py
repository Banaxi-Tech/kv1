#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from infer_kv1_packed import cache_summary
from modeling_kv1 import KV1ForCausalLM, apply_rope, repeat_kv


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "checkpoints" / "BananaMind-KV1-v0.1-8M-Sample10BT-2bit" / "latest"


def parse_int(value):
    return int(str(value).replace("_", ""))


def set_kv_cache_bits(model, bits):
    model.config.kv_cache_bits = bits
    for layer in model.layers:
        layer.self_attn.kv_cache_bits = bits


def load_wikitext_text(args):
    if args.local_text:
        path = Path(args.local_text)
        if not path.exists():
            raise RuntimeError(f"Missing local text file: {path}")
        return path.read_text(encoding="utf-8")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install datasets") from exc

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = []
    for row in ds:
        text = row.get("text", "")
        if text and text.strip():
            texts.append(text)
    return "\n\n".join(texts)


def make_token_chunks(tokenizer, text, seq_len, stride, max_tokens):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if max_tokens > 0:
        ids = ids[:max_tokens]
    if len(ids) < 2:
        raise RuntimeError(f"Need at least 2 tokens, got {len(ids)}")

    chunks = []
    start = 0
    while start + 2 <= len(ids):
        chunk = ids[start : start + seq_len]
        if len(chunk) < 2:
            break
        chunks.append(torch.tensor(chunk, dtype=torch.long))
        if start + seq_len >= len(ids):
            break
        start += stride
    return chunks, len(ids)


def mean_kld_for_logits(ref_logits, quant_logits, include_last_logit):
    if not include_last_logit:
        ref_logits = ref_logits[:, :-1, :]
        quant_logits = quant_logits[:, :-1, :]

    ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
    quant_logp = F.log_softmax(quant_logits.float(), dim=-1)
    ref_p = ref_logp.exp()
    kl = (ref_p * (ref_logp - quant_logp)).sum(dim=-1)
    return kl.sum().item(), kl.numel(), kl.mean().item(), kl.max().item()


def tensor_bytes(value):
    return value.numel() * value.element_size() if torch.is_tensor(value) else 0


def cache_bytes(past_key_values):
    if past_key_values is None:
        return 0
    return sum(tensor_bytes(item) for layer in past_key_values for item in layer)


def quantize_for_attention(x, bits, threshold, eps):
    if bits == 15:
        signs = torch.where(x >= threshold, torch.ones_like(x), -torch.ones_like(x))
        scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
        scale = scale.to(torch.float16).to(dtype=x.dtype)
        return signs * scale

    if bits == 1:
        return (x >= threshold).to(dtype=x.dtype)

    qmax = (1 << bits) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / qmax).clamp_min(eps)
    codes = ((x - x_min) / scale).round().clamp(0, qmax)

    # Inference stores affine metadata in fp16, so match that dequant path.
    x_min = x_min.to(torch.float16).to(dtype=x.dtype)
    scale = scale.to(torch.float16).to(dtype=x.dtype)
    return codes.to(dtype=x.dtype) * scale + x_min


def quantized_attention(attn, x, bits):
    bsz, seq_len, _ = x.shape
    q = attn.q_proj(x).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

    q, k = apply_rope(q, k, attn.rope_theta, start_pos=0)
    k_attn = quantize_for_attention(k, bits, attn.kv_quant_threshold, attn.kv_quant_eps)
    v_attn = quantize_for_attention(v, bits, attn.kv_quant_threshold, attn.kv_quant_eps)

    repeats = attn.num_heads // attn.num_kv_heads
    k_attn = repeat_kv(k_attn, repeats)
    v_attn = repeat_kv(v_attn, repeats)

    y = F.scaled_dot_product_attention(
        q,
        k_attn,
        v_attn,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=seq_len > 1,
    )
    y = y.transpose(1, 2).contiguous().view(bsz, seq_len, attn.hidden_size)
    return attn.o_proj(y)


def quantized_forward(model, input_ids, bits):
    x = model.embed_tokens(input_ids)
    for layer in model.layers:
        attn_out = quantized_attention(layer.self_attn, layer.input_layernorm(x), bits)
        x = x + attn_out
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    x = model.norm(x)
    return model.lm_head(x)


def estimate_packed_cache_bytes(model, seq_len, batch_size, bits):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    vectors = (
        model.config.num_hidden_layers
        * 2
        * batch_size
        * model.config.num_key_value_heads
        * seq_len
    )
    if bits == 15:
        code_bytes_per_vector = (head_dim + 7) // 8
        metadata_bytes_per_vector = 2
        return vectors * (code_bytes_per_vector + metadata_bytes_per_vector)

    code_bytes_per_vector = (head_dim * bits + 7) // 8
    metadata_bytes_per_vector = 0 if bits == 1 else 4
    return vectors * (code_bytes_per_vector + metadata_bytes_per_vector)


def packed_cache_summary(model, seq_len, batch_size, bits, actual_bytes):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    effective_bits = 1 if bits == 15 else bits
    packed_dim = (head_dim * effective_bits + 7) // 8
    fp16_equiv = (
        model.config.num_hidden_layers
        * 2
        * batch_size
        * model.config.num_key_value_heads
        * seq_len
        * head_dim
        * 2
    )
    shrink = fp16_equiv / max(1, actual_bytes)
    if bits == 15:
        name = "1.5-bit scaled-binary"
        meta = " + fp16 scale metadata"
    else:
        name = f"{bits}-bit"
        meta = "" if bits == 1 else " + fp16 min/scale metadata"
    return (
        f"estimated packed {name} KV cache{meta} | "
        f"k_pack={(batch_size, model.config.num_key_value_heads, seq_len, packed_dim)} uint8 | "
        f"v_pack={(batch_size, model.config.num_key_value_heads, seq_len, packed_dim)} uint8 | "
        f"stored={actual_bytes / (1024 ** 2):.3f} MiB | "
        f"fp16_equiv={fp16_equiv / (1024 ** 2):.3f} MiB | "
        f"shrink={shrink:.2f}x"
    )


def evaluate(args):
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

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            print("bf16 requested but unsupported; using fp16")
            dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = KV1ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    text = load_wikitext_text(args)
    chunks, loaded_tokens = make_token_chunks(
        tokenizer,
        text,
        seq_len=args.seq_len,
        stride=args.stride,
        max_tokens=args.max_tokens,
    )

    total_kl = 0.0
    total_positions = 0
    max_position_kl = 0.0
    max_chunk_mean = 0.0
    total_ref_cache_bytes = 0
    total_quant_cache_bytes = 0
    first_ref_cache = None
    first_quant_cache = None

    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    t0 = time.time()

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        for idx, chunk in enumerate(chunks, start=1):
            input_ids = chunk.unsqueeze(0).to(device)

            set_kv_cache_bits(model, 0)
            ref_out = model(input_ids=input_ids, use_cache=True)
            ref_logits = ref_out.logits
            ref_past = ref_out.past_key_values

            quant_logits = quantized_forward(model, input_ids=input_ids, bits=args.quant_kv_cache_bits)

            kl_sum, positions, chunk_mean, chunk_max = mean_kld_for_logits(
                ref_logits,
                quant_logits,
                include_last_logit=args.include_last_logit,
            )
            total_kl += kl_sum
            total_positions += positions
            max_position_kl = max(max_position_kl, chunk_max)
            max_chunk_mean = max(max_chunk_mean, chunk_mean)
            total_ref_cache_bytes += cache_bytes(ref_past)
            quant_cache_bytes = estimate_packed_cache_bytes(
                model,
                seq_len=input_ids.size(1),
                batch_size=input_ids.size(0),
                bits=args.quant_kv_cache_bits,
            )
            total_quant_cache_bytes += quant_cache_bytes

            if first_ref_cache is None:
                first_ref_cache = ref_past
                first_quant_cache = (input_ids.size(1), input_ids.size(0), quant_cache_bytes)

            if args.progress_every and (idx % args.progress_every == 0 or idx == len(chunks)):
                mean_nats = total_kl / max(1, total_positions)
                elapsed = time.time() - t0
                print(
                    f"chunk {idx:>5}/{len(chunks)} | "
                    f"positions {total_positions:,} | "
                    f"mean_kld {mean_nats:.8f} nats ({mean_nats / math.log(2):.8f} bits) | "
                    f"{total_positions / max(elapsed, 1e-9):,.0f} tok/s",
                    flush=True,
                )

    mean_nats = total_kl / max(1, total_positions)
    mean_bits = mean_nats / math.log(2)
    cache_ratio = total_ref_cache_bytes / max(1, total_quant_cache_bytes)

    result = {
        "metric": "mean_kld",
        "definition": (
            "mean over evaluated token positions of "
            f"KL(P_16bit_KV || P_{args.quant_kv_cache_bits_label}_packed_KV)"
        ),
        "quant_kv_cache_bits": args.quant_kv_cache_bits,
        "kld_nats_per_token": mean_nats,
        "kld_bits_per_token": mean_bits,
        "positions": total_positions,
        "chunks": len(chunks),
        "loaded_tokens": loaded_tokens,
        "dataset": args.dataset if not args.local_text else None,
        "dataset_config": args.dataset_config if not args.local_text else None,
        "split": args.split if not args.local_text else None,
        "local_text": args.local_text or None,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "include_last_logit": args.include_last_logit,
        "model": str(model_path),
        "dtype": str(dtype if device == "cuda" else torch.float32),
        "device": device,
        "max_position_kld_nats": max_position_kl,
        "max_chunk_mean_kld_nats": max_chunk_mean,
        "avg_ref_cache_bytes_per_chunk": total_ref_cache_bytes / max(1, len(chunks)),
        "avg_quant_cache_bytes_per_chunk": total_quant_cache_bytes / max(1, len(chunks)),
        "avg_cache_shrink_vs_fp16": cache_ratio,
    }

    print("\nMean KLD result")
    print(f"KLD:        {mean_nats:.10f} nats/token")
    print(f"KLD:        {mean_bits:.10f} bits/token")
    print(f"positions:  {total_positions:,}")
    print(f"chunks:     {len(chunks):,}")
    print(f"cache avg:  {cache_ratio:.2f}x smaller than FP16 baseline")
    if first_ref_cache is not None and first_quant_cache is not None:
        set_kv_cache_bits(model, 0)
        print(f"baseline:   {cache_summary(first_ref_cache, model)}")
        seq_len, batch_size, quant_cache_bytes = first_quant_cache
        print(
            f"{args.quant_kv_cache_bits_label}:      "
            f"{packed_cache_summary(model, seq_len, batch_size, args.quant_kv_cache_bits, quant_cache_bytes)}"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote:      {out}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute llama.cpp-style mean KLD of packed KV cache against 16-bit KV cache on Wikitext."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local-text", default="", help="Optional local text file instead of loading Wikitext from HF.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-tokens", type=parse_int, default=0, help="0 means all loaded Wikitext tokens.")
    parser.add_argument("--include-last-logit", action="store_true")
    parser.add_argument(
        "--quant-kv-cache-bits",
        type=int,
        choices=list(range(1, 9)) + [15],
        default=2,
        help="Use 15 for scaled 1-bit mode: sign bit plus one fp16 scale per vector.",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.quant_kv_cache_bits_label = "1.5bit" if args.quant_kv_cache_bits == 15 else f"{args.quant_kv_cache_bits}bit"
    evaluate(args)


if __name__ == "__main__":
    main()
