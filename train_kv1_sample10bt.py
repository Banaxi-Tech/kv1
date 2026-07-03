#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, GenerationConfig

from configuration_kv1 import KV1Config
from modeling_kv1 import KV1ForCausalLM


ROOT = Path(__file__).resolve().parent
BANANA_ROOT = ROOT.parent / "MiniBananaMind-v2-8M"

DEFAULT_DATA = BANANA_ROOT / "fineweb_edu_10BT_8k_digits.uint16.bin"
DEFAULT_TOKENIZER = BANANA_ROOT / "tokenizer_8k_digits"
DEFAULT_OUT = ROOT / "checkpoints" / "BananaMind-KV1-v0.1-8M-Sample10BT"


def parse_int(value):
    return int(str(value).replace("_", ""))


def parse_kv_cache_bits(value):
    text = str(value).strip().lower()
    if text in {"1.5", "1p5", "1_5", "15"}:
        return 15
    bits = int(text)
    if bits not in (0, 1, 2):
        raise argparse.ArgumentTypeError("kv cache bits must be one of 0, 1, 1.5, 2, or 15")
    return bits


def get_lr(step, max_steps, lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def sample_sequences(data, split_start, split_end, batch_size, seq_len):
    high = split_end - seq_len - 1
    if high <= split_start:
        raise RuntimeError("Dataset split too small for seq_len.")
    starts = np.random.randint(split_start, high, size=(batch_size,))
    return np.stack([data[i : i + seq_len] for i in starts]).astype(np.int64)


def tensor_to_device(x, device):
    x = torch.from_numpy(x)
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True)
    return x.to(device)


def make_batch(data, split_start, split_end, batch_size, seq_len, device):
    return tensor_to_device(sample_sequences(data, split_start, split_end, batch_size, seq_len), device)


def cudagraph_mark_step_begin(device):
    if device != "cuda":
        return
    compiler = getattr(torch, "compiler", None)
    mark_step = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None
    if mark_step is not None:
        mark_step()


@torch.no_grad()
def estimate_loss(model, data, train_end, seq_len, batch_size, device, eval_batches, dtype):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x = make_batch(data, train_end, len(data), batch_size, seq_len, device)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
            cudagraph_mark_step_begin(device)
            out = model(input_ids=x, labels=x, use_cache=False)
        losses.append(out.loss.item())
    model.train()
    loss = sum(losses) / len(losses)
    return loss, math.exp(min(20, loss))


def make_optimizer(model, lr, weight_decay, device):
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
            fused=(device == "cuda"),
        )
    except TypeError:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def make_config(args, tokenizer):
    config = KV1Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.seq_len,
        attention_dropout=args.attention_dropout,
        tie_word_embeddings=True,
        kv_cache_bits=args.kv_cache_bits,
        kv_quant_threshold=args.kv_quant_threshold,
        kv_quant_eps=args.kv_quant_eps,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1,
    )
    config.auto_map = {
        "AutoConfig": "configuration_kv1.KV1Config",
        "AutoModelForCausalLM": "modeling_kv1.KV1ForCausalLM",
    }
    return config


def build_fresh_model(args, tokenizer, device):
    return KV1ForCausalLM(make_config(args, tokenizer)).to(device)


def serializable_args(args):
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def save_checkpoint(model, tokenizer, optimizer, out_dir, step, args):
    out_dir = Path(out_dir)
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt))
    shutil.copyfile(ROOT / "configuration_kv1.py", ckpt / "configuration_kv1.py")
    shutil.copyfile(ROOT / "modeling_kv1.py", ckpt / "modeling_kv1.py")

    GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=128,
    ).save_pretrained(str(ckpt))

    train_args = serializable_args(args)
    with open(ckpt / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(train_args, f, indent=2)

    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "train_args": train_args,
        },
        ckpt / "training_state.pt",
    )

    latest = out_dir / "latest"
    tmp_latest = out_dir / "latest.tmp"
    if tmp_latest.exists():
        shutil.rmtree(tmp_latest)
    shutil.copytree(ckpt, tmp_latest)
    if latest.exists():
        if latest.is_dir():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    os.replace(tmp_latest, latest)
    print(f"saved:  {ckpt}")
    print(f"latest: {latest}")


def load_resume(args, device):
    resume_path = Path(args.out) / "latest" if args.resume == "latest" else Path(args.resume)
    if not resume_path.exists():
        raise RuntimeError(f"Resume checkpoint does not exist: {resume_path}")
    state_path = resume_path / "training_state.pt"
    if not state_path.exists():
        raise RuntimeError(f"No training_state.pt in checkpoint: {state_path}")
    model = KV1ForCausalLM.from_pretrained(str(resume_path)).to(device)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return model, state, resume_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train BananaMind-KV1 8M on Sample 10BT with normal, 1-bit, or 2-bit KV attention.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=parse_int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-tokens", type=parse_int, default=10_000_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", dest="compile", action="store_true", default=False)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--resume", default="", help="Use 'latest' or a checkpoint folder.")

    parser.add_argument(
        "--kv-cache-bits",
        type=parse_kv_cache_bits,
        choices=[0, 1, 2, 15],
        default=1,
        help="0=normal FP KV, 1=0/1 binary, 1.5 or 15=scaled sign bit, 2=affine 2-bit.",
    )
    parser.add_argument("--kv-quant-threshold", type=float, default=0.0)
    parser.add_argument("--kv-quant-eps", type=float, default=1e-6)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.data = args.data.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.out = args.out.resolve()

    if not args.data.exists():
        raise RuntimeError(f"Missing token bin: {args.data}")
    if not args.tokenizer.exists():
        raise RuntimeError(f"Missing tokenizer folder: {args.tokenizer}")

    device = args.device
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    train_end = len(data) - args.val_tokens
    if train_end <= args.seq_len + 1:
        raise RuntimeError("Dataset split too small for seq_len.")

    if args.resume:
        model, state, resume_path = load_resume(args, device)
        optimizer = make_optimizer(model, args.lr, args.weight_decay, device)
        optimizer.load_state_dict(state["optimizer"])
        optimizer_to_device(optimizer, device)
        start_step = int(state["step"])
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        print(f"resuming from: {resume_path}")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build_fresh_model(args, tokenizer, device)
        optimizer = make_optimizer(model, args.lr, args.weight_decay, device)
        start_step = 0

    params = count_params(model)
    if params >= 10_000_000:
        raise RuntimeError(f"Model is >=10M params: {params:,}")

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    if args.max_steps <= 0:
        args.max_steps = train_end // tokens_per_step

    print("BananaMind-KV1 training")
    print(f"data:             {args.data}")
    print(f"tokenizer:        {args.tokenizer}")
    print(f"out:              {args.out}")
    print(f"device:           {device}")
    print(f"amp dtype:        {dtype if device == 'cuda' else 'fp32'}")
    print(f"params:           {params:,}")
    print(f"kv_cache_bits:    {model.config.kv_cache_bits}")
    print(f"tokens:           {len(data):,} ({train_end:,} train / {args.val_tokens:,} val)")
    print(f"seq_len:          {args.seq_len}")
    print(f"batch_size:       {args.batch_size}")
    print(f"grad_accum:       {args.grad_accum}")
    print(f"tokens/step:      {tokens_per_step:,}")
    print(f"max_steps:        {args.max_steps:,}")
    print(f"start_step:       {start_step:,}")
    print(f"target tokens:    {tokens_per_step * args.max_steps:,}")

    if args.compile and device == "cuda":
        print(f"compiling model with mode={args.compile_mode}")
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except TypeError:
            model = torch.compile(model)

    model.train()
    t0 = time.time()

    for step in range(start_step + 1, args.max_steps + 1):
        lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(args.grad_accum):
            x = make_batch(data, 0, train_end, args.batch_size, args.seq_len, device)
            cudagraph_mark_step_begin(device)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                out = model(input_ids=x, labels=x, use_cache=False)
                loss = out.loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()

        if args.grad_clip > 0:
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 10 == 0 or step == start_step + 1:
            elapsed = time.time() - t0
            tok_seen = (step - start_step) * tokens_per_step
            toks_per_sec = tok_seen / max(1e-9, elapsed)
            print(
                f"step {step:>6}/{args.max_steps} | "
                f"loss {total_loss:.4f} | "
                f"ppl {math.exp(min(20, total_loss)):.2f} | "
                f"lr {lr:.2e} | "
                f"{toks_per_sec:,.0f} tok/s"
            )

        if args.eval_every and step % args.eval_every == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            val_loss, val_ppl = estimate_loss(
                raw_model,
                data,
                train_end,
                args.seq_len,
                args.batch_size,
                device,
                args.eval_batches,
                dtype,
            )
            print(f"eval step {step} | sample10bt val loss {val_loss:.4f} ppl {val_ppl:.2f}")

        if args.save_every and step % args.save_every == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(raw_model, tokenizer, optimizer, args.out, step, args)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_checkpoint(raw_model, tokenizer, optimizer, args.out, args.max_steps, args)


if __name__ == "__main__":
    main()
