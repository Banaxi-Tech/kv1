# Project KV1

Project KV1 is an experimental code release for KV-cache-aware low-bit K/V training and inference.

The trained 8M checkpoint is hosted on Hugging Face:

https://huggingface.co/BananaMind/BananaMind-KV1-8M-2Bit-Experimental

This GitHub repo contains the Python code only. It does not include model weights, tokenizer files, datasets, checkpoints, or evaluation JSON artifacts.

## Files

- `configuration_kv1.py` - custom Transformers config for KV1.
- `modeling_kv1.py` - custom KV1 causal LM with 1-bit and 2-bit K/V paths.
- `train_kv1_sample10bt.py` - train KV1 on a local tokenized FineWeb-Edu Sample 10BT binary.
- `generate_kv1_2bit.py` - generate from the Hugging Face 2-bit KV-cache-aware checkpoint.
- `infer_kv1.py` - local inference helper for KV1 checkpoints.
- `infer_kv1_packed.py` - packed-cache inference helper and cache inspection utilities.
- `eval_kld_wikitext_kv_cache.py` - compare low-bit KV cache logits against a 16-bit KV-cache reference on WikiText-2.

## Install

```bash
pip install torch transformers tokenizers safetensors datasets numpy
```

## Generate

```bash
python generate_kv1_2bit.py \
  --model BananaMind/BananaMind-KV1-8M-2Bit-Experimental \
  --prompt "Project KV1 is" \
  --max-new-tokens 80 \
  --greedy
```

Use `--show-cache` to print the packed 2-bit KV-cache tensor shapes.

## Train

`train_kv1_sample10bt.py` expects a local tokenized `uint16` dataset and tokenizer path. Override the defaults with CLI flags if your files live somewhere else:

```bash
python train_kv1_sample10bt.py \
  --kv-cache-bits 2 \
  --data /path/to/fineweb_edu_10BT_8k_digits.uint16.bin \
  --tokenizer /path/to/tokenizer_8k_digits \
  --out checkpoints/kv1-8m-2bit
```

## Evaluate

```bash
python eval_kld_wikitext_kv_cache.py \
  --model checkpoints/kv1-8m-2bit/latest \
  --quant-kv-cache-bits 2
```

The reported KLD compares low-bit KV-cache logits against a 16-bit KV-cache reference.
