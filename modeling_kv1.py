import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

if __package__:
    from .configuration_kv1 import KV1Config
else:
    KV1Config = importlib.import_module("configuration_kv1").KV1Config


class KV1RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def fake_quantize_binary_ste(x, threshold=0.0):
    hard = (x >= threshold).to(dtype=x.dtype)
    return x + (hard - x).detach()


def fake_quantize_scaled_binary_ste(x, threshold=0.0, eps=1e-6):
    signs = torch.where(x >= threshold, torch.ones_like(x), -torch.ones_like(x))
    scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    scale = scale.to(torch.float16).to(dtype=x.dtype)
    dequant = signs * scale
    return x + (dequant - x).detach()


def quantize_binary_bool(x, threshold=0.0):
    return x >= threshold


def dequantize_binary_bool(x, dtype):
    return x.to(dtype=dtype)


def quantize_scaled_binary(x, threshold=0.0, eps=1e-6):
    codes = x >= threshold
    scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    return codes, scale.to(torch.float16)


def dequantize_scaled_binary(codes, scale, dtype):
    signs = torch.where(
        codes,
        torch.ones((), device=codes.device, dtype=dtype),
        -torch.ones((), device=codes.device, dtype=dtype),
    )
    return signs * scale.to(dtype=dtype)


def fake_quantize_affine_ste(x, bits=2, eps=1e-6):
    qmax = (1 << bits) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / qmax).clamp_min(eps)
    codes = ((x - x_min) / scale).round().clamp(0, qmax)
    dequant = codes * scale + x_min
    return x + (dequant - x).detach()


def quantize_affine_codes(x, bits=2, eps=1e-6):
    qmax = (1 << bits) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / qmax).clamp_min(eps)
    codes = ((x - x_min) / scale).round().clamp(0, qmax).to(torch.uint8)
    return codes, x_min.to(torch.float16), scale.to(torch.float16)


def dequantize_affine_codes(codes, x_min, scale, dtype):
    return codes.to(dtype=dtype) * scale.to(dtype=dtype) + x_min.to(dtype=dtype)


def apply_rope(q, k, rope_theta, start_pos=0):
    dtype = q.dtype
    device = q.device
    seq_len = q.size(-2)
    head_dim = q.size(-1)

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(start_pos, start_pos + seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()[None, None, :, :].to(dtype)
    sin = freqs.sin()[None, None, :, :].to(dtype)

    def rotate(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def make_causal_mask(query_len, key_len, past_len, device):
    q_pos = torch.arange(past_len, past_len + query_len, device=device)[:, None]
    k_pos = torch.arange(key_len, device=device)[None, :]
    return (k_pos <= q_pos)[None, None, :, :]


class KV1Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.kv_cache_bits = config.kv_cache_bits
        self.kv_quant_threshold = config.kv_quant_threshold
        self.kv_quant_eps = config.kv_quant_eps

        if config.kv_cache_bits not in (0, 1, 2, 15):
            raise ValueError("KV1 supports kv_cache_bits=0, 1, 2, or 15")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, past_key_value=None, use_cache=False):
        bsz, seq_len, _ = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, self.rope_theta, start_pos=past_len)

        present = None
        if use_cache:
            if self.kv_cache_bits == 0:
                if past_key_value is None:
                    k_attn = k
                    v_attn = v
                else:
                    past_k, past_v = past_key_value
                    if past_k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        raise TypeError("Normal KV inference requires floating-point KV cache tensors.")
                    k_attn = torch.cat((past_k, k), dim=-2)
                    v_attn = torch.cat((past_v, v), dim=-2)
                present = (k_attn, v_attn)
            elif self.kv_cache_bits == 1:
                new_k_codes = quantize_binary_bool(k, self.kv_quant_threshold)
                new_v_codes = quantize_binary_bool(v, self.kv_quant_threshold)

                if past_key_value is None:
                    k_codes = new_k_codes
                    v_codes = new_v_codes
                else:
                    past_k_codes, past_v_codes = past_key_value
                    if past_k_codes.dtype != torch.bool or past_v_codes.dtype != torch.bool:
                        raise TypeError("KV1 1-bit inference requires bool KV cache tensors.")
                    k_codes = torch.cat((past_k_codes, new_k_codes), dim=-2)
                    v_codes = torch.cat((past_v_codes, new_v_codes), dim=-2)

                present = (k_codes, v_codes)
                k_attn = dequantize_binary_bool(k_codes, q.dtype)
                v_attn = dequantize_binary_bool(v_codes, q.dtype)
            elif self.kv_cache_bits == 15:
                new_k_codes, new_k_scale = quantize_scaled_binary(
                    k,
                    threshold=self.kv_quant_threshold,
                    eps=self.kv_quant_eps,
                )
                new_v_codes, new_v_scale = quantize_scaled_binary(
                    v,
                    threshold=self.kv_quant_threshold,
                    eps=self.kv_quant_eps,
                )

                if past_key_value is None:
                    k_codes, k_scale = new_k_codes, new_k_scale
                    v_codes, v_scale = new_v_codes, new_v_scale
                else:
                    past_k_codes, past_k_scale, past_v_codes, past_v_scale = past_key_value
                    if past_k_codes.dtype != torch.bool or past_v_codes.dtype != torch.bool:
                        raise TypeError("KV1 1.5-bit inference requires bool KV cache code tensors.")
                    k_codes = torch.cat((past_k_codes, new_k_codes), dim=-2)
                    k_scale = torch.cat((past_k_scale, new_k_scale), dim=-2)
                    v_codes = torch.cat((past_v_codes, new_v_codes), dim=-2)
                    v_scale = torch.cat((past_v_scale, new_v_scale), dim=-2)

                present = (k_codes, k_scale, v_codes, v_scale)
                k_attn = dequantize_scaled_binary(k_codes, k_scale, q.dtype)
                v_attn = dequantize_scaled_binary(v_codes, v_scale, q.dtype)
            else:
                new_k_codes, new_k_min, new_k_scale = quantize_affine_codes(
                    k,
                    bits=2,
                    eps=self.kv_quant_eps,
                )
                new_v_codes, new_v_min, new_v_scale = quantize_affine_codes(
                    v,
                    bits=2,
                    eps=self.kv_quant_eps,
                )

                if past_key_value is None:
                    k_codes, k_min, k_scale = new_k_codes, new_k_min, new_k_scale
                    v_codes, v_min, v_scale = new_v_codes, new_v_min, new_v_scale
                else:
                    (
                        past_k_codes,
                        past_k_min,
                        past_k_scale,
                        past_v_codes,
                        past_v_min,
                        past_v_scale,
                    ) = past_key_value
                    if past_k_codes.dtype != torch.uint8 or past_v_codes.dtype != torch.uint8:
                        raise TypeError("KV1 2-bit inference requires uint8 KV cache code tensors.")
                    k_codes = torch.cat((past_k_codes, new_k_codes), dim=-2)
                    k_min = torch.cat((past_k_min, new_k_min), dim=-2)
                    k_scale = torch.cat((past_k_scale, new_k_scale), dim=-2)
                    v_codes = torch.cat((past_v_codes, new_v_codes), dim=-2)
                    v_min = torch.cat((past_v_min, new_v_min), dim=-2)
                    v_scale = torch.cat((past_v_scale, new_v_scale), dim=-2)

                present = (k_codes, k_min, k_scale, v_codes, v_min, v_scale)
                k_attn = dequantize_affine_codes(k_codes, k_min, k_scale, q.dtype)
                v_attn = dequantize_affine_codes(v_codes, v_min, v_scale, q.dtype)
        else:
            if self.kv_cache_bits == 0:
                k_attn = k
                v_attn = v
            elif self.kv_cache_bits == 1:
                k_attn = fake_quantize_binary_ste(k, self.kv_quant_threshold)
                v_attn = fake_quantize_binary_ste(v, self.kv_quant_threshold)
            elif self.kv_cache_bits == 15:
                k_attn = fake_quantize_scaled_binary_ste(
                    k,
                    threshold=self.kv_quant_threshold,
                    eps=self.kv_quant_eps,
                )
                v_attn = fake_quantize_scaled_binary_ste(
                    v,
                    threshold=self.kv_quant_threshold,
                    eps=self.kv_quant_eps,
                )
            else:
                k_attn = fake_quantize_affine_ste(k, bits=2, eps=self.kv_quant_eps)
                v_attn = fake_quantize_affine_ste(v, bits=2, eps=self.kv_quant_eps)

        repeats = self.num_heads // self.num_kv_heads
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
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(y), present


class KV1MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class KV1Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = KV1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = KV1Attention(config)
        self.post_attention_layernorm = KV1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = KV1MLP(config)

    def forward(self, x, past_key_value=None, use_cache=False):
        attn_out, present = self.self_attn(
            self.input_layernorm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class KV1PreTrainedModel(PreTrainedModel):
    config_class = KV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["KV1Block"]

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class KV1ForCausalLM(KV1PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([KV1Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = KV1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.post_init()

    def tie_weights(self, *args, **kwargs):
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        **kwargs,
    ):
        del attention_mask, kwargs

        if input_ids is None:
            raise ValueError("input_ids is required")

        x = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        elif len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values length must match number of layers")

        presents = [] if use_cache else None
        for layer, past in zip(self.layers, past_key_values):
            x, present = layer(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
        )

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if getattr(self.config, "tie_word_embeddings", True):
            for key in list(sd.keys()):
                if key == "lm_head.weight" or key.endswith(".lm_head.weight"):
                    del sd[key]
        return sd

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
