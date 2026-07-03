from transformers.configuration_utils import PretrainedConfig


class KV1Config(PretrainedConfig):
    model_type = "banana_kv1"

    def __init__(
        self,
        vocab_size=8192,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        kv_cache_bits=1,
        kv_quant_threshold=0.0,
        kv_quant_eps=1e-6,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.kv_cache_bits = kv_cache_bits
        self.kv_quant_threshold = kv_quant_threshold
        self.kv_quant_eps = kv_quant_eps
