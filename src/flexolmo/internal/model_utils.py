from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig


def olmo2_tiny(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
    return cls.llama_like(
        d_model=32,
        hidden_size_multiplier=1.5,
        n_layers=kwargs.pop("n_layers", 1),
        n_heads=kwargs.pop("n_heads", 1),
        vocab_size=vocab_size,
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=1e-6,
        **kwargs,
    )


TransformerConfig.olmo2_tiny = classmethod(olmo2_tiny)  # type: ignore
