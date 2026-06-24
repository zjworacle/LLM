"""LLaMA-3 style decoder-only transformer (RoPE, GQA, SwiGLU, RMSNorm).

The modern LLaMA architecture differs from GPT-3 in four ways:

* **RoPE** rotary positional embeddings instead of learned absolute positions
* **grouped-query attention (GQA)**: fewer key/value heads than query heads
* **SwiGLU** gated MLP instead of GELU
* **RMSNorm** instead of LayerNorm, and no biases anywhere

As with GPT-3, the reusable pieces live in llm.models.common; this file wires 
them together, owns the token embedding + unembedding projection, and precomputes
the RoPE tables.

Shape notation: batch_size, seq_length, model_dim, head_dim,
max_seq_length (RoPE table length), vocab_size. Token ids enter as
(batch_size, seq_length); the residual stream is (batch_size, seq_length,
model_dim); RoPE tables are (max_seq_length, head_dim); logits are
(batch_size, seq_length, vocab_size).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import (
    Attention,
    KVCache,
    SwiGLUMLP,
    TransformerLayer,
    build_norm,
    init_weights,
    precompute_rope_cache,
)


@dataclass
class Llama3Config:
    """Hyper-parameters for a LLaMA-3 style model.

    Defaults describe a *tiny* model for laptop/Mac bring-up. Larger run configs (1B/3B/8B)
    live in llm.runconfigs.

    # CUSTOMIZE: knobs of interest:
    #   - n_kv_heads < n_heads enables GQA (smaller KV cache).
    #   - rope_theta = 500000 for LLaMA-3's long-context base (10000 = classic RoPE).
    #   - ffn_hidden=None uses LLaMA's 2/3 * 4 * dim heuristic.
    """

    vocab_size: int = 128256  # LLaMA-3 tokenizer vocab. Override to match your tokenizer.
    block_size: int = 256
    n_layers: int = 4
    dim: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2  # GQA: 2 kv heads shared across 4 q heads
    ffn_hidden: int | None = None  # None -> LLaMA heuristic
    ffn_multiple_of: int = 256
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False  # LLaMA-3 does NOT tie embeddings by default
    init_std: float = 0.02


class Llama3(nn.Module):
    """A from-scratch LLaMA-3 style language model."""

    def __init__(self, config: Llama3Config, linear_cls: type[nn.Module] | None = None):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)

        layers = []
        for _ in range(config.n_layers):
            attn = Attention(
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,  # GQA
                use_bias=False,  # LLaMA uses no attention biases
                use_rope=True,
                dropout=0.0,
                linear_cls=linear_cls,
            )
            mlp = SwiGLUMLP(
                dim=config.dim,
                hidden=config.ffn_hidden,
                multiple_of=config.ffn_multiple_of,
                use_bias=False,
                linear_cls=linear_cls,
            )
            attn_norm = build_norm("rmsnorm", config.dim, eps=config.norm_eps)
            mlp_norm = build_norm("rmsnorm", config.dim, eps=config.norm_eps)
            layers.append(TransformerLayer(attn_norm, attn, mlp_norm, mlp))
        self.layers = nn.ModuleList(layers)

        self.final_norm = build_norm("rmsnorm", config.dim, eps=config.norm_eps)
        self.unembed = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.unembed.weight = self.token_emb.weight

        # Precompute RoPE cos/sin tables once and register as buffers so they move with
        # .to(device) and are saved/restored with the model. Not parameters.
        head_dim = config.dim // config.n_heads
        cos, sin = precompute_rope_cache(head_dim, config.block_size, theta=config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(lambda m: init_weights(m, std=config.init_std))

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """See :meth:`llm.models.gpt3.GPT3.forward` — identical contract."""
        _, n_tokens = idx.shape
        offset = caches[0].length if caches is not None else 0
        assert offset + n_tokens <= self.config.block_size, "sequence longer than block_size"

        x = self.token_emb(idx)  # (batch_size, seq_length, model_dim)

        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            # Pass the RoPE tables (max_seq_length, head_dim); the attention module
            # slices them by position offset. Each layer preserves the shape
            # (batch_size, seq_length, model_dim).
            x = layer(x, cos=self.rope_cos, sin=self.rope_sin, cache=cache)

        x = self.final_norm(x)  # (batch_size, seq_length, model_dim)

        loss = None
        if targets is not None:
            logits = self.unembed(x)  # (batch_size, seq_length, vocab_size)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (batch_size*seq_length, vocab_size)
                targets.view(-1),  # (batch_size*seq_length,)
                ignore_index=-1,
            )
        else:
            logits = self.unembed(x[:, -1:, :])  # (batch_size, 1, vocab_size)
        return logits, loss

    # ------------------------------------------------------------------
    def init_caches(self, batch: int, device: torch.device, dtype: torch.dtype) -> list[KVCache]:
        head_dim = self.config.dim // self.config.n_heads
        return [
            KVCache.empty(
                batch,
                self.config.n_kv_heads,  # cache stores kv heads (fewer under GQA)
                self.config.block_size,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.config.n_layers)
        ]

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.tie_embeddings:
            n -= self.token_emb.weight.numel()
        return n
