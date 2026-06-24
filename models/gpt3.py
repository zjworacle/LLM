"""GPT-3 style decoder-only transformer (learned positions, MHA, GELU, LayerNorm).

This is the "classic" GPT architecture (GPT-2/GPT-3 family):

* token embedding + **learned** absolute position embedding
* pre-norm transformer blocks using **LayerNorm**
* **multi-head** self-attention (no GQA, no RoPE)
* **GELU** MLP with 4x expansion
* final LayerNorm, then a linear unembedding projection (weights tied to the token embedding)

The architecture-agnostic pieces live in llm.models.common; this file only wires
them together and owns the embeddings + unembedding.

Shape notation: batch_size, seq_length, model_dim, vocab_size. Input
token ids are (batch_size, seq_length) integers; after embedding the residual
stream is (batch_size, seq_length, model_dim) floats; the unembedding produces
(batch_size, seq_length, vocab_size) logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import (
    Attention,
    GeluMLP,
    KVCache,
    TransformerLayer,
    build_norm,
    init_weights,
)


@dataclass
class GPT3Config:
    """Hyper-parameters for a GPT-3 style model.

    The defaults describe a *tiny* model that trains on a laptop / Mac. Larger run configs
    live in llm.runconfigs.

    # CUSTOMIZE: every field here is a knob. The classic GPT-3 sizes scale
    # (n_layers, dim, n_heads) together, e.g. GPT-3 Small = (12, 768, 12).
    """

    vocab_size: int = 50257  # GPT-2/3 BPE vocab (tiktoken "gpt2"). Override per tokenizer.
    block_size: int = 256  # max context length (sequence length)
    n_layers: int = 4
    dim: int = 256
    n_heads: int = 4
    dropout: float = 0.0
    use_bias: bool = True  # GPT-3 uses biases in Linear/LayerNorm
    norm_eps: float = 1e-5
    tie_embeddings: bool = True  # reuse one (vocab_size, dim) matrix for token_emb + unembed
    init_std: float = 0.02


class GPT3(nn.Module):
    """A from-scratch GPT-3 style language model."""

    def __init__(self, config: GPT3Config, linear_cls: type[nn.Module] | None = None):
        super().__init__()
        self.config = config

        # --- Embeddings -----------------------------------------------------
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        # Learned absolute position embeddings (this is the GPT-specific choice).
        self.pos_emb = nn.Embedding(config.block_size, config.dim)
        self.drop = nn.Dropout(config.dropout)

        # --- Transformer layers --------------------------------------------
        layers = []
        for _ in range(config.n_layers):
            attn = Attention(
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_heads,  # MHA: kv heads == q heads
                use_bias=config.use_bias,
                use_rope=False,  # GPT-3 uses learned positions, not RoPE
                dropout=config.dropout,
                linear_cls=linear_cls,
            )
            mlp = GeluMLP(
                dim=config.dim,
                use_bias=config.use_bias,
                dropout=config.dropout,
                linear_cls=linear_cls,
            )
            attn_norm = build_norm("layernorm", config.dim, eps=config.norm_eps)
            mlp_norm = build_norm("layernorm", config.dim, eps=config.norm_eps)
            layers.append(TransformerLayer(attn_norm, attn, mlp_norm, mlp))
        self.layers = nn.ModuleList(layers)

        # --- Unembedding (output projection) --------------------------------
        self.final_norm = build_norm("layernorm", config.dim, eps=config.norm_eps)
        self.unembed = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Weight tying: the unembedding reuses the token embedding matrix.
            self.unembed.weight = self.token_emb.weight

        # Initialize all weights.
        self.apply(lambda m: init_weights(m, std=config.init_std))

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model.

        Args:
            idx: token ids, shape (batch, seq).
            targets: optional next-token targets, shape (batch, seq). If given, the
                cross-entropy loss is returned alongside the logits.
            caches: optional per-layer KV caches for incremental decoding.

        Returns:
            (logits, loss) where loss is None if targets is None.
        """
        _, n_tokens = idx.shape
        # Position offset accounts for tokens already in the cache during decoding.
        offset = caches[0].length if caches is not None else 0
        assert offset + n_tokens <= self.config.block_size, "sequence longer than block_size"

        pos = torch.arange(offset, offset + n_tokens, device=idx.device)
        # token_emb(idx): (batch_size, seq_length, model_dim);
        # pos_emb(pos): (seq_length, model_dim) -> (1, seq_length, model_dim) broadcast-added.
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]  # (batch_size, seq_length, model_dim)
        x = self.drop(x)  # (batch_size, seq_length, model_dim)

        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            # GPT-3 has no RoPE, so cos/sin stay None. Each layer preserves the shape
            # (batch_size, seq_length, model_dim).
            x = layer(x, cos=None, sin=None, cache=cache)

        x = self.final_norm(x)  # (batch_size, seq_length, model_dim)

        loss = None
        if targets is not None:
            logits = self.unembed(x)  # (batch_size, seq_length, vocab_size)
            # Flatten time/batch for cross-entropy. ignore_index=-1 lets callers mask
            # positions (used by SFT to ignore prompt tokens).
            #   logits view: (batch_size*seq_length, vocab_size); targets: (batch_size*seq_length,)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference fast path: only compute logits for the last position
            # -> (batch_size, 1, vocab_size).
            # IMPROVE: make this configurable if you need all-position logits at infer.
            logits = self.unembed(x[:, -1:, :])  # (batch_size, 1, vocab_size)
        return logits, loss

    # ------------------------------------------------------------------
    def init_caches(self, batch: int, device: torch.device, dtype: torch.dtype) -> list[KVCache]:
        """Allocate per-layer KV caches for generation."""
        head_dim = self.config.dim // self.config.n_heads
        return [
            KVCache.empty(
                batch,
                self.config.n_heads,
                self.config.block_size,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.config.n_layers)
        ]

    def num_params(self, non_embedding: bool = True) -> int:
        """Count parameters (handy for logging tiny-model size)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_embeddings:
            # Subtract the position embedding (token embedding is tied to the head).
            n -= self.pos_emb.weight.numel()
        return n
