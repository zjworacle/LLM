"""LLaMA-4 style decoder-only transformer (LLaMA-3 backbone + Mixture-of-Experts).

LLaMA-4 keeps the modern LLaMA-3 attention stack (RoPE + grouped-query attention +
RMSNorm, no biases) and swaps the dense SwiGLU MLP for a sparse **Mixture-of-Experts**
feed-forward: a learned router sends each token to its top-k experts, plus an always-on
shared expert. This grows model capacity (more parameters) while keeping the per-token
compute close to a single dense MLP.

Like the real LLaMA-4, MoE layers can be *interleaved* with dense layers via
``moe_interleave`` (every Nth layer is MoE; the rest stay dense SwiGLU). The MoE router
is trained with a load-balancing auxiliary loss, which this model folds into the
cross-entropy loss returned from forward.

Everything reusable lives in llm.models.common; this file only wires the pieces
together, owns the token embedding + unembedding, and precomputes the RoPE tables.

Shape notation: batch_size, seq_length, model_dim, head_dim, max_seq_length (RoPE
table length), vocab_size. Token ids enter as (batch_size, seq_length); the residual
stream is (batch_size, seq_length, model_dim); logits are
(batch_size, seq_length, vocab_size).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import (
    Attention,
    KVCache,
    MoEMLP,
    SwiGLUMLP,
    TransformerLayer,
    build_norm,
    collect_moe_aux_loss,
    init_weights,
    precompute_rope_cache,
)


@dataclass
class Llama4Config:
    """Hyper-parameters for a LLaMA-4 style MoE model.

    Defaults describe a *tiny* model for laptop/Mac bring-up. Larger run configs live
    in llm.runconfigs.

    # CUSTOMIZE: knobs of interest:
    #   - n_experts / top_k: total experts vs. how many each token uses.
    #   - n_shared_experts: always-on experts summed into every token (LLaMA-4 uses 1).
    #   - first_dense_layers: keep the first N layers dense (LLaMA-4 does this for the
    #     early, low-level layers where routing adds little and hurts stability).
    #   - moe_interleave: of the remaining layers, 1 = all MoE; 2 = every other MoE.
    #   - aux_loss_coef: weight of the router load-balancing loss.
    """

    vocab_size: int = 128256
    block_size: int = 256
    n_layers: int = 4
    dim: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2  # GQA
    # --- MoE feed-forward ---
    n_experts: int = 8
    top_k: int = 2
    n_shared_experts: int = 1
    first_dense_layers: int = 0  # keep the first N layers dense before MoE kicks in
    moe_interleave: int = 1  # of the MoE-eligible layers, 1 => every one is MoE
    aux_loss_coef: float = 0.01
    ffn_hidden: int | None = None  # None -> LLaMA heuristic (per expert)
    ffn_multiple_of: int = 256
    # --- shared with LLaMA-3 ---
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    init_std: float = 0.02


class Llama4(nn.Module):
    """A from-scratch LLaMA-4 style MoE language model."""

    def __init__(self, config: Llama4Config, linear_cls: type[nn.Module] | None = None):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)

        layers = []
        for layer_idx in range(config.n_layers):
            attn = Attention(
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,  # GQA
                use_bias=False,  # LLaMA uses no attention biases
                use_rope=True,
                dropout=0.0,
                linear_cls=linear_cls,
            )
            # Layer is MoE only after the first_dense_layers warm-up, and then only on
            # the moe_interleave stride. Early layers stay dense SwiGLU (LLaMA-4 keeps
            # the low-level layers dense; routing there adds little and hurts stability).
            past_dense = layer_idx >= config.first_dense_layers
            on_stride = (layer_idx - config.first_dense_layers) % config.moe_interleave == 0
            if past_dense and on_stride:
                mlp: nn.Module = MoEMLP(
                    dim=config.dim,
                    n_experts=config.n_experts,
                    top_k=config.top_k,
                    n_shared=config.n_shared_experts,
                    hidden=config.ffn_hidden,
                    multiple_of=config.ffn_multiple_of,
                    use_bias=False,
                    aux_loss_coef=config.aux_loss_coef,
                    linear_cls=linear_cls,
                )
            else:
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
        """See :meth:`llm.models.gpt3.GPT3.forward` — identical contract.

        When targets are given, the MoE router's load-balancing auxiliary loss is added
        to the cross-entropy loss so the router learns balanced expert usage.
        """
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
            # Fold in the router load-balancing loss from every MoE layer (if any).
            aux_loss = collect_moe_aux_loss(self)
            if aux_loss is not None:
                loss = loss + aux_loss
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
