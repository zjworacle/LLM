"""Shared transformer building blocks used by both GPT-3 and LLaMA-3.

This module is the heart of the from-scratch implementation. Both model files
(gpt3.py and llama3.py) are assembled almost entirely from the components
defined here, differing only in *which* pieces they use:

=================  =========================  ==============================
Component          GPT-3                      LLaMA-3
=================  =========================  ==============================
Normalization      LayerNorm                  RMSNorm
Positional info    learned position embed.    RoPE (rotary)
Attention          multi-head (MHA)           grouped-query (GQA)
MLP activation     GELU                       SwiGLU
=================  =========================  ==============================

The attention module here is written generically so it supports both MHA (by setting
n_kv_heads == n_heads) and GQA (n_kv_heads < n_heads), with optional RoPE and an
optional KV cache for fast inference. On CUDA it transparently uses Dao-AILab
FlashAttention (FA3 with an FA2 fallback) and otherwise falls back to PyTorch's fused
scaled_dot_product_attention.

Throughout, # CUSTOMIZE: marks tunable knobs and # IMPROVE: marks deliberate
simplifications worth upgrading later (e.g. swapping in FlashAttention).

Tensor-shape notation
---------------------
Shape comments below use descriptive axis names::

    batch_size      number of sequences in a micro-batch
    seq_length      tokens per sequence (<= block_size)
    model_dim       config.dim; the residual-stream width
    n_heads         number of query heads
    n_kv_heads      number of key/value heads (== n_heads for MHA, < for GQA)
    head_dim        per-head dim (model_dim // n_heads)
    mlp_hidden      MLP hidden dim (4*model_dim for GELU; ~8/3*model_dim for SwiGLU)
    vocab_size      tokenizer vocabulary size
    max_seq_length  RoPE table length

So e.g. (batch_size, seq_length, model_dim) is the residual stream and
(batch_size, n_heads, seq_length, head_dim) is the per-head q/k/v.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# FlashAttention (optional, CUDA-only acceleration)
# ===========================================================================
# We prefer the Dao-AILab FlashAttention kernels when they are installed and we are
# on a supported CUDA GPU. FlashAttention is IO-aware: it never materializes the full
# (seq_length, seq_length) score matrix in HBM, so it is both faster and uses far less
# memory than the naive softmax(QK^T)V — the gap grows with sequence length.
#
# We try the newest API first (FlashAttention-3, the Hopper-optimized kernels exposed
# by the flash_attn_interface module) and fall back to FlashAttention-2 (flash_attn).
# If neither is importable (e.g. on Apple-Silicon/MPS or CPU) we transparently fall
# back to PyTorch's fused scaled_dot_product_attention.
try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore  # FA3 (Hopper)

    FLASH_ATTN_VERSION = 3
except ImportError:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore  # FA2

        FLASH_ATTN_VERSION = 2
    except ImportError:
        _flash_attn_func = None
        FLASH_ATTN_VERSION = 0


# ===========================================================================
# Normalization layers
# ===========================================================================
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (used by LLaMA-3).

    RMSNorm normalizes by the RMS of the activations (no mean subtraction, no bias),
    which is cheaper than LayerNorm and works just as well for LLMs.

        y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # CUSTOMIZE: LLaMA uses 1e-5; some models use 1e-6.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_length, model_dim) -> same shape (normalizes over model_dim)
        # Compute in float32 for numerical stability, then cast back. This matters on
        # reduced-precision/MPS runs where x may be bf16/fp16.
        dtype = x.dtype
        x = x.float()
        # mean over the last axis (model_dim), keepdim -> (batch_size, seq_length, 1).
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # weight: (model_dim,) broadcasts over (batch_size, seq_length, model_dim).
        return (norm.to(dtype)) * self.weight


def build_norm(kind: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """Factory so model code can request a norm by name.

    # CUSTOMIZE: register additional norms (e.g. a fused norm) here.
    """
    kind = kind.lower()
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        # GPT-3 uses LayerNorm *with* learnable bias.
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm kind: {kind!r}")


# ===========================================================================
# Rotary positional embeddings (RoPE) — used by LLaMA-3
# ===========================================================================
def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin tables used to rotate query/key vectors.

    Args:
        head_dim: per-head dimension (must be even).
        max_seq_len: longest sequence we will ever rotate.
        theta: RoPE base frequency.
            # CUSTOMIZE: LLaMA-3 uses 500000 for long context; classic RoPE uses 10000.
        device: where to allocate the tables.

    Returns:
        (cos, sin), each shaped (max_seq_len, head_dim).
    """
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"
    # Frequencies for each pair of dimensions: 1 / theta^(2i/d).
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # inv_freq: (head_dim/2,)
    positions = torch.arange(max_seq_len, device=device).float()  # (max_seq_length,)
    # Outer product (max_seq_length, head_dim/2), then duplicate the half -> (.., head_dim).
    freqs = torch.outer(positions, inv_freq)  # (max_seq_length, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_length, head_dim)
    return emb.cos(), emb.sin()  # each: (max_seq_length, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim to the front (negated). RoPE helper."""
    # x: (batch_size, n_heads, seq_length, head_dim) -> same shape (splits head_dim in two)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]  # each: (.., head_dim/2)
    return torch.cat((-x2, x1), dim=-1)  # (.., head_dim)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Shapes: q, k are (batch, n_heads, seq, head_dim). cos/sin are the
    precomputed (max_seq, head_dim) tables.

    Args:
        offset: starting position. Non-zero during incremental decoding so cached
            tokens keep their original positions.
    """
    seq = q.shape[-2]  # seq_length
    # Slice the seq_length positions from the (max_seq_length, head_dim) table and add
    # (batch_size, n_heads) broadcast axes.
    cos_s = cos[offset : offset + seq].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, head_dim)
    sin_s = sin[offset : offset + seq].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, head_dim)
    q_rot = (q * cos_s) + (_rotate_half(q) * sin_s)  # (batch_size, n_heads, seq_length, head_dim)
    k_rot = (k * cos_s) + (_rotate_half(k) * sin_s)  # (batch_size, n_kv_heads, seq_length, head_dim)
    return q_rot.type_as(q), k_rot.type_as(k)


# ===========================================================================
# Attention (supports MHA and grouped-query attention + KV cache)
# ===========================================================================
@dataclass
class KVCache:
    """A simple per-layer key/value cache for autoregressive decoding.

    # IMPROVE: this is a plain pre-allocated cache. For production you might use a
    # paged/ring-buffer cache to support very long generations without reallocation.
    """

    k: torch.Tensor  # (batch, n_kv_heads, max_seq, head_dim)
    v: torch.Tensor
    length: int = 0  # number of valid cached positions

    @classmethod
    def empty(
        cls,
        batch: int,
        n_kv_heads: int,
        max_seq: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "KVCache":
        shape = (batch, n_kv_heads, max_seq, head_dim)
        return cls(
            k=torch.zeros(shape, dtype=dtype, device=device),
            v=torch.zeros(shape, dtype=dtype, device=device),
            length=0,
        )


class Attention(nn.Module):
    """Causal self-attention supporting multi-head and grouped-query attention.

    Set n_kv_heads == n_heads for standard MHA (GPT-3). Set n_kv_heads < n_heads
    for GQA (LLaMA-3), where multiple query heads share each key/value head to shrink
    the KV cache.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        use_bias: bool = False,
        use_rope: bool = True,
        dropout: float = 0.0,
        use_flash: bool = True,
        linear_cls: type[nn.Module] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = dim // n_heads
        self.q_per_kv = self.n_heads // self.n_kv_heads  # query heads sharing each kv head
        self.use_rope = use_rope
        self.dropout = dropout
        # Allow FlashAttention only if the user opted in AND the kernels imported.
        # The per-call _use_flash() check below additionally requires a CUDA tensor in
        # half precision, so this stays a no-op on MPS/CPU/fp32 runs.
        self.use_flash = use_flash and _flash_attn_func is not None

        # linear_cls lets the FP8 layer (llm.fp8.linear.FP8Linear) be injected in
        # place of nn.Linear without changing this module. Defaults to nn.Linear.
        # CUSTOMIZE: pass a custom linear class to change the projection implementation.
        Linear = linear_cls or nn.Linear

        # Separate projections keep the code readable; many implementations fuse QKV
        # into one matmul for speed.  # IMPROVE: fuse q/k/v into a single Linear.
        self.q_proj = Linear(dim, self.n_heads    * self.head_dim, bias=use_bias)
        self.k_proj = Linear(dim, self.n_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = Linear(dim, self.n_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = Linear(self.n_heads * self.head_dim, dim, bias=use_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape  # x: (batch_size, seq_length, model_dim)

        # Project to q/k/v and split into heads.
        #   q_proj: (batch_size, seq_length, model_dim) -> (batch_size, seq_length,
        #   n_heads*head_dim); .view splits the heads; .transpose(1,2) moves heads ahead
        #   of seq_length -> (batch_size, n_heads, seq_length, head_dim).
        #   k/v use n_kv_heads (<= n_heads) -> (batch_size, n_kv_heads, seq_length, head_dim).
        q = self.q_proj(x).view(batch_size, seq_length, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Rotary embeddings (LLaMA path). GPT-3 passes cos/sin=None.
        offset = cache.length if cache is not None else 0
        if self.use_rope and cos is not None and sin is not None:
            q, k = apply_rope(q, k, cos, sin, offset=offset)  # shapes unchanged

        # Update KV cache for incremental decoding. After this, k/v span all cached
        # positions: (batch_size, n_kv_heads, cached_length, head_dim) where
        # cached_length = cache.length (= seq_length during decoding).
        if cache is not None:
            end = cache.length + seq_length
            cache.k[:, :, cache.length : end] = k
            cache.v[:, :, cache.length : end] = v
            cache.length = end
            k = cache.k[:, :, :end]  # (batch_size, n_kv_heads, cached_length, head_dim)
            v = cache.v[:, :, :end]  # (batch_size, n_kv_heads, cached_length, head_dim)

        # Attention. is_causal applies the causal mask in the full prefill / training
        # case where q and k have equal length (no cache, or the cache's first step).
        is_causal = (cache is None or offset == 0) and seq_length > 1
        if self._use_flash(q):
            # FlashAttention path (CUDA, fp16/bf16): IO-aware kernel that never
            # materializes the score matrix and handles GQA + causal masking natively.
            y = self._flash_attention(q, k, v, is_causal)
        else:
            # Portable fallback: PyTorch's fused SDPA (works on MPS/CPU/CUDA).
            y = self._sdpa_attention(q, k, v, is_causal)

        # Recombine heads and project out.
        #   transpose(1,2) -> (batch_size, seq_length, n_heads, head_dim);
        #   view -> (batch_size, seq_length, model_dim); o_proj keeps that shape.
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, self.n_heads * self.head_dim)
        y = self.o_proj(y)  # (batch_size, seq_length, model_dim)
        return self.resid_dropout(y)  # (batch_size, seq_length, model_dim)

    def _use_flash(self, q: torch.Tensor) -> bool:
        """True if the FlashAttention kernel can run on this tensor.

        FlashAttention only supports CUDA tensors in half precision (fp16/bf16), so
        we guard on both. Any other case (MPS, CPU, fp32) falls back to SDPA.
        """
        return self.use_flash and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)

    def _flash_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """Run the Dao-AILab FlashAttention kernel.

        flash_attn_func expects (batch, seq, heads, head_dim) layout and natively
        handles grouped-query attention (n_kv_heads < n_heads) and causal masking, so
        we transpose back from the heads-first layout and skip the GQA head expansion.
        """
        # (batch_size, n_heads, seq_length, head_dim) -> (batch_size, seq_length, n_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = _flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=is_causal,
        )
        # FlashAttention-3 returns (output, softmax_lse); FA2 returns just the output.
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)  # back to (batch_size, n_heads, seq_length, head_dim)

    def _sdpa_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """PyTorch fused scaled-dot-product attention (portable fallback).

        Internally: scores = q @ k^T / sqrt(head_dim) (+ causal mask), softmax over the
        key axis, then weighted sum with v. Works on MPS/CPU/CUDA and picks an efficient
        kernel per backend.
        """
        # Expand kv heads to match q heads for GQA (no-op when q_per_kv == 1):
        #   (batch_size, n_kv_heads, seq_length, head_dim) ->
        #   (batch_size, n_heads, seq_length, head_dim).
        if self.q_per_kv > 1:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)
        return F.scaled_dot_product_attention(
            q,  # (batch_size, n_heads, seq_length, head_dim)
            k,  # (batch_size, n_heads, seq_length, head_dim)
            v,  # (batch_size, n_heads, seq_length, head_dim)
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # -> (batch_size, n_heads, seq_length, head_dim)


# ===========================================================================
# Feed-forward / MLP blocks
# ===========================================================================
class GeluMLP(nn.Module):
    """GPT-3 style MLP: Linear -> GELU -> Linear, with a 4x hidden expansion."""

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        use_bias: bool = True,
        dropout: float = 0.0,
        linear_cls: type[nn.Module] | None = None,
    ):
        super().__init__()
        hidden = hidden or 4 * dim  # CUSTOMIZE: GPT-3 uses 4x; change the ratio here.
        Linear = linear_cls or nn.Linear
        # up_proj lifts model_dim -> mlp_hidden (the "keys"); down_proj writes the
        # activated hidden back to model_dim (the "values" added to the residual stream).
        self.up_proj   = Linear(dim, hidden, bias=use_bias)
        self.down_proj = Linear(hidden, dim, bias=use_bias)
        self.act = nn.GELU()  # CUSTOMIZE: nn.GELU(approximate="tanh") matches GPT-2/3.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_length, model_dim) -> up_proj -> (.., mlp_hidden) -> GELU
        # -> down_proj -> (batch_size, seq_length, model_dim)
        return self.dropout(self.down_proj(self.act(self.up_proj(x))))


class SwiGLUMLP(nn.Module):
    """LLaMA-3 style gated MLP (SwiGLU): down_proj(silu(gate_proj(x)) * up_proj(x)).

    The hidden size is usually ~ (2/3) * 4 * dim, rounded to a multiple of
    multiple_of so matmuls stay hardware-friendly.
    """

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        multiple_of: int = 256,
        use_bias: bool = False,
        linear_cls: type[nn.Module] | None = None,
    ):
        super().__init__()
        if hidden is None:
            # LLaMA's heuristic for the SwiGLU hidden size.
            hidden = int(2 * (4 * dim) / 3)
            # CUSTOMIZE: round up to a multiple to keep tensor cores happy.
            hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        Linear = linear_cls or nn.Linear
        # SwiGLU uses two parallel "up" projections: gate_proj is squashed by SiLU and
        # multiplicatively gates up_proj; down_proj writes the result to the residual stream.
        self.gate_proj = Linear(dim, hidden, bias=use_bias)  # silu-activated gate
        self.up_proj = Linear(dim, hidden, bias=use_bias)  # value path
        self.down_proj = Linear(hidden, dim, bias=use_bias)  # back to model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_length, model_dim); gate_proj/up_proj -> (.., mlp_hidden);
        # the gated product stays (.., mlp_hidden); down_proj maps back to (.., model_dim).
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEMLP(nn.Module):
    """LLaMA-4 style Mixture-of-Experts feed-forward (sparse routing + shared expert).

    Instead of one dense MLP, each token is routed by a learned ``router`` to its
    ``top_k`` best SwiGLU experts (out of ``n_experts``); their outputs are combined
    with the router's (renormalized) weights. An always-on ``shared`` expert processes
    every token, giving a dense backbone plus sparse, specialized capacity — the
    LLaMA-4 design. Only top_k + shared experts run per token, so the parameter count
    grows with n_experts while the per-token compute stays roughly constant.

    During training the module also records a Switch-Transformer load-balancing
    auxiliary loss on ``self.aux_loss`` (None at inference); the model adds it to the
    cross-entropy loss so the router learns to spread tokens evenly across experts.

    forward signature matches SwiGLUMLP/GeluMLP, so it drops into TransformerLayer
    unchanged.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 8,
        top_k: int = 2,
        n_shared: int = 1,
        hidden: int | None = None,
        multiple_of: int = 256,
        use_bias: bool = False,
        aux_loss_coef: float = 0.01,
        linear_cls: type[nn.Module] | None = None,
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts, "top_k must be in [1, n_experts]"
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        # aux_loss is (re)written every training forward; consumed by the model.
        self.aux_loss: torch.Tensor | None = None

        # Router / gate: scores each token against every expert. Kept in full precision
        # (plain nn.Linear) since routing decisions are sensitive to small differences.
        self.router = nn.Linear(dim, n_experts, bias=False)

        # Routed experts: each is an independent SwiGLU MLP.
        self.experts = nn.ModuleList(
            SwiGLUMLP(dim, hidden=hidden, multiple_of=multiple_of, use_bias=use_bias, linear_cls=linear_cls)
            for _ in range(n_experts)
        )
        # Optional always-on shared expert(s), summed into every token's output.
        self.shared = (
            SwiGLUMLP(dim, hidden=hidden, multiple_of=multiple_of, use_bias=use_bias, linear_cls=linear_cls)
            if n_shared > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, model_dim = x.shape
        # Routing is per token, so flatten the batch/seq axes into one token axis.
        tokens = x.reshape(-1, model_dim)  # (num_tokens, model_dim)

        router_logits = self.router(tokens)  # (num_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (num_tokens, n_experts)

        # Keep each token's top_k experts and renormalize their weights to sum to 1.
        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (num_tokens, top_k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(tokens)  # (num_tokens, model_dim)
        # Loop over experts; each processes only the tokens routed to it. Readable and
        # exact; the per-expert batch is the sparse subset that selected that expert.
        # IMPROVE: fuse into a grouped/batched matmul (e.g. grouped GEMM) for throughput.
        for expert_id, expert in enumerate(self.experts):
            # (token_pos, slot) pairs where this expert was among the token's top_k.
            token_pos, slot = torch.where(topk_idx == expert_id)
            if token_pos.numel() == 0:
                continue
            weight = topk_probs[token_pos, slot].unsqueeze(-1)  # (n_selected, 1)
            out[token_pos] += weight * expert(tokens[token_pos])  # (n_selected, model_dim)

        if self.shared is not None:
            out = out + self.shared(tokens)  # (num_tokens, model_dim)

        # Load-balancing aux loss is only needed for training the router.
        self.aux_loss = self._load_balance_loss(router_probs, topk_idx) if self.training else None

        return out.view(batch_size, seq_length, model_dim)

    def _load_balance_loss(self, router_probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balancing loss: n_experts * sum_i(f_i * P_i).

        ``f_i`` is the fraction of routing slots dispatched to expert i and ``P_i`` is
        the mean router probability mass on expert i. The product is minimized when both
        are uniform, so this term pushes the router toward balanced expert usage.
        """
        # one-hot over experts, summed across the top_k slots -> per-token dispatch.
        dispatch = F.one_hot(topk_idx, self.n_experts).float()  # (num_tokens, top_k, n_experts)
        tokens_per_expert = dispatch.sum(dim=1).mean(dim=0)  # (n_experts,)
        prob_per_expert = router_probs.mean(dim=0)  # (n_experts,)
        return self.aux_loss_coef * self.n_experts * torch.sum(tokens_per_expert * prob_per_expert)


def collect_moe_aux_loss(model: nn.Module) -> torch.Tensor | None:
    """Sum the load-balancing aux loss across all MoE layers in a model.

    Returns None when there are no MoE layers or none has run a training forward
    (e.g. during inference), so dense models are entirely unaffected.
    """
    losses = [m.aux_loss for m in model.modules() if isinstance(m, MoEMLP) and m.aux_loss is not None]
    if not losses:
        return None
    return torch.stack(losses).sum()


# ===========================================================================
# Transformer layer (pre-norm), parameterized by the choices above
# ===========================================================================
class TransformerLayer(nn.Module):
    """A single pre-norm transformer layer: x + attn(norm(x)); x + mlp(norm(x)).

    The specific norm / attention / MLP submodules are passed in already constructed,
    so the same layer serves both GPT-3 and LLaMA-3.
    """

    def __init__(self, attn_norm: nn.Module, attn: nn.Module, mlp_norm: nn.Module, mlp: nn.Module):
        super().__init__()
        self.attn_norm = attn_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        # Pre-norm residual connections (the standard modern arrangement).
        # All tensors here are (batch_size, seq_length, model_dim); the residual adds
        # keep the shape intact.
        x = x + self.attn(self.attn_norm(x), cos=cos, sin=sin, cache=cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ===========================================================================
# Shared weight initialization
# ===========================================================================
def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """GPT-style initialization applied via model.apply(init_weights).

    # CUSTOMIZE: std=0.02 is the GPT-2/3 default. Some setups scale residual
    # projections by 1/sqrt(2*n_layers); add that here if you want exact parity.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
