"""Concrete model + training run configs (importable as llm.runconfigs).

Each run config bundles a model architecture and its training setup into one named,
ready-to-run pair. The tiny ones are small enough to train on an Apple Silicon Mac
(MPS) or CPU; the larger ones describe real scales and are intended for CUDA / multi-GPU.

configs/runconfigs.py re-exports everything here for backward compatibility.
"""

from __future__ import annotations

from .models.gpt3 import GPT3Config
from .models.llama3 import Llama3Config
from .models.llama4 import Llama4Config
from .train.config import TrainConfig


# ---------------------------------------------------------------------------
# Tiny run configs — the Mac/MPS bring-up target (~a few million parameters).
# ---------------------------------------------------------------------------
def tiny_gpt3() -> tuple[GPT3Config, TrainConfig]:
    """A tiny GPT-3 that fits comfortably on a Mac (MPS) or CPU.

    # CUSTOMIZE: bump n_layers/dim/n_heads to grow the model. With dim=256/4 layers
    # this is ~3-4M non-embedding params — trains in seconds per step on MPS.
    """
    model = GPT3Config(
        vocab_size=50257,  # tiktoken "gpt2"
        block_size=256,
        n_layers=4,
        dim=256,
        n_heads=4,
        dropout=0.0,
    )
    train = TrainConfig(
        out_dir="out/tiny_gpt3",
        batch_size=8,
        max_steps=1000,
        precision="auto",  # fp32 on MPS during bring-up
        use_fp8=False,
    )
    return model, train


def tiny_llama3() -> tuple[Llama3Config, TrainConfig]:
    """A tiny LLaMA-3 (RoPE + GQA + SwiGLU) for Mac/CPU bring-up."""
    model = Llama3Config(
        vocab_size=50257,  # reuse gpt2 vocab for the tiny demo (small embedding)
        block_size=256,
        n_layers=4,
        dim=256,
        n_heads=4,
        n_kv_heads=2,  # GQA
        rope_theta=10_000.0,  # classic RoPE base is fine at tiny scale
    )
    train = TrainConfig(
        out_dir="out/tiny_llama3",
        batch_size=8,
        max_steps=1000,
        precision="auto",
        use_fp8=False,
    )
    return model, train


def tiny_llama4() -> tuple[Llama4Config, TrainConfig]:
    """A tiny LLaMA-4 (LLaMA-3 backbone + sparse MoE feed-forward) for Mac/CPU bring-up."""
    model = Llama4Config(
        vocab_size=50257,  # reuse gpt2 vocab for the tiny demo
        block_size=256,
        n_layers=4,
        dim=256,
        n_heads=4,
        n_kv_heads=2,  # GQA
        n_experts=8,
        top_k=2,
        n_shared_experts=1,
        moe_interleave=1,  # every layer is MoE at this tiny scale
        rope_theta=10_000.0,  # classic RoPE base is fine at tiny scale
    )
    train = TrainConfig(
        out_dir="out/tiny_llama4",
        batch_size=8,
        max_steps=1000,
        precision="auto",
        use_fp8=False,
    )
    return model, train


# ---------------------------------------------------------------------------
# Larger run configs — real scales. These will NOT fit a tiny machine and are intended
# for CUDA single-GPU (small) or multi-GPU/FSDP (llama3-8b). Listed so the same code
# paths (train/finetune/generate) work unchanged at scale.
# CUSTOMIZE: tune block_size / batch_size / max_steps for your hardware budget.
# ---------------------------------------------------------------------------
def gpt3_small() -> tuple[GPT3Config, TrainConfig]:
    """GPT-3 "small" (≈125M params) — matches the GPT-2/GPT-3-small shape."""
    model = GPT3Config(
        vocab_size=50257,
        block_size=1024,
        n_layers=12,
        dim=768,
        n_heads=12,
        dropout=0.0,
        tie_embeddings=True,
    )
    train = TrainConfig(
        out_dir="out/gpt3_small",
        batch_size=8,
        grad_accum_steps=4,
        max_steps=10_000,
        precision="bf16",  # bf16 on CUDA; fp32 fallback on CPU/MPS
        use_fp8=False,
    )
    return model, train


def llama3_1b() -> tuple[Llama3Config, TrainConfig]:
    """A ~1B-parameter LLaMA-3 style config."""
    model = Llama3Config(
        vocab_size=128256,
        block_size=2048,
        n_layers=16,
        dim=2048,
        n_heads=32,
        n_kv_heads=8,  # GQA
        rope_theta=500_000.0,
    )
    train = TrainConfig(
        out_dir="out/llama3_1b",
        batch_size=4,
        grad_accum_steps=8,
        max_steps=50_000,
        precision="bf16",
        use_fp8=True,  # FP8 pays off at this scale on Hopper/Ada GPUs
    )
    return model, train


def llama3_8b() -> tuple[Llama3Config, TrainConfig]:
    """The LLaMA-3 8B configuration (needs FSDP / multi-GPU to train)."""
    model = Llama3Config(
        vocab_size=128256,
        block_size=8192,
        n_layers=32,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,  # GQA
        ffn_hidden=14336,  # LLaMA-3 8B uses an explicit FFN size
        rope_theta=500_000.0,
    )
    train = TrainConfig(
        out_dir="out/llama3_8b",
        batch_size=1,
        grad_accum_steps=16,
        max_steps=100_000,
        precision="bf16",
        use_fp8=True,
    )
    return model, train


def llama4_moe() -> tuple[Llama4Config, TrainConfig]:
    """A LLaMA-4 style MoE config: many experts, top-2 routing, interleaved dense layers.

    Total parameters scale with n_experts while per-token compute stays near a dense
    model (only top_k + shared experts run per token). Needs a CUDA GPU (FSDP for the
    big variants) to train.
    """
    model = Llama4Config(
        vocab_size=128256,
        block_size=2048,
        n_layers=16,
        dim=2048,
        n_heads=32,
        n_kv_heads=8,  # GQA
        n_experts=16,
        top_k=2,
        n_shared_experts=1,
        first_dense_layers=1,  # keep the first layer dense, MoE for the rest
        moe_interleave=2,  # of the MoE-eligible layers, every other one is MoE
        rope_theta=500_000.0,
    )
    train = TrainConfig(
        out_dir="out/llama4_moe",
        batch_size=4,
        grad_accum_steps=8,
        max_steps=50_000,
        precision="bf16",
        use_fp8=True,
    )
    return model, train


# ---------------------------------------------------------------------------
# Registry so scripts/CLI can select a run config by name.
# CUSTOMIZE: register more run configs here.
# ---------------------------------------------------------------------------
RUN_CONFIGS = {
    "tiny-gpt3": tiny_gpt3,
    "tiny-llama3": tiny_llama3,
    "tiny-llama4": tiny_llama4,
    "gpt3-small": gpt3_small,
    "llama3-1b": llama3_1b,
    "llama3-8b": llama3_8b,
    "llama4-moe": llama4_moe,
}


def get_run_config(name: str):
    """Look up a run-config factory by name, with a helpful error message."""
    if name not in RUN_CONFIGS:
        raise KeyError(f"unknown run config {name!r}; available: {sorted(RUN_CONFIGS)}")
    return RUN_CONFIGS[name]()
