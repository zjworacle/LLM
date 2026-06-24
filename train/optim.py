"""Optimizer and learning-rate schedule.

This mirrors the recipe used by GPT-3 / LLaMA and kept by DeepSeek-V3:

* **AdamW** with betas (0.9, 0.95) and decoupled weight decay.
* Weight decay applied **only** to matmul weights (2D tensors); biases, norms, and
  embeddings are excluded — they should not be pulled toward zero.
* A **warmup + cosine decay** learning-rate schedule.

On the FP8 master-weights question: PyTorch's AdamW already keeps its moment
estimates and the parameter update in the parameter's dtype. We keep parameters in
FP32 (the "master weights"), so the optimizer state is FP32 too. If you later store
params in BF16, you would want a custom optimizer that holds FP32 master weights and
BF16 moments — see the note in :func:`build_optimizer`.
# IMPROVE: add a fused/foreach optimizer and optional BF16 moment storage for memory.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Create an AdamW optimizer with sensible parameter grouping.

    Parameters with >= 2 dimensions (linear/embedding weights) get weight decay; every
    1D parameter (biases, norm gains) is excluded.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # CUSTOMIZE: pass fused=True on CUDA for a speedup (not supported on MPS).
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def lr_at_step(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Warmup-then-cosine learning rate for a given step (0-indexed).

    * Linear warmup from 0 -> base_lr over warmup_steps.
    * Cosine decay from base_lr -> min_lr over the remaining steps.
    * After max_steps the LR is clamped at min_lr.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay over the post-warmup span.
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Overwrite the learning rate on every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr
