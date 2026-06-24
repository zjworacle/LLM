"""LoRA (Low-Rank Adaptation) adapters.

LoRA freezes the pretrained weight W and learns a low-rank update ΔW = B @ A,
so the effective weight is W + (alpha/r) * B @ A. Only A (r x in) and B
(out x r) are trained — a tiny fraction of the parameters — which makes fine-tuning
cheap and checkpoints small.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).

This module provides:
* :class:`LoRALinear` — wraps a frozen nn.Linear and adds the low-rank path.
* :func:`apply_lora` — inject LoRA into the projection layers of a model by name.
* :func:`merge_lora` / :func:`unmerge_lora` — fold the adapter into the base weight for
  zero-overhead inference (and undo it).
* :func:`lora_state_dict` — extract just the adapter tensors for a small checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default projections to adapt. For attention these are q/k/v/o; for MLP the inner
# linears (up/down for GELU, gate/up/down for SwiGLU).
# CUSTOMIZE: narrow this (e.g. just q_proj/v_proj) to train fewer params.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """A frozen base nn.Linear augmented with a trainable low-rank update.

    The base layer's weight/bias are frozen (requires_grad=False). Forward returns
    base(x) + scaling * (x @ A^T @ B^T).

    Args:
        base: the pretrained linear layer to wrap (its weights are frozen).
        r: LoRA rank. r=0 disables the adapter (acts as the frozen base).
        alpha: scaling numerator; effective scale is alpha / r.
        dropout: dropout applied to the input of the LoRA path.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.merged = False

        in_features = base.in_features
        out_features = base.out_features
        if r > 0:
            # A initialized with Kaiming, B with zeros => ΔW starts at 0 (no-op at init).
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0 and not self.merged:
            # Low-rank path: (x @ A^T) @ B^T, scaled.
            delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            out = out + self.scaling * delta
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ΔW into the base weight for fast inference (idempotent)."""
        if self.r > 0 and not self.merged:
            self.base.weight.add_(self.scaling * (self.lora_B @ self.lora_A))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo :meth:`merge`."""
        if self.r > 0 and self.merged:
            self.base.weight.sub_(self.scaling * (self.lora_B @ self.lora_A))
            self.merged = False


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> nn.Module:
    """Wrap matching nn.Linear submodules with :class:`LoRALinear`, in place.

    Matching is by the leaf attribute name (e.g. q_proj). All non-LoRA parameters
    are frozen, so after this call only the adapters train.

    Returns the same model for chaining.
    """
    # Freeze everything first; LoRALinear re-exposes only its adapter params.
    for p in model.parameters():
        p.requires_grad = False

    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(model, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora(child, r=r, alpha=alpha, dropout=dropout, targets=targets)
    return model


def merge_lora(model: nn.Module) -> None:
    """Merge all LoRA adapters in a model into their base weights (in place)."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Unmerge all LoRA adapters in a model (in place)."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the LoRA adapter tensors (for small fine-tuning checkpoints)."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Ensure only LoRA adapter params require grad (call after loading weights)."""
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
