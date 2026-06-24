"""Mixed-precision policy: decide *which* layers run in FP8.

DeepSeek-V3 deliberately keeps several components in high precision (BF16/FP32) because
they are sensitive to low-precision noise or are cheap anyway:

* token embeddings
* the output LM head
* normalization layers
* attention softmax / the attention operator itself
* MoE gating (not used here — our models are dense)

Only the big Linear GEMMs inside attention (q/k/v/o projections) and the MLP run in
FP8. This module provides two ways to honor that policy:

1. linear_cls=FP8Linear passed at *construction* time (preferred — the model builds
   FP8 layers directly). This is what the trainer does when use_fp8=True.
2. :func:`convert_to_fp8` to swap eligible nn.Linear layers in an *existing* model
   in-place (handy for experiments or models built without the hook).
"""

from __future__ import annotations

import torch.nn as nn

from .linear import FP8Linear

# Names of submodules to NEVER convert to FP8, matched against the attribute path.
# CUSTOMIZE: add module names here to keep more layers in high precision.
HIGH_PRECISION_NAMES = (
    "unembed",  # unembedding projection — sensitive, and vocab dim is usually not 128-aligned
    "token_emb",
    "pos_emb",
    "norm",  # any LayerNorm/RMSNorm (attn_norm/mlp_norm/final_norm)
)


def _should_keep_high_precision(name: str) -> bool:
    """True if a module path should be excluded from FP8 conversion."""
    return any(key in name for key in HIGH_PRECISION_NAMES)


def convert_to_fp8(model: nn.Module, force_simulated: bool = False) -> nn.Module:
    """Replace eligible nn.Linear layers with :class:`FP8Linear`, in place.

    Embeddings, norms, and the LM head are left untouched per the policy above. Layers
    whose dimensions are not multiples of 128 are still swapped, but FP8Linear will
    transparently fall back to a high-precision matmul for them.

    Args:
        model: the module to convert (mutated in place and also returned).
        force_simulated: force the simulated FP8 path even on capable GPUs.

    Returns:
        The same model, for chaining.
    """
    for name, child in list(model.named_children()):
        full = name
        if isinstance(child, nn.Linear) and not _should_keep_high_precision(full):
            fp8 = FP8Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                force_simulated=force_simulated,
            )
            # Preserve the trained/initialized weights.
            with_no_grad_copy(fp8, child)
            setattr(model, name, fp8)
        else:
            # Recurse, but skip subtrees explicitly marked high-precision.
            if not _should_keep_high_precision(name):
                convert_to_fp8(child, force_simulated=force_simulated)
    return model


def with_no_grad_copy(dst: FP8Linear, src: nn.Linear) -> None:
    """Copy weights/bias from an nn.Linear into an FP8Linear."""
    import torch

    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)
