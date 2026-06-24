"""Fine-grained FP8 (E4M3) quantization primitives.

Implements the two granularities from DeepSeek-V3:

* **Activations**: per 1x128 *tile* — i.e. each row is split into chunks of 128
  channels, and every chunk gets its own scale (per token, per 128 channels).
* **Weights**: per 128x128 *block* — the weight matrix is tiled into 128x128 blocks,
  each with its own scale.

The core idea: FP8 has a tiny dynamic range, so instead of one scale for a whole tensor
(which a single outlier can dominate), we use many small-group scales. This is what lets
DeepSeek-V3 use the higher-precision E4M3 format on *all* tensors.

All functions are written for clarity, not peak speed.
# IMPROVE: the tiling here uses plain reshapes/unfolds and is not fused. A production
# implementation would use a custom kernel (or torchao) to avoid extra memory traffic.
"""

from __future__ import annotations

import torch

# E4M3 (4 exponent bits, 3 mantissa bits) has a maximum representable magnitude of 448.
# DeepSeek-V3 uses E4M3 for all tensors (relying on fine-grained scaling for range).
# CUSTOMIZE: torch.float8_e5m2 (max 57344) trades mantissa for range if you ever
# need it for gradients on hardware that prefers the hybrid scheme.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
GROUP = 128  # the 128 in "1x128 tile" and "128x128 block"


def _round_to_fp8_grid(x: torch.Tensor) -> torch.Tensor:
    """Round a high-precision tensor onto the E4M3 grid, returning the same dtype.

    This is the core of the *simulated* FP8 path. We cast to the FP8 dtype and back so
    the values are forced onto exactly the levels real FP8 hardware would use.

    MPS limitation: Apple's backend cannot even *store* float8_e4m3fn tensors, so on
    MPS we briefly hop to the CPU for the cast and move the result back. This is slower
    but numerically identical — and the real speed path is CUDA anyway.
    # IMPROVE: replace the CPU hop with an arithmetic E4M3 emulation kernel to keep
    # everything on-device on Macs.
    """
    if x.device.type == "mps":
        return x.detach().cpu().to(FP8_DTYPE).to(torch.float32).to(x.device)
    return x.to(FP8_DTYPE).to(torch.float32)


def _amax_to_scale(amax: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert a max-abs value to a quantization scale.

    The scale maps the group's largest magnitude onto FP8_MAX:
        scale = amax / FP8_MAX           (so x/scale lands in [-FP8_MAX, FP8_MAX])

    # CUSTOMIZE: DeepSeek-V3 rounds some scales to powers of two to avoid extra error in
    # the backward pass; we keep continuous scales for simplicity here.
    """
    return (amax / FP8_MAX).clamp(min=eps)


def quantize_activation_1x128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D activation (rows, cols) with per-(row, 128-col-group) scales.

    Returns:
        (x_fp8, scales) where x_fp8 is the E4M3 tensor (same shape as x) and
        scales is (rows, cols/128) — one scale per 1x128 tile.

    The cols dimension must be a multiple of 128.
    """
    rows, cols = x.shape
    assert cols % GROUP == 0, f"cols ({cols}) must be a multiple of {GROUP}"
    n_groups = cols // GROUP

    # Reshape to expose the 128-wide groups: (rows, n_groups, 128).
    xg = x.reshape(rows, n_groups, GROUP).float()
    amax = xg.abs().amax(dim=-1, keepdim=True)  # (rows, n_groups, 1)
    scale = _amax_to_scale(amax)
    x_fp8 = (xg / scale).to(FP8_DTYPE).reshape(rows, cols)
    return x_fp8, scale.squeeze(-1)  # scales: (rows, n_groups)


def quantize_weight_128x128(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight (out, in) with per-128x128-block scales.

    Returns:
        (w_fp8, scales) where scales is (out/128, in/128).

    Both dimensions must be multiples of 128.
    # IMPROVE: pad to a multiple of 128 instead of asserting, so arbitrary shapes work.
    """
    out_dim, in_dim = w.shape
    assert out_dim % GROUP == 0 and in_dim % GROUP == 0, (
        f"weight dims ({out_dim}x{in_dim}) must both be multiples of {GROUP}"
    )
    ob, ib = out_dim // GROUP, in_dim // GROUP

    # Tile into (ob, 128, ib, 128) then compute a scale per (ob, ib) block.
    wb = w.reshape(ob, GROUP, ib, GROUP).float()
    amax = wb.abs().amax(dim=(1, 3), keepdim=True)  # (ob,1,ib,1)
    scale = _amax_to_scale(amax)
    w_fp8 = (wb / scale).to(FP8_DTYPE).reshape(out_dim, in_dim)
    return w_fp8, scale.reshape(ob, ib)


def dequantize_activation_1x128(
    x_fp8: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of :func:`quantize_activation_1x128`."""
    rows, cols = x_fp8.shape
    n_groups = scales.shape[1]
    xg = x_fp8.reshape(rows, n_groups, GROUP).float()
    x = xg * scales.unsqueeze(-1)
    return x.reshape(rows, cols).to(out_dtype)


def dequantize_weight_128x128(
    w_fp8: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of :func:`quantize_weight_128x128`."""
    out_dim, in_dim = w_fp8.shape
    ob, ib = scales.shape
    wb = w_fp8.reshape(ob, GROUP, ib, GROUP).float()
    w = wb * scales.reshape(ob, 1, ib, 1)
    return w.reshape(out_dim, in_dim).to(out_dtype)


def quant_dequant_activation(x: torch.Tensor) -> torch.Tensor:
    """Round-trip an activation through FP8 (the *simulated* FP8 path).

    Uses per-(row, 128-channel) tile scales, rounds onto the E4M3 grid in fp32, and
    rescales. Works on MPS/CPU/CUDA. Returns a tensor in the original dtype.
    """
    rows, cols = x.shape
    assert cols % GROUP == 0, f"cols ({cols}) must be a multiple of {GROUP}"
    n_groups = cols // GROUP
    xg = x.reshape(rows, n_groups, GROUP).float()
    scale = _amax_to_scale(xg.abs().amax(dim=-1, keepdim=True))
    q = _round_to_fp8_grid(xg / scale)  # onto the E4M3 grid
    deq = (q * scale).reshape(rows, cols)
    return deq.to(x.dtype)


def quant_dequant_weight(w: torch.Tensor) -> torch.Tensor:
    """Round-trip a weight through FP8 (the *simulated* FP8 path).

    Uses per-128x128-block scales.
    """
    out_dim, in_dim = w.shape
    assert out_dim % GROUP == 0 and in_dim % GROUP == 0, (
        f"weight dims ({out_dim}x{in_dim}) must both be multiples of {GROUP}"
    )
    ob, ib = out_dim // GROUP, in_dim // GROUP
    wb = w.reshape(ob, GROUP, ib, GROUP).float()
    scale = _amax_to_scale(wb.abs().amax(dim=(1, 3), keepdim=True))
    q = _round_to_fp8_grid(wb / scale)
    deq = (q * scale).reshape(out_dim, in_dim)
    return deq.to(w.dtype)


def is_fp8_supported(device: torch.device) -> bool:
    """Whether *real* FP8 matmul (torch._scaled_mm) is usable on this device.

    Real FP8 GEMM requires a recent NVIDIA GPU (Hopper H100/H800 or Ada). On MPS/CPU and
    older CUDA GPUs we must use the simulated path.

    # CUSTOMIZE: this is a conservative capability probe. If you know your exact GPU you
    # can simplify it to a compute-capability check (>= (8, 9)).
    """
    if device.type != "cuda" or not hasattr(torch, "_scaled_mm"):
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 9 or torch.cuda.get_device_capability(device) >= (8, 9)
    except Exception:
        return False
