"""FP8Linear: a drop-in nn.Linear replacement with DeepSeek-V3 FP8 GEMMs.

Two execution paths share one module:

1. **Real FP8** (CUDA Hopper/Ada): uses torch._scaled_mm for a genuine FP8 matmul.
2. **Simulated FP8** (MPS / CPU / older CUDA — the default on a Mac): quantizes operands
   to E4M3 and dequantizes them back before a normal high-precision matmul. The result
   is numerically what FP8 would produce, so you can develop and validate the recipe on
   any machine.

In both paths the master weights are stored in high precision (FP32/BF16) — exactly as
DeepSeek-V3 keeps FP32 master weights — and only the *matmul inputs* are cast to FP8.

The three GEMMs the paper puts in FP8 are all reproduced in the simulated autograd
function below:

* **Fprop**  : y  = x @ Wᵀ                     (forward)
* **Dgrad**  : dx = dy @ W                      (grad w.r.t. input)
* **Wgrad**  : dW = dyᵀ @ x                     (grad w.r.t. weight)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import (
    GROUP,
    is_fp8_supported,
    quant_dequant_activation,
    quant_dequant_weight,
)


def _eligible(in_features: int, out_features: int) -> bool:
    """FP8 fine-grained tiling needs both dims to be multiples of 128.

    When a layer is not eligible (e.g. a tiny test layer, or the vocab-sized LM head)
    we transparently fall back to a normal high-precision matmul.
    # IMPROVE: pad to a multiple of 128 to make more layers eligible.
    """
    return in_features % GROUP == 0 and out_features % GROUP == 0


class _SimulatedFP8Matmul(torch.autograd.Function):
    """Autograd function running fprop/dgrad/wgrad as simulated FP8 GEMMs.

    Each GEMM quantizes its operands to E4M3 (then dequantizes) and accumulates in
    FP32, mirroring DeepSeek-V3's "FP8 inputs, FP32 accumulation" design.
    """

    @staticmethod
    def forward(ctx, x2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x2d: (rows, in)  weight: (out, in)
        # Fprop: quantize both operands, matmul in fp32, return in x's dtype.
        xq = quant_dequant_activation(x2d)
        wq = quant_dequant_weight(weight)
        ctx.save_for_backward(xq, wq)
        ctx.in_dtype = x2d.dtype
        out = (xq.float() @ wq.float().t()).to(x2d.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        xq, wq = ctx.saved_tensors
        # Quantize the incoming gradient too (the paper keeps dgrad/wgrad in FP8).
        gq = quant_dequant_activation(grad_out.reshape(-1, grad_out.shape[-1]))

        # Dgrad: dx = dy @ W  (W already quantized).
        grad_x = (gq.float() @ wq.float()).to(ctx.in_dtype)
        # Wgrad: dW = dyᵀ @ x  (x already quantized).
        grad_w = (gq.float().t() @ xq.float()).to(wq.dtype)
        return grad_x, grad_w


class FP8Linear(nn.Module):
    """Linear layer whose matmul uses FP8 (real on Hopper, simulated elsewhere).

    The constructor signature matches nn.Linear so it can be swapped in directly by
    passing linear_cls=FP8Linear to the model building blocks.

    Args:
        in_features, out_features, bias: as in nn.Linear.
        force_simulated: skip the real _scaled_mm path even on capable GPUs (useful
            for parity testing). On MPS/CPU the simulated path is used regardless.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        force_simulated: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.force_simulated = force_simulated

        # Master weights in high precision (filled by the model's init_weights).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

        self.eligible = _eligible(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layers that don't tile cleanly fall back to a standard matmul.
        if not self.eligible:
            return F.linear(x, self.weight, self.bias)

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])

        if (not self.force_simulated) and is_fp8_supported(x.device):
            out2d = self._real_fp8_matmul(x2d)
        else:
            # The simulated path: works on MPS / CPU / any GPU.
            out2d = _SimulatedFP8Matmul.apply(x2d, self.weight)

        out = out2d.reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    # ------------------------------------------------------------------
    def _real_fp8_matmul(self, x2d: torch.Tensor) -> torch.Tensor:
        """Genuine FP8 GEMM via torch._scaled_mm (CUDA Hopper/Ada only).

        # IMPROVE: this uses simple per-tensor scales for the real-hardware path. For
        # full DeepSeek-V3 fidelity you would pass the 1x128 / 128x128 group scales to a
        # grouped-scaled GEMM kernel. The simulated path already models the fine-grained
        # behaviour for correctness studies.
        """
        from .quant import FP8_DTYPE, FP8_MAX

        x_amax = x2d.abs().amax().clamp(min=1e-12)
        w_amax = self.weight.abs().amax().clamp(min=1e-12)
        x_scale = x_amax / FP8_MAX
        w_scale = w_amax / FP8_MAX

        x_fp8 = (x2d / x_scale).to(FP8_DTYPE)
        w_fp8 = (self.weight / w_scale).to(FP8_DTYPE)

        # _scaled_mm computes (x_fp8 * x_scale) @ (w_fp8 * w_scale)ᵀ with fp32 accumulate.
        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=x_scale.to(x2d.device),
            scale_b=w_scale.to(x2d.device),
            out_dtype=x2d.dtype,
        )
        # Some torch versions return a tuple (out, amax); normalize to the tensor.
        return out[0] if isinstance(out, tuple) else out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, eligible={self.eligible}"
        )
