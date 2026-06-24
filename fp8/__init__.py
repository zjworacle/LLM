"""DeepSeek-V3-style FP8 mixed-precision training (pure PyTorch).

This subpackage reproduces the FP8 recipe from the DeepSeek-V3 technical report
(arXiv:2412.19437, Section 3.3):

* FP8 E4M3 for the three Linear GEMMs (forward, dgrad, wgrad).
* **Fine-grained** quantization: activations scaled per 1x128 *tile*, weights per
  128x128 *block*, so a few outliers cannot wreck the whole tensor's scale.
* Online (per-step) max-abs scale computation.
* High-precision (FP32) accumulation.
* Everything else (embeddings, output head, norm, attention softmax, gating) stays in
  high precision.

Because Apple Silicon (MPS) and CPUs have **no FP8 hardware**, the actual matmul runs
through a *simulated* path by default: tensors are quantized to E4M3 and immediately
dequantized back to a high-precision dtype, so the numerics match what FP8 would
produce while running on any device. On a Hopper/Ada CUDA GPU you can opt into the real
torch._scaled_mm path for genuine speed-ups.
"""

from .linear import FP8Linear  # noqa: F401
from .policy import convert_to_fp8  # noqa: F401
