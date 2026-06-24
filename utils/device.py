"""Device and dtype selection, with an Apple-Silicon-first (MPS) policy.

Why this module exists
----------------------
The very first milestone of this project is to train a *tiny* model on an Apple
Silicon Mac. Apple GPUs are exposed through PyTorch's mps backend, which has a
few important quirks compared to CUDA:

* There is **no FP8 hardware**, and torch._scaled_mm is unavailable. Any FP8 in
  this project therefore runs through a *simulated* quantize/dequantize path on MPS
  (see llm.fp8).
* float64 is **not supported** on MPS.
* bfloat16 support exists on recent PyTorch builds but is less battle-tested than
  on CUDA. We default autocast on MPS to float16 only if explicitly requested and
  otherwise prefer plain float32 for correctness during the bring-up phase.

This module centralizes those decisions so the rest of the codebase never hard-codes
a device or dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Order of preference (the MPS-first policy):  mps -> cuda -> cpu.

    Args:
        prefer: Force a specific device string (e.g. "cpu", "cuda",
            "mps"). If the requested device is unavailable we fall back to the
            automatic order and never raise — this keeps scripts runnable everywhere.

    # CUSTOMIZE: if you primarily run on CUDA clusters, flip the order below so that
    # cuda is tried before mps.
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "mps" and _mps_ok():
            return torch.device("mps")
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "cpu":
            return torch.device("cpu")
        # Requested device not available -> fall through to auto-selection.

    if _mps_ok():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mps_ok() -> bool:
    """True if the MPS backend is built and usable on this machine."""
    # is_available checks both that we are on Apple Silicon and that the current
    # torch build includes the MPS backend.
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


# ---------------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Precision:
    """Resolved precision settings for a given device.

    Attributes:
        compute_dtype: dtype used for matmul-heavy compute (autocast target).
        param_dtype: dtype used to *store* model parameters (master weights).
        autocast_enabled: whether to wrap forward passes in torch.autocast.
        autocast_device_type: the device_type string passed to torch.autocast.
    """

    compute_dtype: torch.dtype
    param_dtype: torch.dtype
    autocast_enabled: bool
    autocast_device_type: str


def resolve_precision(device: torch.device, mode: str = "auto") -> Precision:
    """Choose sensible compute/storage dtypes for a device.

    Args:
        device: target device from :func:`pick_device`.
        mode: one of "auto", "fp32", "bf16", "fp16".

    # IMPROVE: during the MPS bring-up we deliberately keep things conservative
    # (fp32 master weights everywhere). Once the tiny model trains cleanly you can
    # switch mode="bf16" to roughly halve activation memory.
    """
    dev = device.type

    if mode == "fp32":
        return Precision(torch.float32, torch.float32, False, dev)

    if mode == "bf16":
        # bf16 has fp32's exponent range, so it is the safest reduced precision.
        # Master weights stay fp32 (DeepSeek-V3 keeps fp32 master weights too).
        return Precision(torch.bfloat16, torch.float32, True, dev)

    if mode == "fp16":
        # fp16 has a narrow exponent range and usually needs a GradScaler on CUDA.
        # CUSTOMIZE: wire up torch.cuda.amp.GradScaler in the trainer if you use this.
        return Precision(torch.float16, torch.float32, True, dev)

    # mode == "auto": pick per device.
    if dev == "cuda":
        # Most modern NVIDIA GPUs do bf16 well.
        return Precision(torch.bfloat16, torch.float32, True, "cuda")
    if dev == "mps":
        # Conservative default for the first Mac milestone: full fp32, no autocast.
        # CUSTOMIZE: set mode="bf16" explicitly once the fp32 path is validated.
        return Precision(torch.float32, torch.float32, False, "mps")
    # CPU: fp32 only (bf16 on CPU is slow and uneven).
    return Precision(torch.float32, torch.float32, False, "cpu")


def device_summary(device: torch.device) -> str:
    """Human-readable one-liner describing the selected device (for logging)."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"cuda ({name}, {total:.1f} GB)"
    if device.type == "mps":
        return "mps (Apple Silicon unified memory)"
    return "cpu"


def seed_everything(seed: int = 1234) -> None:
    """Seed Python, NumPy-free torch RNGs for reproducible tiny runs.

    # CUSTOMIZE: add numpy/random seeding here if you introduce those libraries.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
