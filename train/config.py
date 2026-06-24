"""Training hyper-parameter configuration.

Kept separate from the model configs so the same model can be trained under different
training regimes (precision, batch size, FP8 on/off, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Everything the trainer needs that is *not* part of the model architecture.

    # CUSTOMIZE: these defaults are tuned for a *tiny* model on a laptop/Mac. Scale
    # batch_size / grad_accum / max_steps up for real training runs.
    """

    # --- Optimization ---------------------------------------------------
    lr: float = 3e-4
    min_lr: float = 3e-5  # cosine-decay floor
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95  # GPT-3/LLaMA use 0.95 (not the Adam default 0.999)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000

    # --- Batching -------------------------------------------------------
    batch_size: int = 8
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    # block_size is taken from the model config; repeated here only if you override.

    # --- Precision / FP8 ------------------------------------------------
    precision: str = "auto"  # "auto" | "fp32" | "bf16" | "fp16" (see utils.device)
    use_fp8: bool = False  # enable DeepSeek-V3-style FP8 mixed precision
    # CUSTOMIZE: on Mac/MPS FP8 always runs through the *simulated* path automatically.

    # --- Logging / checkpointing ---------------------------------------
    out_dir: str = "out/tiny"
    log_every: int = 10
    eval_every: int = 200
    ckpt_every: int = 500
    seed: int = 1234

    # --- Device ---------------------------------------------------------
    device: str | None = None  # None -> auto (mps -> cuda -> cpu)

    # --- Misc -----------------------------------------------------------
    extra: dict = field(default_factory=dict)  # scratch space for experiments
