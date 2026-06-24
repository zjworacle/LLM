"""Training loop with FP8 toggle, gradient accumulation, and checkpointing.

The :class:`Trainer` is intentionally framework-light: you give it a model, a
:class:`~llm.train.config.TrainConfig`, and a get_batch callable that returns a
(input_ids, target_ids) pair. The trainer owns device placement, the precision
policy, the optimizer + LR schedule, gradient accumulation, clipping, logging, and
checkpoints.

FP8: when cfg.use_fp8 is set the eligible Linear layers are swapped for
FP8Linear *before* the optimizer is built, so the optimizer tracks the FP8 layers'
high-precision master weights. On MPS this transparently uses the simulated FP8 path.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import torch
import torch.nn as nn

from ..utils.device import (
    device_summary,
    pick_device,
    resolve_precision,
    seed_everything,
)
from .config import TrainConfig
from .optim import build_optimizer, lr_at_step, set_lr

# A callable that yields one (inputs, targets) batch (ids as LongTensors).
BatchFn = Callable[[], tuple[torch.Tensor, torch.Tensor]]


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, get_batch: BatchFn):
        self.cfg = cfg
        seed_everything(cfg.seed)

        # --- Device + precision -----------------------------------------
        self.device = pick_device(cfg.device)
        self.precision = resolve_precision(self.device, cfg.precision)

        # --- Optional FP8 conversion (before optimizer construction) ----
        if cfg.use_fp8:
            from ..fp8.policy import convert_to_fp8

            # On MPS/CPU this forces the simulated path; on capable CUDA it uses HW.
            force_sim = self.device.type != "cuda"
            convert_to_fp8(model, force_simulated=force_sim)

        self.model = model.to(self.device, dtype=self.precision.param_dtype)
        self.get_batch = get_batch

        # --- Optimizer + schedule ---------------------------------------
        self.optimizer = build_optimizer(
            self.model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )

        # fp16 on CUDA needs loss scaling; bf16/fp32 do not.
        self.scaler = None
        if self.precision.compute_dtype == torch.float16 and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        self.step = 0
        os.makedirs(cfg.out_dir, exist_ok=True)

    # -------------------------------------------------------------------
    def _autocast(self):
        """Context manager applying the resolved autocast policy (or a no-op)."""
        if self.precision.autocast_enabled:
            return torch.autocast(
                device_type=self.precision.autocast_device_type,
                dtype=self.precision.compute_dtype,
            )
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _run_step(self) -> float:
        """One optimizer step = grad_accum_steps micro-batches. Returns avg loss."""
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = self.get_batch()
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                _, loss = self.model(x, targets=y)
                # Scale so accumulated grads equal the mean over the full batch.
                loss = loss / cfg.grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        # Gradient clipping (unscale first if using a GradScaler).
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return total_loss

    # -------------------------------------------------------------------
    def _log_extra(self) -> str:
        """Hook for subclasses to append extra metrics to the per-step log line.

        Returns a string that is inserted before the timing field; the empty default
        keeps the base log unchanged. Subclasses (e.g. DPOTrainer) override it to add
        metrics like reward accuracy. Must end with a trailing space if non-empty.
        """
        return ""

    # -------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        print(f"[trainer] device: {device_summary(self.device)}")
        print(
            f"[trainer] precision: compute={self.precision.compute_dtype} "
            f"autocast={self.precision.autocast_enabled} fp8={cfg.use_fp8}"
        )
        if hasattr(self.model, "num_params"):
            print(f"[trainer] params: {self.model.num_params():,}")

        self.model.train()
        t0 = time.time()
        for self.step in range(cfg.max_steps):
            # Apply the warmup+cosine LR for this step.
            lr = lr_at_step(
                self.step,
                base_lr=cfg.lr,
                min_lr=cfg.min_lr,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
            )
            set_lr(self.optimizer, lr)

            loss = self._run_step()

            if self.step % cfg.log_every == 0:
                dt = time.time() - t0
                print(
                    f"step {self.step:>6} | loss {loss:.4f} | lr {lr:.2e} "
                    f"{self._log_extra()}| {dt:.1f}s"
                )
            if cfg.ckpt_every > 0 and self.step > 0 and self.step % cfg.ckpt_every == 0:
                self.save_checkpoint()

        self.save_checkpoint(final=True)
        print(f"[trainer] done in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------
    def save_checkpoint(self, final: bool = False) -> str:
        """Save model + optimizer state. Returns the path written."""
        name = "final.pt" if final else f"step_{self.step}.pt"
        path = os.path.join(self.cfg.out_dir, name)
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": vars(self.cfg),
            },
            path,
        )
        print(f"[trainer] saved checkpoint -> {path}")
        return path
