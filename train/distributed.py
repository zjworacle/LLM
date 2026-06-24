"""Distributed-training helpers (DDP / FSDP / XLA), with safe single-device fallbacks.

The Mac/MPS milestone runs on a *single* device, so by default everything here is a
no-op: :func:`init_distributed` reports world size 1 and :func:`wrap_model` returns the
model unchanged. The hooks exist so that the same training code scales to multi-GPU
without edits once you move to a CUDA cluster.

# IMPROVE: this is a thin stub. For real multi-node runs wire up a proper launcher
# (torchrun) and an FSDP auto-wrap policy keyed on the TransformerLayer class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DistInfo:
    """Resolved distributed topology for the current process."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    """Initialize a process group if launched under torchrun; else single-process.

    Detects the standard RANK/WORLD_SIZE/LOCAL_RANK env vars that
    torchrun sets. On a laptop none are present, so we return a single-process
    topology and never touch torch.distributed.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # CUSTOMIZE: use "gloo" for CPU-only multi-process; "nccl" for CUDA clusters.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def wrap_model(model: nn.Module, info: DistInfo, strategy: str = "ddp") -> nn.Module:
    """Wrap a model for distributed training, or return it unchanged when single-device.

    Args:
        strategy: "ddp" (data parallel, replicated weights) or "fsdp" (sharded).
    """
    if not info.enabled:
        return model

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [info.local_rank] if torch.cuda.is_available() else None
        return DDP(model, device_ids=device_ids)

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # CUSTOMIZE: add an auto_wrap_policy on TransformerLayer for per-layer sharding.
        return FSDP(model)

    raise ValueError(f"unknown strategy: {strategy!r}")


def cleanup_distributed(info: DistInfo) -> None:
    """Tear down the process group if one was created."""
    if info.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
