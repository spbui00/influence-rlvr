"""Minimal multi-GPU helpers for sharding CG influence scoring.

CG scoring is embarrassingly parallel: each pool/target example's gradient is
independent, so we shard the work across GPUs and assemble the result with a
single all-reduce. This is *not* DDP training — there is no gradient sync; each
rank holds its own model replica and computes independent per-example grads.

Launch the entry point under torchrun:  torchrun --nproc_per_node=N -m ...
With no torchrun env (WORLD_SIZE<=1) everything degrades to single-process and
the helpers are no-ops, so the single-GPU path is unchanged.
"""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Init the process group from torchrun env vars. Returns (rank, world, local_rank).

    No-op (returns 0,1,0) when not launched under torchrun.

    Uses a long timeout and an immediate warm-up all-reduce: CG scoring runs a
    LONG (tens of minutes) generation phase with no collectives, and ranks finish
    their shards at different times. NCCL builds its communicator lazily on the
    first collective — if that lands after a long, skewed compute phase, the
    straggler blows past the default 10-min timeout. Warming up the communicator
    here (while all ranks are synced at startup) avoids that entirely.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(hours=6),  # tolerate long, skewed generation between collectives
        )
    # Warm up the communicator NOW, while ranks are synchronized.
    warm = torch.zeros(1, device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(warm)
    dist.barrier()
    return dist.get_rank(), dist.get_world_size(), local_rank


def dist_info() -> tuple[int, int, int]:
    """(rank, world, local_rank) — works whether or not the group is initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", "0"))
    return 0, 1, 0


def is_main() -> bool:
    return dist_info()[0] == 0


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce across ranks (no-op when world==1)."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def barrier() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
