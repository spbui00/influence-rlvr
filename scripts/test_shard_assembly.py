"""Deterministic correctness check for the CG-sharding assembly.

The sharding in `_run_cg` is: each rank computes a disjoint slice of the
H-rows (over targets) and the score matrix (over pool), writes them into a
zero-filled buffer, and one all-reduce(SUM) assembles the full result on every
rank. This script exercises that EXACT pattern with a deterministic synthetic
"gradient" (no model, no generation), so the result must be **identical for any
world size** — which proves the index-split + all-reduce plumbing is correct,
independent of the (non-deterministic) GPU generation that confounds the
end-to-end fingerprint.

    python -m scripts.test_shard_assembly                                  # world=1
    torchrun --standalone --nproc_per_node=2 -m scripts.test_shard_assembly
    torchrun --standalone --nproc_per_node=4 -m scripts.test_shard_assembly

All must print the SAME checksum and PASS.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.dist_utils import all_reduce_sum_, cleanup, dist_info, init_distributed, is_main

N_TARGET = 5
N_TRAIN = 23
D = 7  # stand-in gradient dimension


def _fake_grad(idx: int, salt: int) -> np.ndarray:
    """Deterministic 'gradient' vector for item `idx` (no randomness)."""
    return np.array([np.sin(0.13 * idx + 0.31 * d + salt) for d in range(D)], dtype=np.float64)


def main() -> None:
    init_distributed()
    rank, world, local_rank = dist_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── H-solve, sharded over targets (mirrors the target loop) ───────────────
    H = torch.zeros(N_TARGET, D, device=device, dtype=torch.float64)
    for j in range(rank, N_TARGET, world):
        H[j] = torch.from_numpy(_fake_grad(j, salt=1)).to(device)
    all_reduce_sum_(H)

    # ── scoring, sharded over pool (mirrors the pool loop) ────────────────────
    matrix = np.zeros((N_TARGET, N_TRAIN), dtype=np.float64)
    for i in range(rank, N_TRAIN, world):
        g = torch.from_numpy(_fake_grad(i, salt=2)).to(device)
        matrix[:, i] = (H @ g).cpu().numpy()
    if world > 1:
        mt = torch.from_numpy(matrix).to(device)
        all_reduce_sum_(mt)
        matrix = mt.cpu().numpy()
    scores = matrix.mean(axis=0)

    # ── unsharded reference (what world=1 single-process would compute) ───────
    Href = np.stack([_fake_grad(j, salt=1) for j in range(N_TARGET)])
    ref = np.stack([Href @ _fake_grad(i, salt=2) for i in range(N_TRAIN)]).T.mean(axis=0)

    ok = bool(np.allclose(scores, ref, atol=1e-9))
    if is_main():
        print(f"world={world}: checksum sum={scores.sum():.10f} "
              f"max|Δ|={np.abs(scores - ref).max():.2e} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("Sharding assembly mismatch — plumbing bug!")
    cleanup()


if __name__ == "__main__":
    main()
