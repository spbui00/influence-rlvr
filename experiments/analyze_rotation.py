"""Does dynamic IF re-ranking actually rotate the selected set, or re-pick the same top?

Reads the per-window rankings saved by run_if_prune (influence/ranked_order_step*.npy)
and reports, at the per-window pick size, the carry-over between consecutive windows
and the union across all windows. The whole static-vs-dynamic question turns on this:

  high carry-over / union ~= one window  -> NOT rotating; demotion isn't firing at this
                                            cadence/lr -> concentration (overfit a fixed slice).
  low carry-over / union >> one window   -> rotating as intended; flat held-out is elsewhere.

  srun --environment=pt --account=a0133 --partition=normal --time=00:05:00 --mem=16G \
    python experiments/analyze_rotation.py --run-name fin_ifg_gold
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-root", default="./outputs")
    ap.add_argument("--top-n", type=int, default=960,
                    help="pick size to compare (≈ window_steps × prompts_per_step)")
    args = ap.parse_args(argv)

    if_dir = Path(args.output_root) / args.run_name / "influence"
    fs = sorted(glob.glob(str(if_dir / "ranked_order_step*.npy")),
                key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    if len(fs) < 2:
        print(f"need >=2 ranked_order files in {if_dir}, found {len(fs)}: {fs}")
        return

    N = args.top_n
    tops = {int(re.search(r"step(\d+)", f).group(1)): set(np.load(f)[:N].tolist()) for f in fs}
    steps = sorted(tops)
    print(f"windows scored at steps {steps} (comparing top-{N} of each)\n")

    print("consecutive windows:")
    for a, b in zip(steps, steps[1:]):
        carry = len(tops[a] & tops[b]) / N
        print(f"  step{a:>3} -> step{b:<3}   carry-over {carry:5.0%}   turnover {1 - carry:5.0%}")

    first_last = len(tops[steps[0]] & tops[steps[-1]]) / N
    print(f"\nfirst -> last  (step{steps[0]} -> step{steps[-1]}): carry-over {first_last:.0%}")

    union = set().union(*tops.values())
    print(f"\nunion of every window's top-{N}: {len(union)} unique prompts "
          f"({len(union) / N:.1f}x one window)")
    print(f"  reference: no rotation -> {N};  full rotation -> ~{N * len(steps)}")
    verdict = ("CONCENTRATION (same slice re-picked) — demotion not firing"
               if len(union) < 1.5 * N else
               "ROTATING (set turns over) — dynamic re-ranking is working")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
