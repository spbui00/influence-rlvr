#!/usr/bin/env python3
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from influence_rlvr.eval_stats import (
    align_paired_scores,
    mcnemar_p_value_cc,
    paired_accuracy_bootstrap,
    summarize_binary_accuracy,
    wilson_ci,
)


def _read_seed(run_dir: Path) -> int | None:
    meta_b = run_dir / "eval_baseline.json"
    if meta_b.is_file():
        data = json.loads(meta_b.read_text())
        md = data.get("metadata") or {}
        if "seed" in md:
            return int(md["seed"])
    rc = run_dir / "run_config.json"
    if rc.is_file():
        cfg = json.loads(rc.read_text())
        if "seed" in cfg:
            return int(cfg["seed"])
    return None


def _print_single_run(run_dir: Path, n_boot: int, confidence: float, seed: int) -> None:
    run_dir = run_dir.resolve()
    base = run_dir / "eval_baseline.json"
    post = run_dir / "eval_after_train.json"
    if not base.is_file():
        raise SystemExit(f"missing {base}")
    if not post.is_file():
        raise SystemExit(f"missing {post}")

    y0, y1 = align_paired_scores(base, post)
    y0i = (y0 >= 0.5).astype(np.int64)
    y1i = (y1 >= 0.5).astype(np.int64)

    m0, k0, n0 = summarize_binary_accuracy(y0)
    m1, k1, n1 = summarize_binary_accuracy(y1)
    w0_lo, w0_hi = wilson_ci(k0, n0, confidence=confidence)
    w1_lo, w1_hi = wilson_ci(k1, n1, confidence=confidence)

    rng = np.random.default_rng(seed)
    d_pt, d_lo, d_hi = paired_accuracy_bootstrap(
        y0, y1, n_boot=n_boot, confidence=confidence, rng=rng
    )
    b, c, p_mc = mcnemar_p_value_cc(y0i, y1i)

    print(f"run_dir: {run_dir}")
    print(f"n_examples (paired): {n0}")
    print(
        f"baseline accuracy: {m0:.4f}  Wilson {int(confidence*100)}% CI "
        f"[{w0_lo:.4f}, {w0_hi:.4f}]"
    )
    print(
        f"post accuracy:     {m1:.4f}  Wilson {int(confidence*100)}% CI "
        f"[{w1_lo:.4f}, {w1_hi:.4f}]"
    )
    print(
        f"mean paired delta (post - baseline): {d_pt:+.4f}  "
        f"bootstrap {int(confidence*100)}% CI [{d_lo:+.4f}, {d_hi:+.4f}]"
    )
    print(f"McNemar (continuity-corrected): b={b} (base right→post wrong), "
          f"c={c} (base wrong→post right), p≈{p_mc:.4g}")


def _collect_run_dirs(patterns: list[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if not matches:
            print(f"warning: no matches for glob {pat!r}", file=sys.stderr)
        for m in matches:
            p = Path(m).resolve()
            if p.is_dir() and p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def _print_multi_run(
    run_dirs: list[Path], n_boot: int, confidence: float, rng_seed: int
) -> None:
    rows: list[tuple[int | None, float, float, float]] = []
    for rd in run_dirs:
        base = rd / "eval_baseline.json"
        post = rd / "eval_after_train.json"
        if not base.is_file() or not post.is_file():
            print(f"skip (missing eval JSON): {rd}", file=sys.stderr)
            continue
        y0, y1 = align_paired_scores(base, post)
        m0, _, _ = summarize_binary_accuracy(y0)
        m1, _, _ = summarize_binary_accuracy(y1)
        delta = float(np.mean(y1 - y0))
        sd = _read_seed(rd)
        rows.append((sd, m0, m1, delta))

    if not rows:
        raise SystemExit("no valid run directories")

    print(f"{'seed':>6}  {'baseline':>10}  {'post':>10}  {'delta':>10}")
    print("-" * 44)
    for sd, m0, m1, d in rows:
        ss = str(sd) if sd is not None else "?"
        print(f"{ss:>6}  {m0:10.4f}  {m1:10.4f}  {d:+10.4f}")

    deltas = np.array([r[3] for r in rows], dtype=np.float64)
    print("-" * 44)
    if len(deltas) > 1:
        std_d = float(np.std(deltas, ddof=1))
        print(
            f"across {len(rows)} run(s): mean(delta)={float(np.mean(deltas)):+.4f}  "
            f"std(delta)={std_d:.4f}"
        )
    else:
        print(f"single run delta={deltas[0]:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compare GSM8K eval_baseline.json vs eval_after_train.json "
            "(Wilson CIs, paired bootstrap on mean accuracy delta, McNemar)."
        )
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory containing eval_baseline.json and eval_after_train.json",
    )
    p.add_argument(
        "--multi-run",
        nargs="+",
        metavar="GLOB",
        default=None,
        help=(
            "One or more glob patterns for run dirs (each dir must contain both "
            "eval JSON files). Example: outputs/nemotron_math_s*/rlvr-output"
        ),
    )
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seed for paired bootstrap resampling.",
    )
    args = p.parse_args()

    if args.run_dir is None and args.multi_run is None:
        p.error("provide --run-dir or --multi-run")
    if args.run_dir is not None and args.multi_run is not None:
        p.error("use either --run-dir or --multi-run, not both")

    if args.multi_run is not None:
        run_dirs = _collect_run_dirs(list(args.multi_run))
        if not run_dirs:
            raise SystemExit("no directories matched --multi-run patterns")
        if len(run_dirs) == 1:
            _print_single_run(
                run_dirs[0], args.bootstrap_samples, args.confidence, args.rng_seed
            )
        else:
            _print_multi_run(
                run_dirs, args.bootstrap_samples, args.confidence, args.rng_seed
            )
    else:
        _print_single_run(
            args.run_dir, args.bootstrap_samples, args.confidence, args.rng_seed
        )


if __name__ == "__main__":
    main()
