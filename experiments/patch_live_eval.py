"""Splice re-evaluated checkpoint JSONs back into live_eval.csv.

The in-training LiveEvalCallback logs held-out accuracy at `live_eval_examples`
samples (outputs/<run>/live_eval.csv). When a run resumes without carrying
LIVE_EVAL_EXAMPLES, that subsample silently drops (e.g. 256 -> 64), making the
post-resume rows noisy. To repair them, re-grade those checkpoints at the larger
n with experiments/cluster/cross_eval.slurm (writes outputs/<run>/eval/
eval_step<N>.json), then run this to overwrite the matching CSV rows.

For every step that has an eval_step<N>.json, the CSV row is rewritten from the
JSON (n, accuracy, per_category); rows without a JSON are left as-is. The output
byte-matches what LiveEvalCallback would have written (accuracy column %.6f,
per_category as a full-precision JSON string). The original CSV is copied to
live_eval.csv.bak first.

Run:
    python -m experiments.patch_live_eval --run-name xdomain_phys_gold_c
    python -m experiments.patch_live_eval --run-name xdomain_phys_baseline_c --dry-run
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Splice re-eval JSONs into live_eval.csv.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", default="./outputs")
    p.add_argument("--benchmark", default="webinstruct_test",
                   help="Benchmark key in the eval JSON to read accuracy from.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the patched CSV without writing it.")
    args = p.parse_args(argv)

    run_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    csv_path = run_dir / "live_eval.csv"
    eval_dir = run_dir / "eval"
    if not csv_path.exists():
        raise FileNotFoundError(f"No live_eval.csv under {run_dir}")

    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]

    out_rows: list[list[str]] = []
    patched = 0
    for r in rows:
        step = int(r[0])
        ej = eval_dir / f"eval_step{step}.json"
        if not ej.exists():
            out_rows.append(r)
            continue
        res = json.loads(ej.read_text())["results"][args.benchmark]
        n = int(res["n"])
        acc = float(res["accuracy"])
        per_cat = res.get("per_category") or {}
        # Mirror LiveEvalCallback.writerow exactly: accuracy %.6f, per_category as a
        # full-precision json string (csv.writer adds the outer quoting).
        new_row = [str(step), str(n), f"{acc:.6f}", json.dumps(per_cat)]
        if new_row != r:
            patched += 1
        out_rows.append(new_row)

    print(f"{args.run_name}: {patched}/{len(rows)} rows patched from {eval_dir}")
    if args.dry_run:
        # Re-quote through csv for an honest byte-accurate preview.
        buf = io.StringIO()
        cw = csv.writer(buf)
        cw.writerow(header)
        cw.writerows(out_rows)
        print(buf.getvalue(), end="")
        return

    bak = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, bak)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)
    print(f"  wrote {csv_path} (backup at {bak})")


if __name__ == "__main__":
    main()
