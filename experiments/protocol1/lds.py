"""Stage 4 — assemble the Protocol-1 LDS report (numpy-only, runs anywhere).

Correlates, per (REFERENCE CHECKPOINT, CONTINUATION HORIZON), the influence
predicted at the checkpoint with the reward measured after continuing k steps:

  predicted ghat(S_m)  = sum_{i in S_m} IF[i]        (fixed-size subsets, so the
                                                      plain sum == visit-weighted)
  measured  y_m(c, k)  = target reward after k continuation steps from
                         checkpoint c on subset S_m

Layout: subset runs under <runs-root>/step<c>/subset_<m>/target_eval.json, whose
"horizons" block holds one per-target reward vector per eval step k (train.py's
HorizonEvalCallback; a legacy flat file counts as its single final horizon; a
flat <runs-root>/subset_<m> layout is matched to every variant). Influence
artifacts are step-suffixed, so each variant@step row is correlated ONLY against
retrains from that step. Reading the table:
  fix a horizon, scan steps    -> where in training is IF most predictive
  fix a step, scan horizons    -> how far ahead the prediction stays good (drift)

Two granularities per (variant, step, horizon):
  pooled LDS       Spearman over m of (sum_i IF_scores[S_m],  mean_t y[m,t])
  per-target LDS   TRAK-style: for each target t, Spearman over m of
                   (sum_i IF_matrix[t, S_m],  y[m,t]) — mean +/- std over targets
                   (zero-variance targets excluded)

Plus, per (step, horizon), a difficulty-confound baseline: predict y from the
subset's mean base-model band_pass_rate (no influence at all). A variant is only
interesting above this line.

  python -m experiments.protocol1.lds --ref-dir <...>/p1_ref --runs-root <...>/p1_runs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks for ties (no scipy dependency)."""
    def rank(x):
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        avg = (cum - 1 + cum - counts) / 2.0  # average rank within each tie group
        return avg[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def read_eval(path: Path) -> dict[int, np.ndarray]:
    """{horizon: per_target[T]} from one target_eval.json (legacy flat = 1 horizon)."""
    d = json.loads(path.read_text())
    if "horizons" in d:
        return {int(k): np.asarray(v["per_target"], dtype=np.float64)
                for k, v in d["horizons"].items()}
    return {int(d.get("steps") or -1): np.asarray(d["per_target"], dtype=np.float64)}


def collect_measured(run_dir: Path, M: int) -> dict[int, tuple[np.ndarray, list[int]]]:
    """{horizon: (y[Md, T], done ids)} over subset_<m> dirs; horizons with <3 runs
    are dropped (nothing to correlate)."""
    per_h: dict[int, dict[int, np.ndarray]] = {}
    for m in range(M):
        p = run_dir / f"subset_{m}" / "target_eval.json"
        if not p.exists():
            continue
        for k, rates in read_eval(p).items():
            per_h.setdefault(k, {})[m] = rates
    out = {}
    for k, by_m in sorted(per_h.items()):
        if len(by_m) >= 3:
            done = sorted(by_m)
            out[k] = (np.stack([by_m[m] for m in done]), done)
    return out


def collect_measured_by_step(runs_root: Path, M: int) -> dict[int | None, dict]:
    """{ref_step: {horizon: (y, done)}} from step<c>/ subdirs; flat layout -> key None."""
    out: dict[int | None, dict] = {}
    for d in sorted(runs_root.glob("step*")):
        m = re.fullmatch(r"step(\d+)", d.name)
        if m and (got := collect_measured(d, M)):
            out[int(m.group(1))] = got
    if not out and (got := collect_measured(runs_root, M)):
        out[None] = got                      # legacy flat layout: single unlabeled sweep
    if not out:
        raise SystemExit(f"no step<c>/subset_<m>/target_eval.json (or flat subset_<m>/) "
                         f"with >=3 measured runs under {runs_root}")
    return out


def find_variants(ref_dir: Path) -> dict[str, dict]:
    """{variant@step: {matrix: Path, rows: Path|None, scores: Path|None, step: int|None}}
    for every scored influence artifact under <ref-dir>/influence/. Keyed per
    (dir, step) so the same variant scored at several ref checkpoints reports as
    separate rows."""
    out: dict[str, dict] = {}
    for mat in sorted((ref_dir / "influence").rglob("*_if_target_matrix_step*.npy")):
        d = mat.parent
        step_s = mat.stem.rsplit("step", 1)[-1]
        step = int(step_s) if step_s.isdigit() else None
        name = (d.name if d.name != "influence" else mat.stem) + f"@step{step_s}"
        scores = sorted(d.glob(f"*_if_scores_step{step_s}.npy"))
        rows = Path(str(mat).replace("_if_target_matrix_", "_if_target_rows_"))
        out[name] = {"matrix": mat, "rows": rows if rows.exists() else None,
                     "scores": scores[0] if scores else None, "step": step}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-1 LDS report (per ref checkpoint x horizon).")
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="reference run dir (containing influence/<variant>/)")
    ap.add_argument("--runs-root", type=Path, required=True,
                    help="dir containing step<c>/subset_<m>/ continuation runs")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <runs-root>/lds_report.json")
    args = ap.parse_args(argv)

    meta = json.loads((args.data_dir / "lds_meta.json").read_text())
    subsets = np.load(args.data_dir / "subsets.npy")          # [M, k]
    pool = [json.loads(l) for l in (args.data_dir / "pool.jsonl").open()]
    band = np.asarray([r.get("band_pass_rate") or 0.0 for r in pool], dtype=np.float64)
    M, k = subsets.shape
    measured = collect_measured_by_step(args.runs_root, M)
    print(f"pool {len(pool)} | subsets manifest M={M} (k={k}, alpha={meta['alpha']})")

    report: dict = {"meta": meta, "measured": {}, "rows": []}
    rows_out: list[tuple] = []

    def key(step):  # None (flat) sorts last
        return (step is None, step)

    for step in sorted(measured, key=key):
        report["measured"][str(step)] = {}
        for horizon, (y, done) in measured[step].items():
            sub = subsets[done]
            y_mean = y.mean(axis=1)
            base_lds = spearman(np.array([band[s].mean() for s in sub]), y_mean)
            print(f"  step {step} k={horizon}: {len(done)}/{M} measured | mean target "
                  f"reward {y_mean.mean():.4f} (std over subsets {y_mean.std():.4f})"
                  + ("  [WARNING: no variance]" if y_mean.std() < 1e-6 else ""))
            report["measured"][str(step)][str(horizon)] = {
                "n_measured": len(done),
                "mean_reward": {"mean": float(y_mean.mean()), "std": float(y_mean.std())},
                "baseline_band_rate_pooled_lds": base_lds,
            }
            rows_out.append((step, horizon, "baseline: subset mean base pass rate",
                             base_lds, None, None, None, len(done)))

    for name, files in sorted(find_variants(args.ref_dir).items()):
        by_horizon = measured.get(files["step"], measured.get(None))
        if not by_horizon:
            print(f"  (skip {name}: no measured runs for step {files['step']})")
            continue
        mat = np.load(files["matrix"])                        # [T, N] per-target influence
        scores = (np.load(files["scores"]) if files["scores"] is not None
                  else mat.mean(axis=0))                      # [N] pooled scores
        tr = (np.load(files["rows"]).astype(int) if files["rows"] is not None
              else np.arange(mat.shape[0]))

        for horizon, (y, done) in by_horizon.items():
            sub, y_mean, T = subsets[done], y.mean(axis=1), y.shape[1]
            pooled = spearman(np.array([scores[s].sum() for s in sub]), y_mean)
            per_t = []
            for r, t in enumerate(tr):
                if t >= T or y[:, t].std() == 0:
                    continue
                ghat_t = np.array([mat[r, s].sum() for s in sub])
                if ghat_t.std() > 0:
                    per_t.append(spearman(ghat_t, y[:, t]))
            pt = np.asarray(per_t)
            row = {
                "step": files["step"], "horizon": horizon, "variant": name,
                "pooled_lds": pooled,
                "per_target_lds_mean": (float(pt.mean()) if pt.size else None),
                "per_target_lds_std": (float(pt.std()) if pt.size else None),
                "n_targets_used": int(pt.size), "n_measured": len(done),
                "matrix": str(files["matrix"]),
            }
            report["rows"].append(row)
            rows_out.append((files["step"], horizon, name, pooled,
                             row["per_target_lds_mean"], row["per_target_lds_std"],
                             int(pt.size), len(done)))

    rows_out.sort(key=lambda r: (key(r[0]), r[1], r[2]))
    print(f"\n{'step':>5} {'k':>5} {'variant':<42} {'pooled':>8} {'per-tgt':>8} {'std':>6} {'T':>4} {'Md':>4}")
    last = object()
    for step, horizon, name, pooled, ptm, pts, n, md in rows_out:
        if step != last and last is not object():
            print()
        last = step
        print(f"{(step if step is not None else '-'):>5} {horizon:>5} {name:<42} {pooled:>+8.3f} "
              f"{(f'{ptm:+.3f}' if ptm is not None else '   -'):>8} "
              f"{(f'{pts:.3f}' if pts is not None else '  -'):>6} "
              f"{(n if n is not None else '-'):>4} {md:>4}")
    if not report["rows"]:
        print("(no influence variants found — run score_ref first; expected "
              f"matrices under {args.ref_dir / 'influence'})")

    out = args.out or (args.runs_root / "lds_report.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
