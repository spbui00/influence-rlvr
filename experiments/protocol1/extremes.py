"""Stage 5 — the EXTREMES probe: does selecting by influence change outcomes?

The random-subset LDS was ≈0, but random 50% subsets administer a nearly constant
dose of tail prompts (top-50 content varies only ±3.5 across subsets), so a
correct tail ranking is invisible to it. The maximum-contrast probe trains on the
top-k and bottom-k prompts BY EACH SCORE VARIANT and compares the resulting target
reward against the 100 existing random-subset submodels from the same checkpoint
(their y-distribution is the null — no new random controls needed). k = 500
matches the LDS subset size, so the comparison is apples-to-apples.

  make    build subsets_extremes_step<C>.npy + meta from every score variant found
          under <ref-dir>/influence (plus a base-pass-rate difficulty control row
          pair). Higher score = predicted MORE helpful (top = keep-best).
  report  z-score each extreme row's per-horizon reward against the random-subset
          distribution at the same (checkpoint, horizon).

  python -m experiments.protocol1.extremes make   --ref-dir outputs/p1_ref --step 100
  python -m experiments.protocol1.extremes report --runs-root outputs/p1_runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def find_score_vectors(ref_dir: Path, step: int) -> dict[str, np.ndarray]:
    """{variant_name: scores[N]} for every influence variant scored at `step`."""
    out = {}
    for f in sorted((ref_dir / "influence").rglob(f"*_if_scores_step{step}.npy")):
        out[f.parent.name] = np.load(f)
    return out


def make(args) -> None:
    pool = [json.loads(l) for l in (args.data_dir / "pool.jsonl").open()]
    n = len(pool)
    variants = find_score_vectors(args.ref_dir, args.step)
    # difficulty control: base-model pass rate ("top" = easiest first)
    variants["baseline_band_rate"] = np.asarray(
        [r.get("band_pass_rate") or 0.0 for r in pool], dtype=np.float64)
    if args.only:
        keep = {v.strip() for v in args.only.split(",") if v.strip()}
        missing = keep - set(variants)
        if missing:
            raise SystemExit(f"--only names not found: {sorted(missing)} "
                             f"(have: {sorted(variants)})")
        variants = {k: v for k, v in variants.items() if k in keep}
    if not variants:
        raise SystemExit(f"no *_if_scores_step{args.step}.npy under {args.ref_dir}/influence")

    # --tag makes ADD-ON manifests (e.g. raw-dot rows generated after the main batch
    # already started training) that never renumber earlier rows: separate files,
    # separate run-dir prefix, and `report` joins across all tags.
    suffix = f"_{args.tag}" if args.tag else ""
    run_prefix = f"row_{args.tag}" if args.tag else "row"

    rows, meta = [], []
    for name, s in variants.items():
        if s.shape != (n,):
            raise SystemExit(f"{name}: scores shape {s.shape} != pool {n}")
        order = np.argsort(-s, kind="stable")     # descending; zeros/ties keep index order
        for tail, idx in (("top", order[:args.k]), ("bottom", order[-args.k:])):
            meta.append({"row": len(rows), "step": args.step, "variant": name,
                         "tail": tail, "k": args.k, "run_prefix": run_prefix,
                         "score_mean": float(s[idx].mean()),
                         "n_zero_score": int((s[idx] == 0).sum())})
            rows.append(np.sort(idx))
    subs = np.stack(rows).astype(np.int32)

    out_npy = args.data_dir / f"subsets_extremes_step{args.step}{suffix}.npy"
    out_meta = args.data_dir / f"extremes_meta_step{args.step}{suffix}.json"
    np.save(out_npy, subs)
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{out_npy.name}: {len(rows)} rows x {args.k} (run_prefix={run_prefix})")
    for m in meta:
        print(f"  row {m['row']:>2}  {m['variant']:<28} {m['tail']:<6} "
              f"score_mean={m['score_mean']:+.2e}  zeros={m['n_zero_score']}")


def collect_random_null(runs_root: Path, step: int) -> dict[int, np.ndarray]:
    """{horizon: y_mean[Md]} across the existing random-subset submodels."""
    per_h: dict[int, list[float]] = {}
    for p in sorted((runs_root / f"step{step}").glob("subset_*/target_eval.json")):
        d = json.loads(p.read_text())
        for k, v in d.get("horizons", {}).items():
            per_h.setdefault(int(k), []).append(float(v["mean_reward"]))
    return {k: np.asarray(v) for k, v in per_h.items() if len(v) >= 10}


def report(args) -> None:
    out_rows = []
    for step in args.steps:
        metas = sorted(args.data_dir.glob(f"extremes_meta_step{step}*.json"))
        if not metas:
            continue
        meta = [m for p in metas for m in json.loads(p.read_text())]
        null = collect_random_null(args.runs_root, step)
        for m in meta:
            prefix = m.get("run_prefix", "row")
            p = (args.runs_root / "extremes" / f"step{step}" /
                 f"{prefix}_{m['row']}" / "target_eval.json")
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for k_s, v in sorted(d.get("horizons", {}).items(), key=lambda kv: int(kv[0])):
                k = int(k_s)
                y = float(v["mean_reward"])
                if k in null:
                    mu, sd = float(null[k].mean()), float(null[k].std())
                    z = (y - mu) / sd if sd > 0 else float("nan")
                    pct = float((null[k] < y).mean()) * 100
                else:
                    mu = sd = z = pct = float("nan")
                out_rows.append((step, m["variant"], m["tail"], k, y, mu, sd, z, pct))

    if not out_rows:
        raise SystemExit(f"no extremes results under {args.runs_root}/extremes "
                         f"(and/or no extremes_meta_step*.json in {args.data_dir})")
    print(f"{'step':>5} {'variant':<28} {'tail':<7} {'k':>4} {'y':>7} "
          f"{'rand mu':>8} {'sd':>6} {'z':>6} {'pctl':>6}")
    last = None
    for step, var, tail, k, y, mu, sd, z, pct in out_rows:
        if (step, var) != last and last is not None:
            print()
        last = (step, var)
        print(f"{step:>5} {var:<28} {tail:<7} {k:>4} {y:>7.4f} "
              f"{mu:>8.4f} {sd:>6.4f} {z:>+6.2f} {pct:>5.1f}%")
    out = args.runs_root / "extremes_report.json"
    out.write_text(json.dumps([
        {"step": s, "variant": v, "tail": t, "horizon": k, "y": y,
         "random_mean": mu, "random_sd": sd, "z": z, "percentile": pct}
        for s, v, t, k, y, mu, sd, z, pct in out_rows], indent=2) + "\n")
    print(f"\nreport -> {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-1 extremes probe (make / report).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--ref-dir", type=Path, required=True)
    mk.add_argument("--data-dir", type=Path, default=DATA_DIR)
    mk.add_argument("--step", type=int, required=True)
    mk.add_argument("-k", type=int, default=500, help="matches the LDS subset size")
    mk.add_argument("--only", default="",
                    help="comma list of variant names to include (default: all found)")
    mk.add_argument("--tag", default="",
                    help="ADD-ON manifest: writes subsets_extremes_step<C>_<tag>.npy with "
                         "run_prefix row_<tag> — never renumbers rows of earlier manifests")
    rp = sub.add_parser("report")
    rp.add_argument("--runs-root", type=Path, required=True)
    rp.add_argument("--data-dir", type=Path, default=DATA_DIR)
    rp.add_argument("--steps", type=int, nargs="+", default=[100, 200])
    args = ap.parse_args(argv)
    make(args) if args.cmd == "make" else report(args)


if __name__ == "__main__":
    main()
