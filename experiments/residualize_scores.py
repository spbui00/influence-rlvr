"""Residualize an influence-score vector against answer-FORMAT features.

The gold-gradient influence was found to be largely an answer-format filter
(short / Float answers), not semantic transfer. This regresses the per-example
influence score on observable format proxies — answer length, answer_type,
domain — and keeps the RESIDUAL ranking: the influence that is NOT explained by
format. It answers, cheaply and post-hoc: is there any content signal hiding
under the format?

  high R^2 (format explains most of the score) -> influence ~= format filter.
  residual still selects structured / domain-balanced data -> content signal exists,
    and the gradient-space fixes (#2 common-mode projection, reasoning target) are
    worth running. The saved residual .npy is directly LDS-scorable (same pool order),
    so feed it to `lds.py score` once a properly-specified sweep exists.

  python -m experiments.residualize_scores --run-name xdomain_phys_gold_c \
      --scores outputs/xdomain_phys_gold_c/influence/step10/tracin_adam_if_scores_step10.npy

Torch-free (numpy + datasets); runs locally.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import load_train_pool


def _ols_r2(X, y):
    """Return (residual, R^2) for an OLS fit y ~ X (X already has an intercept col)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return resid, (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def _design(n, groups):
    """Stack [intercept | selected feature groups] into a design matrix."""
    cols = [np.ones(n)]
    for g in groups:
        cols.extend(g)
    return np.column_stack(cols)


def _onehot(values):
    levels = sorted(set(values))
    return [np.array([1.0 if v == lv else 0.0 for v in values]) for lv in levels], levels


def _compose(idx, alen, atype, cat):
    return (f"len(mean={alen[idx].mean():.1f} med={np.median(alen[idx]):.0f}) "
            f"domains={dict(Counter(cat[i] for i in idx))} "
            f"types={Counter(atype[i] for i in idx).most_common(3)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", default="./outputs")
    p.add_argument("--scores", required=True, help="per-example influence .npy (pool order)")
    p.add_argument("--keep-fraction", type=float, default=0.25,
                   help="top-fraction selection to characterize")
    p.add_argument("--ranked-order", default=None,
                   help="optional ranked_order_step*.npy to assert pool/score alignment")
    args = p.parse_args(argv)

    cfg = ExperimentConfig.load(Path(args.output_root).expanduser().resolve()
                                / args.run_name / "config.json")
    scores = np.load(args.scores).astype(np.float64)
    pool = load_train_pool(cfg)
    assert len(pool) == len(scores), f"pool {len(pool)} != scores {len(scores)}"

    # Alignment guard: argsort(-scores) must equal the saved selection order.
    if args.ranked_order:
        order = np.load(args.ranked_order)
        assert np.array_equal(np.argsort(-scores), order), \
            "scores do not align with ranked_order — pool rebuild mismatch!"
        print("alignment guard: argsort(-scores) == ranked_order  ✓")

    n = len(scores)
    alen = np.array([len(str(s)) for s in pool["solution"]], dtype=np.float64)
    atype = list(pool["answer_type"])
    cat = list(pool["category"])
    len_feats = [np.log1p(alen), alen]
    type_feats, _ = _onehot(atype)
    dom_feats, _ = _onehot(cat)

    # Incremental R^2: which format axis explains the influence score?
    print(f"\n==== residualize {Path(args.scores).name} (n={n}) ====")
    print("variance of influence score explained by format features (R^2):")
    for label, groups in [("length only", [len_feats]),
                          ("answer_type only", [type_feats]),
                          ("domain only", [dom_feats]),
                          ("length+type", [len_feats, type_feats]),
                          ("ALL (len+type+domain)", [len_feats, type_feats, dom_feats])]:
        _, r2 = _ols_r2(_design(n, groups), scores)
        print(f"  {label:24s} R^2 = {r2:.3f}")

    resid, r2_all = _ols_r2(_design(n, [len_feats, type_feats, dom_feats]), scores)

    # Spearman(raw, residual): how much did residualizing reorder the pool?
    rr = np.argsort(np.argsort(scores)).astype(np.float64)
    rs = np.argsort(np.argsort(resid)).astype(np.float64)
    rr -= rr.mean(); rs -= rs.mean()
    sp = float((rr * rs).sum() / np.sqrt((rr**2).sum() * (rs**2).sum()))
    print(f"\nSpearman(raw rank, residual rank) = {sp:+.3f}   "
          f"(low => format was most of the ranking)")

    # Selection shift: does residualizing de-bias the kept set?
    k = max(1, round(args.keep_fraction * n))
    raw_top = np.argsort(-scores)[:k]
    res_top = np.argsort(-resid)[:k]
    overlap = len(set(raw_top.tolist()) & set(res_top.tolist())) / k
    catarr = np.array(cat)
    print(f"\ntop-{args.keep_fraction:.0%} selection ({k} of {n}):  raw∩residual overlap = {overlap:.1%}")
    print(f"  RAW      kept: {_compose(raw_top, alen, atype, catarr)}")
    print(f"  RESIDUAL kept: {_compose(res_top, alen, atype, catarr)}")

    out = Path(args.scores).with_name(Path(args.scores).stem + "_resid.npy")
    np.save(out, resid)
    print(f"\nsaved residual scores -> {out}  (LDS-scorable; format R^2={r2_all:.3f})")


if __name__ == "__main__":
    main()
