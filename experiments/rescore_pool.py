"""Re-score a checkpoint's training pool with a method VARIANT, for LDS comparison.

Loads the run's SAVED config.json (so cosine / difficulty / target carve / method
match the original scoring exactly) and overrides only the variant knob(s) — then
scores one checkpoint and saves a per-example score .npy under a fresh sub-dir.
The output is directly LDS-scorable (same pool order) against the original scores.

Gold scoring needs no servers (teacher-forced; 1 GPU). The Adam preconditioner for
tracin-adam is read from the checkpoint's optimizer.pt on disk.

  # #2 common-mode projection re-score at the same checkpoint the gold scores used:
  python -m experiments.rescore_pool --run-name xdomain_phys_gold_c \
      --checkpoint-step 10 --if-common-mode top-pc --tag proj_toppc
  python -m experiments.rescore_pool --run-name xdomain_phys_gold_c \
      --checkpoint-step 10 --if-common-mode pool-mean --tag proj_poolmean
  # -> outputs/xdomain_phys_gold_c/influence/<tag>_step10/tracin_adam_if_scores_step10.npy
"""
import argparse
from pathlib import Path

from .config import ExperimentConfig
from .data import load_if_target_set, load_train_pool
from .influence import compute_pool_influence


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", default="./outputs")
    p.add_argument("--checkpoint-step", type=int, required=True)
    p.add_argument("--tag", default="rescore", help="sub-dir name: influence/<tag>_step<N>/")
    # variant overrides (only these differ from the saved config)
    p.add_argument("--if-common-mode", choices=["top-pc", "pool-mean"], default=None,
                   help="strip a format common-mode direction before collapsing H")
    p.add_argument("--if-method", default=None, help="override cfg.if_method (e.g. cg)")
    p.add_argument("--if-grad", default=None, choices=["gold", "rollout"],
                   help="override cfg.if_grad (rollout needs a gen server — gold is serverless)")
    p.add_argument("--if-cosine", dest="if_cosine", action="store_true", default=None)
    p.add_argument("--no-if-cosine", dest="if_cosine", action="store_false")
    args = p.parse_args(argv)

    from influence_rlvr import detect_device, load_adapter_checkpoint
    from .train import build_model

    cfg = ExperimentConfig.load(Path(args.output_root).expanduser().resolve()
                                / args.run_name / "config.json")
    if args.if_common_mode:
        cfg.if_common_mode = args.if_common_mode
    if args.if_method:
        cfg.if_method = args.if_method
    if args.if_grad:
        cfg.if_grad = args.if_grad
    if args.if_cosine is not None:
        cfg.if_cosine = args.if_cosine

    device = detect_device()
    model, tok = build_model(cfg, device)
    ckpt = cfg.grpo_output_dir / f"checkpoint-{args.checkpoint_step}"
    if not ckpt.is_dir():
        raise SystemExit(f"[rescore] no checkpoint at {ckpt}")
    load_adapter_checkpoint(model, str(ckpt))

    pool = load_train_pool(cfg)
    target = load_if_target_set(cfg)
    save_dir = cfg.run_dir / "influence" / f"{args.tag}_step{args.checkpoint_step}"
    print(f"[rescore] {args.run_name} ckpt-{args.checkpoint_step} | method={cfg.if_method} "
          f"grad={cfg.if_grad} cosine={cfg.if_cosine} "
          f"common_mode={cfg.if_common_mode or 'off'}")
    print(f"[rescore] pool={len(pool)} target={len(target)} -> {save_dir}")
    compute_pool_influence(cfg, model, tok, pool, target, device,
                           checkpoint_step=args.checkpoint_step, save_dir=save_dir)
    print(f"[rescore] done — scores under {save_dir}/")


if __name__ == "__main__":
    main()
