"""Linear Datamodeling Score (LDS) harness for influence methods.

LDS measures how well an attribution method predicts the effect of retraining on
random data subsets: sample M subsets of the train pool, retrain on each, measure
a target metric, and correlate the measured outcomes with the method's *predicted*
outcome g_hat(S) = sum_{i in S} score[i]. LDS = Spearman(g_hat, measured).

The expensive part (the retraining sweep) is DECOUPLED from scoring: the sweep
writes one measured outcome per subset to a JSONL, and `score` correlates ANY
per-example score vector against it. So the sweep runs once (over days, resumably)
and every influence method — gold / rollout / cg / future ones — is evaluated for
free afterward.

  make-subsets : deterministic manifest of M random subsets (shared by all variants/methods)
  make-extremes: 3-row manifest (top-k / random-k / bottom-k by a scores .npy) — the
                 maximum-contrast probe: if these don't separate on the measured
                 metric, no random-subset sweep at any alpha will.
  run          : retrain per subset + measure the target metric, appending one JSONL
                 row per subset. RESUMABLE — re-run the same command to continue.
                   --variant nll    : teacher-forced gold-NLL SFT, measure target gold-NLL.
                                      Cheap (no generation); the metric the gold influence
                                      NATIVELY predicts, so it's the matched faithfulness test.
                   --variant reward : GRPO on the subset, measure held-out verifier accuracy.
                                      Expensive (full RL per subset); what we actually care about.
  score        : LDS = Spearman(sum_{i in S} score[i], measured(S)) for any scores .npy.

  python -m experiments.lds make-subsets --run-name xdomain_phys_gold_c -M 200 --alpha 0.25
  python -m experiments.lds run  --variant nll    --run-name xdomain_phys_gold_c --ref-step 10
  python -m experiments.lds run  --variant reward --run-name xdomain_phys_gold_c --ref-step 10 --train-steps 60
  python -m experiments.lds score --variant nll   --run-name xdomain_phys_gold_c \
      --scores outputs/xdomain_phys_gold_c/influence/step10/tracin_adam_if_scores_step10.npy

`--subsets-tag foo` reads/writes subsets_foo.npy instead of subsets.npy, so new subset
designs (different alpha, extremes) never invalidate measured_*.jsonl files produced
under an older manifest. When set, the measured/score --tag defaults to the same value.

  python -m experiments.lds make-subsets  --run-name xdomain_phys_gold_c -M 200 --alpha 0.05 --subsets-tag a05
  python -m experiments.lds make-extremes --run-name xdomain_phys_gold_c --subsets-tag extremes \
      --scores outputs/xdomain_phys_gold_c/influence/step10/tracin_adam_if_scores_step10.npy

make-extremes and score are numpy-only, make-subsets additionally needs `datasets`;
all three run locally. run needs a GPU (experiments/cluster/lds.slurm).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .config import ExperimentConfig


# ── paths ────────────────────────────────────────────────────────────────────
def _lds_dir(cfg) -> Path:
    return cfg.run_dir / "lds"


def _subsets_path(cfg, tag: str = "") -> Path:
    return _lds_dir(cfg) / (f"subsets_{tag}.npy" if tag else "subsets.npy")


def _meta_path(cfg, tag: str = "") -> Path:
    return _lds_dir(cfg) / (f"subsets_meta_{tag}.json" if tag else "subsets_meta.json")


def _measured_path(cfg, variant: str, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return _lds_dir(cfg) / f"measured_{variant}{suffix}.jsonl"


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["subset_id"] for l in path.read_text().splitlines() if l.strip()}


# ── make-subsets ─────────────────────────────────────────────────────────────
def make_subsets(cfg, M: int, alpha: float, seed: int, subsets_tag: str = "") -> np.ndarray:
    from .data import load_train_pool
    n = len(load_train_pool(cfg))
    k = max(1, round(alpha * n))
    # Sequential draws from one stream: re-running with a larger M (same seed/alpha)
    # reproduces the first M subsets, so a sweep can be extended without re-measuring.
    rng = np.random.default_rng(seed)
    subs = np.stack([np.sort(rng.choice(n, size=k, replace=False)) for _ in range(M)]).astype(np.int32)
    d = _lds_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.save(_subsets_path(cfg, subsets_tag), subs)
    _meta_path(cfg, subsets_tag).write_text(json.dumps(
        {"pool_size": n, "M": M, "alpha": alpha, "subset_size": k, "seed": seed,
         "run_name": cfg.run_name}, indent=2))
    print(f"make-subsets: {M} subsets of {k}/{n} ({alpha:.0%}) -> {_subsets_path(cfg, subsets_tag)}")
    return subs


def make_extremes(cfg, scores_path: str, k: int, seed: int, subsets_tag: str) -> np.ndarray:
    """Top-k / random-k / bottom-k rows by a per-example score vector (higher = predicted
    more helpful, matching kept_step*.npy). Measured with the same `run` variants."""
    s = np.load(scores_path).astype(np.float64).ravel()
    n = s.size
    order = np.argsort(-s)
    rng = np.random.default_rng(seed)
    rows = {
        "top": np.sort(order[:k]),
        "random": np.sort(rng.choice(n, size=k, replace=False)),
        "bottom": np.sort(order[-k:]),
    }
    subs = np.stack(list(rows.values())).astype(np.int32)
    d = _lds_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.save(_subsets_path(cfg, subsets_tag), subs)
    _meta_path(cfg, subsets_tag).write_text(json.dumps(
        {"kind": "extremes", "pool_size": int(n), "rows": list(rows), "k": int(k),
         "seed": seed, "scores_file": str(scores_path), "run_name": cfg.run_name}, indent=2))
    print(f"make-extremes: rows={list(rows)} k={k}/{n} by {Path(scores_path).name} "
          f"-> {_subsets_path(cfg, subsets_tag)}")
    return subs


# ── variant A: gold-NLL SFT ──────────────────────────────────────────────────
def _encode_gold(tokenizer, samples, device):
    """Right-pad (prompt + \\boxed{gold} + eos) -> input_ids/attn/comp_mask [B,L].
    Mirrors influence_scoring.encode_batch so the NLL is identical to the gold scorer."""
    import torch
    from influence_rlvr.utils import tokenize_prompts_batch
    eos = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    rows = []
    for s in samples:
        _, p_ids, _ = tokenize_prompts_batch(tokenizer, [s["prompt"]], device)
        p = p_ids[0].tolist()
        gold = str(s.get("solution", "") or "")
        c = tokenizer(f"\\boxed{{{gold}}}", add_special_tokens=False)["input_ids"]
        if eos is not None:
            c = c + [int(eos)]
        rows.append((p, c))
    L = max(len(p) + len(c) for p, c in rows)
    B = len(rows)
    ii = torch.full((B, L), int(pad_id), dtype=torch.long, device=device)
    am = torch.zeros((B, L), dtype=torch.long, device=device)
    cm = torch.zeros((B, L), dtype=torch.long, device=device)
    for bi, (p, c) in enumerate(rows):
        seq = p + c
        ii[bi, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        am[bi, :len(seq)] = 1
        cm[bi, len(p):len(seq)] = 1
    return ii, am, cm


def _gold_nll_vec(model, tokenizer, samples, device):
    """Per-example mean gold-NLL [B] for a minibatch (grad flows to LoRA params)."""
    from influence_rlvr.gradients import gold_nll_per_example_functional
    params = dict(model.named_parameters())
    ii, am, cm = _encode_gold(tokenizer, samples, device)
    return gold_nll_per_example_functional(model, params, ii, am, cm)


def _target_nll(model, tokenizer, target, device, batch=16):
    import torch
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(target), batch):
            v = _gold_nll_vec(model, tokenizer, target[i:i + batch], device)
            tot += float(v.sum())
            n += int(v.numel())
    return tot / max(n, 1)


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _restore(model, snap):
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])


def run_nll(cfg, ref_step, train_steps, lr, mb, seed, epochs, tag, limit=None, subsets_tag=""):
    import torch
    from influence_rlvr import clear_cache, detect_device, load_adapter_checkpoint
    from .data import load_if_target_set, load_train_pool
    from .train import build_model
    device = detect_device()
    subs = np.load(_subsets_path(cfg, subsets_tag))
    model, tok = build_model(cfg, device)
    ref_ckpt = cfg.grpo_output_dir / f"checkpoint-{ref_step}"
    if ref_step and ref_ckpt.is_dir():
        load_adapter_checkpoint(model, str(ref_ckpt))
        print(f"[nll] reference = {ref_ckpt}")
    else:
        print(f"[nll] reference = base model (ref_step={ref_step})")
    ref = _snapshot(model)
    pool = load_train_pool(cfg)
    target = list(load_if_target_set(cfg))
    out = _measured_path(cfg, "nll", tag)
    done = _done_ids(out)
    todo = [j for j in range(len(subs)) if j not in done]
    if limit:
        todo = todo[:limit]
    # train_steps set -> fixed-step sampled minibatches (cheap diagnostic); else FULL-coverage
    # epochs (the matched test: g_hat sums influence over the WHOLE subset, so train on all of it).
    spec = f"train_steps={train_steps}" if train_steps else f"epochs={epochs} (full coverage)"
    print(f"[nll] {len(done)} done, {len(todo)} to run (ref_step={ref_step}, {spec}, lr={lr}, mb={mb}, "
          f"subsets={_subsets_path(cfg, subsets_tag).name})")
    with out.open("a") as f:
        for j in todo:
            _restore(model, ref)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
            idx = subs[j]
            rng = np.random.default_rng(seed * 1_000_003 + j)
            model.train()
            if train_steps:
                for _ in range(train_steps):
                    bi = rng.choice(idx, size=min(mb, len(idx)), replace=False)
                    loss = _gold_nll_vec(model, tok, [pool[int(t)] for t in bi], device).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            else:
                for _ in range(epochs):
                    perm = rng.permutation(idx)
                    for k in range(0, len(perm), mb):
                        chunk = perm[k:k + mb]
                        loss = _gold_nll_vec(model, tok, [pool[int(t)] for t in chunk], device).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
            model.eval()
            val = _target_nll(model, tok, target, device)
            f.write(json.dumps({"subset_id": int(j), "value": val, "metric": "gold_nll",
                                "higher_is_better": False, "ref_step": ref_step,
                                "train_steps": train_steps, "epochs": (None if train_steps else epochs),
                                "lr": lr, "mb": mb, "subset_size": int(idx.size),
                                "subsets_tag": subsets_tag}) + "\n")
            f.flush()
            print(f"[nll] subset {j}: target gold-NLL = {val:.5f}")
            del opt
            clear_cache(device)


# ── variant B: GRPO + held-out accuracy ──────────────────────────────────────
def run_reward(cfg, ref_step, train_steps, eval_examples, seed, tag, limit=None, subsets_tag=""):
    import torch
    import torch.distributed as dist
    from influence_rlvr import clear_cache, detect_device, load_adapter_checkpoint
    from trl import GRPOTrainer
    from .data import load_if_target_set, load_train_pool
    from .train import build_reward_funcs, build_model, make_grpo_config
    from .evaluate import score_examples

    os.environ.setdefault("WANDB_MODE", "disabled")
    # Single-GPU (plain python) or DDP (accelerate launch --num_processes N): every
    # rank trains; only rank 0 evals + writes; a barrier keeps ranks in lockstep.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    def _barrier():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    device = detect_device()
    subs = np.load(_subsets_path(cfg, subsets_tag))
    model, tok = build_model(cfg, device)
    ref_ckpt = cfg.grpo_output_dir / f"checkpoint-{ref_step}"
    if ref_step and ref_ckpt.is_dir():
        load_adapter_checkpoint(model, str(ref_ckpt))
        if is_main:
            print(f"[reward] reference = {ref_ckpt}")
    elif is_main:
        print(f"[reward] reference = base model (ref_step={ref_step})")
    ref = _snapshot(model)
    pool = load_train_pool(cfg)
    target = list(load_if_target_set(cfg))[:eval_examples]
    scratch = _lds_dir(cfg) / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    out = _measured_path(cfg, "reward", tag)
    done = _done_ids(out)
    todo = [j for j in range(len(subs)) if j not in done]
    if limit:
        todo = todo[:limit]
    if is_main:
        print(f"[reward] {len(done)} done, {len(todo)} to run (ref_step={ref_step}, "
              f"train_steps={train_steps}, eval_examples={len(target)}, world={world})")
    f = out.open("a") if is_main else None
    for j in todo:
        _restore(model, ref)
        sub_ds = pool.select([int(t) for t in subs[j]])
        args = make_grpo_config(cfg, max_steps=train_steps, shuffle=True)
        # Isolate from the real run dir + never checkpoint during the sweep.
        args.output_dir = str(scratch)
        args.save_strategy = "no"
        args.save_steps = 10 ** 9
        args.report_to = []
        args.seed = cfg.seed + seed  # vary for seed-averaging the reward LDS
        if world > 1:
            # cfg.grad_accum is the single-GPU spec; split it across ranks so
            # prompts/step (and thus the experiment) is invariant to world size.
            args.gradient_accumulation_steps = max(1, cfg.grad_accum // world)
        trainer = GRPOTrainer(model=model, reward_funcs=build_reward_funcs(cfg),
                              args=args, train_dataset=sub_ds, processing_class=tok)
        trainer.train()
        # Generation-friendly state for the post-train eval, then restore.
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        if is_main:
            assert f is not None
            acc = float(score_examples(target, model, tok, cfg, device, vllm=False)["accuracy"])
            f.write(json.dumps({"subset_id": int(j), "value": acc, "metric": "accuracy",
                                "higher_is_better": True, "ref_step": ref_step,
                                "train_steps": train_steps, "eval_examples": len(target),
                                "subset_size": int(subs.shape[1]),
                                "subsets_tag": subsets_tag}) + "\n")
            f.flush()
            print(f"[reward] subset {j}: target accuracy = {acc:.4f}")
        _barrier()  # non-main ranks wait out the rank-0 eval before the next subset
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        # vLLM server mode re-inits a weight-sync group per trainer; close it so the
        # next subset's trainer re-inits cleanly (same fix as run_if_prune).
        if is_main and getattr(cfg, "use_vllm", False) and getattr(cfg, "vllm_mode", "") == "server":
            try:
                trainer.vllm_generation.vllm_client.close_communicator()
            except Exception as e:
                print(f"  vllm teardown skipped ({type(e).__name__}: {e})")
        del trainer
        clear_cache(device)
    if f is not None:
        f.close()


# ── score (decoupled; any method) ────────────────────────────────────────────
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def score(cfg, variant, scores_paths, tag="", subsets_tag=""):
    subs = np.load(_subsets_path(cfg, subsets_tag))
    meta_p = _meta_path(cfg, subsets_tag)
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    rows = [json.loads(l) for l in _measured_path(cfg, variant, tag).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["subset_id"])
    ids = [r["subset_id"] for r in rows]
    hib = bool(rows[0]["higher_is_better"])
    measured = np.array([r["value"] for r in rows], dtype=np.float64)
    goodness = measured if hib else -measured  # align so higher = better target outcome
    if meta.get("kind") == "extremes":
        # 3 labeled rows, not an LDS: report measured value + predicted sum per row.
        labels = meta.get("rows") or [str(i) for i in range(len(subs))]
        print(f"extremes (variant={variant}, metric={rows[0]['metric']}, "
              f"higher_is_better={hib}, tag={tag or '-'}):")
        for sp in scores_paths:
            s = np.load(sp)
            print(f"  {Path(sp).name}")
            for r in rows:
                j = r["subset_id"]
                print(f"    {labels[j]:8s} measured = {r['value']:.5f}   "
                      f"ghat = {float(s[subs[j]].sum()):+.4f}")
        return
    print(f"LDS (variant={variant}, metric={rows[0]['metric']}, {len(rows)} subsets, "
          f"higher_is_better={hib}):")
    for sp in scores_paths:
        s = np.load(sp)
        ghat = np.array([s[subs[i]].sum() for i in ids], dtype=np.float64)
        print(f"  {Path(sp).name:48s} LDS = {_spearman(ghat, goodness):+.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", default="./outputs")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("make-subsets")
    a.add_argument("-M", "--num-subsets", type=int, default=200)
    a.add_argument("--alpha", type=float, default=0.25, help="subset fraction of the pool")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--subsets-tag", default="", help="write subsets_<tag>.npy instead of subsets.npy")

    e = sub.add_parser("make-extremes")
    e.add_argument("--scores", required=True, help="per-example score .npy (higher = better)")
    e.add_argument("-k", type=int, default=1500, help="examples per row")
    e.add_argument("--seed", type=int, default=0, help="seed for the random row")
    e.add_argument("--subsets-tag", default="extremes")

    r = sub.add_parser("run")
    r.add_argument("--variant", choices=["nll", "reward"], required=True)
    r.add_argument("--ref-step", type=int, default=10, help="reference checkpoint step (0=base)")
    r.add_argument("--epochs", type=int, default=3,
                   help="nll: full-coverage passes over each subset (the matched test)")
    r.add_argument("--train-steps", type=int, default=None,
                   help="override epochs with N sampled-minibatch steps (nll diagnostic) "
                        "or GRPO steps (reward; default 60)")
    r.add_argument("--lr", type=float, default=None, help="default: cfg.learning_rate")
    r.add_argument("--mb", type=int, default=16, help="nll minibatch size")
    r.add_argument("--eval-examples", type=int, default=256, help="reward: target prompts to grade")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tag", default="", help="suffix for measured_<variant>_<tag>.jsonl "
                                             "(keep training specs from colliding; "
                                             "defaults to --subsets-tag when that is set)")
    r.add_argument("--subsets-tag", default="", help="read subsets_<tag>.npy instead of subsets.npy")
    r.add_argument("--limit", type=int, default=None, help="process at most N subsets this run")

    s = sub.add_parser("score")
    s.add_argument("--variant", choices=["nll", "reward"], required=True)
    s.add_argument("--scores", nargs="+", required=True, help="per-example score .npy file(s)")
    s.add_argument("--tag", default="", help="measured_<variant>_<tag>.jsonl to score against "
                                             "(defaults to --subsets-tag when that is set)")
    s.add_argument("--subsets-tag", default="", help="read subsets_<tag>.npy instead of subsets.npy")

    args = p.parse_args(argv)
    cfg = ExperimentConfig.load(Path(args.output_root).expanduser().resolve()
                                / args.run_name / "config.json")

    # A tagged manifest without an explicit measured tag inherits it, so rows from
    # different manifests can never silently land in the same measured file.
    tag = getattr(args, "tag", "") or getattr(args, "subsets_tag", "")

    if args.cmd == "make-subsets":
        make_subsets(cfg, args.num_subsets, args.alpha, args.seed, args.subsets_tag)
    elif args.cmd == "make-extremes":
        make_extremes(cfg, args.scores, args.k, args.seed, args.subsets_tag)
    elif args.cmd == "score":
        score(cfg, args.variant, args.scores, tag, args.subsets_tag)
    elif args.cmd == "run":
        lr = args.lr if args.lr is not None else cfg.learning_rate
        if args.variant == "nll":
            run_nll(cfg, args.ref_step, args.train_steps, lr, args.mb, args.seed,
                    args.epochs, tag, args.limit, args.subsets_tag)
        else:
            run_reward(cfg, args.ref_step, args.train_steps or 60, args.eval_examples,
                       args.seed, tag, args.limit, args.subsets_tag)


if __name__ == "__main__":
    main()
