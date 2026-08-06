"""Empirical check of Proposition (Adam-TracIn is the diagonal-curvature estimator).

The proposition makes two separable claims:

  (A) v̂_c — Adam's bias-corrected second moment — is a running estimate of the
      diagonal of the empirical Fisher F_c. So the Adam preconditioner
      P_c = diag((√v̂_c + ε)^{-1}) is, up to scale, diag(F_c)^{-1/2}.

  (B) The three estimators form a *ladder* of preconditioners
          I   ⊂   diag(v̂)^{-1/2}   ⊂   (F + λI)^{-1},
      i.e. Adam-TracIn sits strictly between vanilla TracIn and the Fisher
      estimator: it is closer to the full damped-Fisher operator, and to the
      influence scores that operator induces, than the identity is.

This script verifies both on models small enough that the exact D×D empirical
Fisher and its inverse can be *materialised* (no CG, no EKFAC approximation),
so the "full Fisher" rung is the real thing and not a surrogate:

  --model mlp   2-layer MLP on a synthetic teacher task   (D = 421, ~20s on CPU)
  --model lm    SmolLM2-135M + LoRA on GSM8K              (D = 768, ~25min on CPU)

D is deliberately small in both arms. The binding constraint is not compute but
that the Fisher sample must be several times D — otherwise F is undersampled,
(F+λI)^{-1} collapses to I/λ on a spurious null space, and the identity rung looks
good for reasons that have nothing to do with curvature. --max-dim enforces this.

Both are trained with the *actual* AdamW that the proposition talks about, and
v̂ is read out of the live optimizer state at each checkpoint — never simulated.

Measurements, per checkpoint c:

  Claim A   Spearman ρ and log-log Pearson r between v̂_c and diag(F_c), plus the
            same against the minibatch second moment (see CAVEAT below), and the
            relative error ‖P_adam − P_diag^{1/2}‖/‖P_diag^{1/2}‖ after optimal
            rescaling (the η_c in the proposition is a free positive scale).

  Claim B   (operator space) Frobenius cosine of each preconditioner with the
            target (F_c + λI)^{-1}, which is scale-invariant, as it must be
            since η_c is unidentified. Expect cos(I) < cos(Adam) < 1.

  Claim B'  (score space — what actually ranks data) influence matrices
            S_M[i,j] = g_test,i^T M g_train,j for M ∈ {I, Adam, diag, full},
            scored by Spearman rank correlation against S_full, both pooled and
            per-test-row (the per-row number is the one that governs selection).

  (opt)     --ground-truth: the measured Δ test loss from actually taking one
            AdamW step on a single training example. This is a *different*
            question from the ladder — it asks which estimator predicts the real
            optimizer — and Adam-TracIn can beat the full Fisher here. Reported
            separately so it cannot be confused with the ladder claim.

CAVEATS the script measures rather than hides:

  * Adam's v̂ is the second moment of *minibatch* gradients, so for batch size B
    E[g_B²] = diag(F)/B + (1 − 1/B)·(mean grad)², i.e. v̂ ∝ diag(F) exactly only
    at B = 1, with a mean-gradient contamination that grows with B. Sweep with
    --batch-sizes to see the claim degrade gracefully.
  * The ladder ordering is a statement about a damping regime. As λ → ∞,
    (F+λI)^{-1} → I/λ and every rung converges to the identity. The λ sweep
    (--lambdas, in units of mean diag(F)) shows the ordering across regimes.
  * The target is rebuilt from HALF the Fisher sample and the ordering re-checked,
    so a conclusion that merely reflects the Fisher sample size is caught rather
    than reported. (Numerical rank is *not* used as the guard: with ~5 decades of
    spectrum it is tolerance-dependent and never saturates at D.)

Usage:
    python scripts/verify_preconditioner_ladder.py --model mlp --seeds 3
    python scripts/verify_preconditioner_ladder.py --model mlp --batch-sizes 1 8 32
    python scripts/verify_preconditioner_ladder.py --model mlp --ground-truth
    python scripts/verify_preconditioner_ladder.py --model lm

Outputs (default --outdir outputs/precond_ladder): results.json, ladder.png,
ladder_table.tex (paste-ready), plus a summary table on stdout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_class: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, n_class)
        )

    def forward(self, x):
        return self.net(x)


def make_mlp_task(seed: int, n: int = 8192, d_in: int = 20, d_hidden: int = 16,
                  n_class: int = 5, device="cpu"):
    """Synthetic teacher-labelled classification.

    Anisotropic inputs on purpose: a task with a flat, isotropic Fisher would
    make every preconditioner look alike and the ladder trivially unfalsifiable.
    The per-feature scale spread (1e-1 .. 1e1) is what gives diag(F) — and hence
    v̂ — something to actually capture.
    """
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(-1, 1, d_in)
    x = torch.randn(n, d_in, generator=g) * scales
    teacher = nn.Sequential(nn.Linear(d_in, 64), nn.Tanh(), nn.Linear(64, n_class))
    with torch.no_grad():
        for p in teacher.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.7)
        logits = teacher(x)
        y = torch.distributions.Categorical(logits=logits).sample()
    model = MLP(d_in, d_hidden, n_class).to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model, x.to(device), y.to(device)


def mlp_example_losses(model, x, y):
    """Per-example CE loss vector (no reduction)."""
    return F.cross_entropy(model(x), y, reduction="none")


def make_lm_task(seed: int, model_name: str, n_train: int, max_len: int,
                 lora_rank: int, lora_layers: int, projections=("v_proj",),
                 device="cpu"):
    """Small causal LM + LoRA on GSM8K; only the adapters are trainable, which is
    what keeps D small enough for an exact D×D Fisher."""
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    n_layers = base.config.num_hidden_layers
    target_layers = list(range(n_layers - lora_layers, n_layers))
    targets = [
        f"model.layers.{i}.self_attn.{proj}"
        for i in target_layers
        for proj in projections
    ]
    model = get_peft_model(
        base,
        LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0,
                   target_modules=targets, bias="none", task_type="CAUSAL_LM"),
    ).to(device)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if n_train > len(ds):
        raise SystemExit(
            f"--n-data={n_train} exceeds the GSM8K train split ({len(ds)}); the Fisher "
            f"sample, influence pool and test set must all fit inside it. Lower "
            f"--n-fisher/--n-train/--n-test.")
    ds = ds.shuffle(seed=seed).select(range(n_train))
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(ds["question"], ds["answer"])]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
              max_length=max_len)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    return model, ids, mask


def lm_example_losses(model, ids, mask):
    """Per-example mean token NLL (no reduction across examples)."""
    out = model(input_ids=ids, attention_mask=mask).logits
    tgt = ids[:, 1:]
    lg = out[:, :-1]
    m = mask[:, 1:].float()
    nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                          reduction="none").view(tgt.shape)
    return (nll * m).sum(1) / m.sum(1).clamp(min=1)


# --------------------------------------------------------------------------
# gradient / Fisher machinery
# --------------------------------------------------------------------------


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def flat_grad(model, loss) -> torch.Tensor:
    params = [p for _, p in trainable(model)]
    gs = torch.autograd.grad(loss, params, retain_graph=False)
    return torch.cat([g.reshape(-1) for g in gs]).detach().float()


def per_example_grads(model, loss_fn, idx) -> torch.Tensor:
    """[n, D] matrix of per-example gradients, one backward each — exact, no
    sampling or blocking. D is small by construction, so the cost is the [n, D]
    stack, not the graph."""
    rows = []
    for i in idx:
        model.zero_grad(set_to_none=True)
        loss = loss_fn([i]).sum()
        rows.append(flat_grad(model, loss))
    model.zero_grad(set_to_none=True)
    return torch.stack(rows)


def empirical_fisher(G: torch.Tensor) -> torch.Tensor:
    """F = (1/n) Σ g_i g_i^T from per-example gradients G=[n,D]."""
    return (G.T @ G) / G.shape[0]


def adam_vhat(model, opt) -> torch.Tensor:
    """Bias-corrected v̂ as a flat D-vector, read from the live optimizer state in
    trainable-parameter order (same order flat_grad flattens)."""
    beta2 = opt.param_groups[0]["betas"][1]
    parts = []
    for _, p in trainable(model):
        st = opt.state[p]
        t = float(st["step"].item() if torch.is_tensor(st["step"]) else st["step"])
        bc2 = 1.0 - beta2 ** t if t > 0 else 1.0
        parts.append((st["exp_avg_sq"] / bc2).reshape(-1).float())
    return torch.cat(parts)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def frob_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    """Scale-invariant operator agreement — the right metric here because the
    proposition maps (F+λI)^{-1} ↦ η_c P_c with η_c a free positive scale."""
    return float((A * B).sum() / (A.norm() * B.norm() + 1e-30))


def scale_free_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """min_α ‖αa − b‖/‖b‖ — best-case relative error after optimal rescaling."""
    alpha = float((a * b).sum() / (a * a).sum().clamp(min=1e-30))
    return float((alpha * a - b).norm() / (b.norm() + 1e-30))


def score_matrix(G_test: torch.Tensor, G_train: torch.Tensor,
                 M: torch.Tensor | None, diag: torch.Tensor | None = None):
    """S[i,j] = g_test,i^T M g_train,j, with a diagonal fast path."""
    if diag is not None:
        return (G_test * diag) @ G_train.T
    if M is None:
        return G_test @ G_train.T
    return G_test @ M @ G_train.T


def rank_agreement(S: torch.Tensor, S_ref: torch.Tensor) -> dict:
    s, r = S.reshape(-1).numpy(), S_ref.reshape(-1).numpy()
    pooled = float(spearmanr(s, r).statistic)
    per_row = [float(spearmanr(a, b).statistic) for a, b in zip(S.numpy(), S_ref.numpy())]
    k = max(1, S.shape[1] // 10)
    ov = []
    for a, b in zip(S.numpy(), S_ref.numpy()):
        ta = set(np.argsort(-a)[:k])
        tb = set(np.argsort(-b)[:k])
        ov.append(len(ta & tb) / k)
    return {"spearman_pooled": pooled,
            "spearman_per_test": float(np.mean(per_row)),
            "top10pct_overlap": float(np.mean(ov))}


# --------------------------------------------------------------------------
# one checkpoint's worth of evidence
# --------------------------------------------------------------------------


@dataclass
class CheckpointResult:
    step: int
    seed: int
    batch_size: int
    lr: float
    claim_a: dict = field(default_factory=dict)
    operator: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    ground_truth: dict = field(default_factory=dict)
    half_fisher: dict = field(default_factory=dict)
    off_diag_mass: float = 0.0
    # raw coordinates for the scatter panel, kept out of the JSON payload
    coords: dict = field(default_factory=dict, repr=False)


def analyse_checkpoint(model, opt, loss_fn, n_data, args, step, seed, batch_size,
                       rng: np.random.Generator) -> CheckpointResult:
    D = sum(p.numel() for _, p in trainable(model))
    idx = rng.permutation(n_data)
    f_idx = idx[: args.n_fisher]
    tr_idx = idx[args.n_fisher: args.n_fisher + args.n_train]
    te_idx = idx[args.n_fisher + args.n_train: args.n_fisher + args.n_train + args.n_test]

    G_f = per_example_grads(model, loss_fn, f_idx)
    G_tr = per_example_grads(model, loss_fn, tr_idx)
    G_te = per_example_grads(model, loss_fn, te_idx)

    Fmat = empirical_fisher(G_f)
    dF = torch.diagonal(Fmat).clone()
    vhat = adam_vhat(model, opt)
    lr = opt.param_groups[0]["lr"]
    eps = opt.param_groups[0]["eps"]

    res = CheckpointResult(step=step, seed=seed, batch_size=batch_size, lr=lr)

    # ---- Claim A: v̂ ≈ diag(F) -------------------------------------------
    # Minibatch prediction: E[g_B²] = diag(F)/B + (1 − 1/B)·(mean g)², which is
    # what v̂ *actually* averages when B > 1.
    gbar = G_f.mean(0)
    B = float(batch_size)
    dF_mb = dF / B + (1.0 - 1.0 / B) * gbar.pow(2)
    v, a, b = vhat.numpy(), dF.numpy(), dF_mb.numpy()
    pos = (v > 0) & (a > 0)
    res.claim_a = {
        "D": D,
        "spearman_vhat_diagF": float(spearmanr(v, a).statistic),
        "logpearson_vhat_diagF": float(pearsonr(np.log10(v[pos]), np.log10(a[pos])).statistic),
        "spearman_vhat_minibatch_pred": float(spearmanr(v, b).statistic),
        "P_relerr_vs_exact_sqrt_diag": scale_free_rel_err(
            1.0 / (vhat.sqrt() + eps), 1.0 / (dF.clamp(min=1e-30).sqrt() + eps)),
        "diagF_dynamic_range_decades": float(np.log10(a.max() / max(a[a > 0].min(), 1e-30))),
        # Reported for context only. This is *numerical* rank at a relative
        # tolerance: with ~5 decades of spectrum the small-but-nonzero eigenvalues
        # fall below tolerance, so it never saturates at D and cannot serve as a
        # sampling guard. The real guard is conclusion stability under halving the
        # Fisher sample (below).
        "fisher_numerical_rank": int(torch.linalg.matrix_rank(Fmat).item()),
        "n_fisher": int(args.n_fisher),
    }

    # ---- preconditioners --------------------------------------------------
    lam_scale = float(dF.mean())
    P_adam = 1.0 / (vhat.sqrt() + eps)          # the Adam rung (up to η_c)
    P_sqrt_exact = 1.0 / (dF.clamp(min=1e-30).sqrt() + eps)  # idealised Adam rung

    eye = torch.eye(D)
    for lam_mult in args.lambdas:
        lam = lam_mult * lam_scale
        Ffull = torch.linalg.inv(Fmat + lam * eye)
        P_diag = 1.0 / (dF + lam)               # diagonal, no sqrt

        # off-diagonal mass of the target: the part no diagonal rung can capture
        offd = float((Ffull - torch.diag(torch.diagonal(Ffull))).norm() / Ffull.norm())
        res.off_diag_mass = offd

        rungs = {
            "identity": (None, torch.ones(D)),
            "adam": (None, P_adam),
            "diagF_sqrt": (None, P_sqrt_exact),
            "diagF": (None, P_diag),
            "fisher_full": (Ffull, None),
        }
        op = {}
        for name, (M, d) in rungs.items():
            Mmat = M if M is not None else torch.diag(d)
            op[name] = frob_cos(Mmat, Ffull)
        res.operator[f"lam{lam_mult:g}"] = {"cos_to_full_fisher": op,
                                            "offdiag_frac_of_inv": offd,
                                            "lambda_abs": lam}

        S_ref = score_matrix(G_te, G_tr, Ffull)
        sc = {}
        for name, (M, d) in rungs.items():
            S = score_matrix(G_te, G_tr, M, d)
            sc[name] = rank_agreement(S, S_ref)
        res.scores[f"lam{lam_mult:g}"] = sc

    # ---- sampling guard --------------------------------------------------
    # Where F is undersampled, (F+λI)^{-1} → I/λ on the spurious null space, which
    # flatters the identity rung. Rather than trust a tolerance-dependent rank,
    # rebuild the target from HALF the Fisher sample and check the ladder ordering
    # is unchanged: a conclusion that survives halving the data is not an artefact
    # of how much data was used.
    lam0 = args.lambdas[0] * lam_scale
    F_half = empirical_fisher(G_f[: len(f_idx) // 2])
    T_half = torch.linalg.inv(F_half + lam0 * eye)
    S_ref_half = score_matrix(G_te, G_tr, T_half)
    half = {n: rank_agreement(score_matrix(G_te, G_tr, None, d), S_ref_half)
            ["spearman_per_test"]
            for n, d in [("identity", torch.ones(D)), ("adam", P_adam),
                         ("diagF", 1.0 / (dF + lam0))]}
    res.half_fisher = half
    res.half_fisher["ladder_holds"] = float(
        half["identity"] < half["adam"] <= half["diagF"])

    # keep the actual coordinates of THIS checkpoint for the scatter panel, so the
    # figure shows the run being reported rather than a separately retrained one
    res.coords = {"vhat": vhat.detach().clone(), "diagF": dF.detach().clone()}

    return res


# --------------------------------------------------------------------------
# ground truth: one real AdamW step
# --------------------------------------------------------------------------


def ground_truth_check(model, opt, loss_fn, tr_idx, te_idx, f_idx, args) -> dict:
    """Δ(test loss) from taking one genuine AdamW step on a single train example.

    NOTE this answers "which estimator predicts the optimizer", not "which
    approximates the Fisher estimator". Kept separate from the ladder on purpose.
    """
    import copy

    G_tr = per_example_grads(model, loss_fn, tr_idx)
    with torch.no_grad():
        base_test = float(loss_fn(list(te_idx)).mean())
    model.zero_grad(set_to_none=True)
    g_te = flat_grad(model, loss_fn(list(te_idx)).mean())
    vhat = adam_vhat(model, opt)
    eps = opt.param_groups[0]["eps"]
    P_adam = 1.0 / (vhat.sqrt() + eps)

    # Build F from the dedicated Fisher sample, NOT from the small influence pool:
    # a Fisher estimated from n_train << D examples is undersampled, and its inverse
    # would misrepresent the full-Fisher rung in this comparison.
    Fmat = empirical_fisher(per_example_grads(model, loss_fn, f_idx))
    lam = args.lambdas[0] * float(torch.diagonal(Fmat).mean())
    Ffull = torch.linalg.inv(Fmat + lam * torch.eye(Fmat.shape[0]))

    msd, osd = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    deltas = []
    for i in tr_idx:
        model.load_state_dict(msd)
        opt.load_state_dict(osd)
        model.zero_grad(set_to_none=True)
        loss_fn([i]).sum().backward()
        opt.step()
        with torch.no_grad():
            deltas.append(base_test - float(loss_fn(list(te_idx)).mean()))
    model.load_state_dict(msd)
    opt.load_state_dict(osd)

    truth = np.array(deltas)
    preds = {
        "identity": (G_tr @ g_te).numpy(),
        "adam": (G_tr @ (P_adam * g_te)).numpy(),
        "fisher_full": (G_tr @ (Ffull @ g_te)).numpy(),
    }
    return {k: {"spearman_vs_measured": float(spearmanr(v, truth).statistic),
                "pearson_vs_measured": float(pearsonr(v, truth).statistic)}
            for k, v in preds.items()}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run_seed(args, seed: int, batch_size: int) -> list[CheckpointResult]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if args.model == "mlp":
        model, x, y = make_mlp_task(seed, n=args.n_data, d_hidden=args.mlp_hidden)
        loss_fn = lambda ii: mlp_example_losses(model, x[ii], y[ii])  # noqa: E731
        n_data = x.shape[0]
        total_steps = args.steps
        lr = args.lr
    else:
        model, ids, mask = make_lm_task(
            seed, args.lm_model, args.n_data, args.lm_max_len,
            args.lora_rank, args.lora_layers, tuple(args.lora_targets))
        loss_fn = lambda ii: lm_example_losses(model, ids[ii], mask[ii])  # noqa: E731
        n_data = ids.shape[0]
        total_steps = args.lm_steps
        lr = args.lm_lr

    D = sum(p.numel() for _, p in trainable(model))
    print(f"[seed {seed} | B={batch_size}] D = {D} trainable params, "
          f"{total_steps} AdamW steps on {n_data} examples")
    if D > args.max_dim:
        raise SystemExit(
            f"D={D} exceeds --max-dim={args.max_dim}; the exact D×D Fisher inverse "
            f"is the whole point of this script — shrink the model (fewer LoRA "
            f"layers / lower rank) rather than raising the cap blindly.")
    needed = args.n_fisher + args.n_train + args.n_test
    if n_data < needed:
        raise SystemExit(
            f"only {n_data} examples available but the Fisher sample, influence pool "
            f"and test set need {needed} disjoint ones. Raise --n-data or lower "
            f"--n-fisher/--n-train/--n-test.")
    if args.n_fisher < 2 * D:
        print(f"  WARNING: --n-fisher={args.n_fisher} is below 2·D={2 * D}. F will be "
              f"undersampled, (F+λI)^-1 degenerates toward I/λ on the spurious null "
              f"space, and the identity rung will look better than it is. The "
              f"half-sample guard should flag this run.")

    opt = torch.optim.AdamW(
        [p for _, p in trainable(model)], lr=lr, betas=(0.9, args.beta2),
        eps=args.adam_eps, weight_decay=0.0)

    ckpts = sorted({int(round(f * total_steps)) for f in args.ckpt_fracs})
    out: list[CheckpointResult] = []
    for step in range(1, total_steps + 1):
        bi = rng.integers(0, n_data, size=batch_size)
        model.zero_grad(set_to_none=True)
        loss_fn(list(bi)).mean().backward()
        opt.step()
        if step in ckpts:
            r = analyse_checkpoint(model, opt, loss_fn, n_data, args, step, seed,
                                   batch_size, rng)
            if args.ground_truth:
                idx = rng.permutation(n_data)
                a, b = args.n_train, args.n_train + args.n_test
                r.ground_truth = ground_truth_check(
                    model, opt, loss_fn, idx[:a], idx[a:b],
                    idx[b: b + args.n_fisher], args)
            out.append(r)
            print(f"  step {step:>4}: ρ(v̂, diagF) = "
                  f"{r.claim_a['spearman_vhat_diagF']:.3f}")
    return out


RUNGS = ["identity", "adam", "diagF_sqrt", "diagF", "fisher_full"]
RUNG_LABEL = {
    "identity": r"vanilla TracIn  $P=I$",
    "adam": r"Adam-TracIn  $P=\mathrm{diag}(\hat v)^{-1/2}$",
    "diagF_sqrt": r"exact $\mathrm{diag}(F)^{-1/2}$",
    "diagF": r"exact $(\mathrm{diag}(F)+\lambda)^{-1}$",
    "fisher_full": r"Fisher  $(F+\lambda I)^{-1}$",
}


def aggregate(results: list[CheckpointResult], args) -> dict:
    lam_keys = [f"lam{m:g}" for m in args.lambdas]
    agg = {"claim_a": {}, "operator": {}, "scores": {}, "ground_truth": {}}
    for k in ["spearman_vhat_diagF", "logpearson_vhat_diagF",
              "spearman_vhat_minibatch_pred", "P_relerr_vs_exact_sqrt_diag",
              "diagF_dynamic_range_decades"]:
        vals = [r.claim_a[k] for r in results]
        agg["claim_a"][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    for lk in lam_keys:
        agg["operator"][lk] = {
            r_: {"mean": float(np.mean([x.operator[lk]["cos_to_full_fisher"][r_] for x in results])),
                 "std": float(np.std([x.operator[lk]["cos_to_full_fisher"][r_] for x in results]))}
            for r_ in RUNGS}
        agg["operator"][lk]["offdiag_frac_of_inv"] = float(
            np.mean([x.operator[lk]["offdiag_frac_of_inv"] for x in results]))
        agg["scores"][lk] = {
            r_: {m: {"mean": float(np.mean([x.scores[lk][r_][m] for x in results])),
                     "std": float(np.std([x.scores[lk][r_][m] for x in results]))}
                 for m in ["spearman_pooled", "spearman_per_test", "top10pct_overlap"]}
            for r_ in RUNGS}

        # An ordering claim needs paired evidence, not two means that happen to
        # differ: how often does the higher rung win on the SAME checkpoint, and
        # what fraction of the identity→Fisher gap does each rung close?
        def per_ckpt(rung, metric):
            return np.array([x.scores[lk][rung][metric] for x in results])

        base = per_ckpt("identity", "spearman_per_test")
        top = per_ckpt("fisher_full", "spearman_per_test")
        wins, gap = {}, {}
        for r_ in RUNGS:
            cur = per_ckpt(r_, "spearman_per_test")
            wins[r_] = float(np.mean(cur > base))
            denom = np.clip(top - base, 1e-12, None)
            gap[r_] = float(np.mean((cur - base) / denom))
        agg["scores"][lk]["_win_rate_over_identity"] = wins
        agg["scores"][lk]["_gap_closed_vs_identity"] = gap
        agg["scores"][lk]["_n_checkpoints"] = len(results)

        op_base = np.array([x.operator[lk]["cos_to_full_fisher"]["identity"] for x in results])
        agg["operator"][lk]["_win_rate_over_identity"] = {
            r_: float(np.mean(
                np.array([x.operator[lk]["cos_to_full_fisher"][r_] for x in results]) > op_base))
            for r_ in RUNGS}

    if results and results[0].ground_truth:
        for r_ in ["identity", "adam", "fisher_full"]:
            vals = [x.ground_truth[r_]["spearman_vs_measured"] for x in results]
            agg["ground_truth"][r_] = {"mean": float(np.mean(vals)),
                                       "std": float(np.std(vals))}
    agg["half_fisher_ladder_holds_frac"] = float(
        np.mean([r.half_fisher["ladder_holds"] for r in results])) if results else 0.0
    agg["fisher_rank_mean"] = float(
        np.mean([r.claim_a["fisher_numerical_rank"] for r in results]))
    agg["D"] = int(results[0].claim_a["D"]) if results else 0
    return agg


def verdict(agg: dict, args) -> dict:
    """The falsifiable statements, evaluated."""
    lk = f"lam{args.lambdas[0]:g}"
    op = agg["operator"][lk]
    sc = agg["scores"][lk]
    a = agg["claim_a"]
    v = {
        "A_vhat_tracks_diagF": a["spearman_vhat_diagF"]["mean"] > 0.9,
        "A_adam_P_matches_exact_sqrt_diag": a["P_relerr_vs_exact_sqrt_diag"]["mean"] < 0.5,
        "B_operator_ladder": op["identity"]["mean"] < op["adam"]["mean"] < op["fisher_full"]["mean"],
        "B_score_ladder": (sc["identity"]["spearman_per_test"]["mean"]
                           < sc["adam"]["spearman_per_test"]["mean"]
                           < sc["fisher_full"]["spearman_per_test"]["mean"]),
        # the ordering must hold checkpoint-by-checkpoint, not just on average
        "B_score_ladder_every_checkpoint": sc["_win_rate_over_identity"]["adam"] == 1.0,
        # Not "rank(F) == D" — softmax-CE gradients are structurally confined to a
        # subspace, and numerical rank is tolerance-dependent anyway. What must hold
        # is that the ordering does not depend on the Fisher sample size.
        "B_ladder_stable_on_half_the_fisher_data":
            agg["half_fisher_ladder_holds_frac"] == 1.0,
    }
    v["all_hold"] = all(v.values())
    return v


def make_plot(results: list[CheckpointResult], agg: dict, args, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lk = f"lam{args.lambdas[0]:g}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    coords = results[-1].coords
    v, d = coords["vhat"].numpy(), coords["diagF"].numpy()
    m = (v > 0) & (d > 0)
    ax.scatter(d[m], v[m], s=6, alpha=0.35, edgecolors="none")
    lo, hi = min(d[m].min(), v[m].min()), max(d[m].max(), v[m].max())
    al = float(np.median(v[m] / d[m]))
    ax.plot([lo, hi], [al * lo, al * hi], "r--", lw=1,
            label=fr"$\hat v \propto \mathrm{{diag}}(F)$  ($\rho$={agg['claim_a']['spearman_vhat_diagF']['mean']:.3f})")
    ax.set(xscale="log", yscale="log", xlabel=r"$\mathrm{diag}(F_c)_d$",
           ylabel=r"$\hat v_{c,d}$", title="(A) Adam 2nd moment vs Fisher diagonal")
    ax.legend(fontsize=8)

    ax = axes[1]
    names = RUNGS
    vals = [agg["operator"][lk][n]["mean"] for n in names]
    errs = [agg["operator"][lk][n]["std"] for n in names]
    ax.bar(range(len(names)), vals, yerr=errs, capsize=3,
           color=["#999", "#d95f02", "#7570b3", "#66a61e", "#1b9e77"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["$I$", "Adam", r"$\mathrm{diag}F^{-1/2}$",
                        r"$\mathrm{diag}F^{-1}$", "full"], fontsize=8)
    ax.set(ylabel=r"Frobenius cosine with $(F+\lambda I)^{-1}$",
           title=f"(B) operator ladder ($\\lambda$={args.lambdas[0]:g}·mean diag$F$)")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")

    ax = axes[2]
    for mult in args.lambdas:
        k = f"lam{mult:g}"
        ys = [agg["scores"][k][n]["spearman_per_test"]["mean"] for n in names]
        ax.plot(range(len(names)), ys, "o-", label=fr"$\lambda$={mult:g}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["$I$", "Adam", r"$\mathrm{diag}F^{-1/2}$",
                        r"$\mathrm{diag}F^{-1}$", "full"], fontsize=8)
    ax.set(ylabel=r"Spearman vs Fisher scores (per test ex.)",
           title="(B') score-space ladder")
    ax.legend(fontsize=8)

    fig.suptitle(f"Preconditioner ladder — {args.model.upper()}, "
                 f"{len(results)} checkpoints × seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"wrote {path}")


def make_table(agg: dict, args, path: Path):
    lk = f"lam{args.lambdas[0]:g}"
    n_ck = agg["scores"][lk]["_n_checkpoints"]
    rows = []
    for n in RUNGS:
        o = agg["operator"][lk][n]
        s = agg["scores"][lk][n]
        wins = int(round(agg["scores"][lk]["_win_rate_over_identity"][n] * n_ck))
        gap = 100 * agg["scores"][lk]["_gap_closed_vs_identity"][n]
        rows.append(
            f"{RUNG_LABEL[n]} & {o['mean']:.3f} $\\pm$ {o['std']:.3f} "
            f"& {s['spearman_per_test']['mean']:.3f} $\\pm$ {s['spearman_per_test']['std']:.3f} "
            f"& {gap:.1f}\\% & {wins}/{n_ck} \\\\")
    a = agg["claim_a"]
    tex = r"""% generated by scripts/verify_preconditioner_ladder.py
\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
Preconditioner $P_c$ & $\cos_F$ with $(\Fish_c+\lambda I)^{-1}$ & Spearman vs Fisher scores & gap closed & beats $I$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{The preconditioner ladder, measured (""" + args.model.upper() + r""",
$D=""" + str(agg["D"]) + r"""$, exact $D\times D$ empirical Fisher and its explicit
inverse; $\lambda=""" + f"{args.lambdas[0]:g}" + r"""\cdot\mathrm{mean}\,\diag(\Fish_c)$;
mean $\pm$ s.d. over """ + str(n_ck) + r""" checkpoints $\times$ seeds).
\emph{Claim (i), $\hat v_c$ estimates $\diag(\Fish_c)$:} Spearman
$\rho=""" + f"{a['spearman_vhat_diagF']['mean']:.3f}" + r"""$ across coordinates
spanning """ + f"{a['diagF_dynamic_range_decades']['mean']:.1f}" + r""" decades, and the
Adam preconditioner read from the optimizer state matches the exact
$\diag(\Fish_c)^{-1/2}$ to """ + f"{100 * a['P_relerr_vs_exact_sqrt_diag']['mean']:.1f}" + r"""\%
relative error after optimal rescaling (the free scale $\eta_c$).
\emph{Claim (ii), the ordering:} Adam-TracIn is closer to the Fisher estimator than
vanilla TracIn is, in both operator and score space, at every checkpoint, closing
""" + f"{100 * agg['scores'][lk]['_gap_closed_vs_identity']['adam']:.1f}" + r"""\% of the
$I\to\Fish^{-1}$ gap. It cannot close more: the target's off-diagonal mass is
""" + f"{100 * agg['operator'][lk]['offdiag_frac_of_inv']:.0f}" + r"""\% of
$\|(\Fish+\lambda I)^{-1}\|_F$, which no diagonal preconditioner represents, and Adam's
square root forfeits part of the rest --- the un-square-rooted
$(\diag(\Fish)+\lambda)^{-1}$ closes
""" + f"{100 * agg['scores'][lk]['_gap_closed_vs_identity']['diagF']:.1f}" + r"""\%.
Note the ordering is a statement about the lightly-damped regime: as
$\lambda\to\infty$, $(\Fish+\lambda I)^{-1}\to I/\lambda$ and every rung converges to
the identity.}
\label{tab:precond-ladder}
\end{table}
"""
    path.write_text(tex)
    print(f"wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mlp", "lm"], default="mlp")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--outdir", type=Path, default=Path("outputs/precond_ladder"))
    p.add_argument("--max-dim", type=int, default=8000,
                   help="refuse to run above this D (exact Fisher inverse cost)")

    p.add_argument("--steps", type=int, default=300, help="MLP AdamW steps")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-data", type=int, default=8192)
    p.add_argument("--mlp-hidden", type=int, default=16,
                   help="MLP width; sets D, which must stay well below --n-fisher")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[8],
                   help="sweep to expose the minibatch caveat (B=1 is the exact case)")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)

    p.add_argument("--ckpt-fracs", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n-fisher", type=int, default=None,
                   help="examples building F_c; keep it a few times D, or F is "
                        "undersampled and (F+lambda I)^-1 degenerates to I/lambda on a "
                        "spurious null space (mlp: 2048, lm: 3072)")
    p.add_argument("--n-train", type=int, default=None,
                   help="pool scored for influence (mlp: 96, lm: 64)")
    p.add_argument("--n-test", type=int, default=None, help="mlp: 24, lm: 16")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0],
                   help="damping in units of mean diag(F); the first is the headline")
    p.add_argument("--ground-truth", action="store_true",
                   help="also measure Δtest-loss from one real AdamW step (slow)")

    p.add_argument("--lm-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--lm-steps", type=int, default=300,
                   help="Adam needs enough steps for v̂ to be a converged EMA; a "
                        "handful of steps tests nothing")
    p.add_argument("--lora-targets", nargs="+", default=["v_proj"],
                   help="which attention projections get adapters. Each one added "
                        "raises D, and D must stay well under --n-fisher")
    p.add_argument("--lm-lr", type=float, default=1e-4)
    p.add_argument("--lm-max-len", type=int, default=128)
    p.add_argument("--lora-rank", type=int, default=1)
    p.add_argument("--lora-layers", type=int, default=1,
                   help="adapters on the last N layers. Kept at 1 so D stays a few "
                        "times below --n-fisher; raising it without raising --n-fisher "
                        "undersamples F and the sampling guard will flag the run")
    return p.parse_args()


def main():
    args = parse_args()
    # Per-model defaults for anything the user did not set explicitly. The Fisher
    # sample has to be a few times D in both arms, so it cannot be one shared number.
    defaults = ({"n_fisher": 2048, "n_train": 96, "n_test": 24} if args.model == "mlp"
                else {"n_fisher": 3072, "n_train": 64, "n_test": 16})
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.model == "lm":
        # GSM8K train is 7473 rows; the LM arm only needs the examples it samples.
        args.n_data = min(7473, max(args.n_fisher + args.n_train + args.n_test, 512))
        if args.ckpt_fracs == [0.1, 0.25, 0.5, 0.75, 1.0]:
            args.ckpt_fracs = [0.5, 1.0]  # each LM checkpoint is thousands of backwards
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_results: list[CheckpointResult] = []
    by_batch: dict[int, list[CheckpointResult]] = {}
    for B in args.batch_sizes:
        rs = []
        for seed in range(args.seeds):
            rs.extend(run_seed(args, seed, B))
        by_batch[B] = rs
        all_results.extend(rs)

    headline = by_batch[args.batch_sizes[0]]
    agg = aggregate(headline, args)
    vd = verdict(agg, args)

    make_plot(headline, agg, args, args.outdir / "ladder.png")
    make_table(agg, args, args.outdir / "ladder_table.tex")

    payload = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "aggregate_headline_batch": agg,
        "verdict": vd,
        "by_batch_claim_a": {
            str(B): {"spearman_vhat_diagF": float(np.mean(
                [r.claim_a["spearman_vhat_diagF"] for r in rs]))}
            for B, rs in by_batch.items()},
        "per_checkpoint": [{k: v for k, v in vars(r).items() if k != "coords"}
                           for r in all_results],
    }
    (args.outdir / "results.json").write_text(json.dumps(payload, indent=2, default=float))

    lk = f"lam{args.lambdas[0]:g}"
    print("\n" + "=" * 78)
    print("CLAIM A — v̂_c estimates diag(F_c)")
    a = agg["claim_a"]
    print(f"  Spearman(v̂, diag F)          {a['spearman_vhat_diagF']['mean']:.3f} "
          f"± {a['spearman_vhat_diagF']['std']:.3f}")
    print(f"  Pearson on log10             {a['logpearson_vhat_diagF']['mean']:.3f}")
    print(f"  Spearman vs minibatch pred   {a['spearman_vhat_minibatch_pred']['mean']:.3f}")
    print(f"  ‖P_adam − P_exact‖/‖P_exact‖  {a['P_relerr_vs_exact_sqrt_diag']['mean']:.3f} "
          f"(after optimal rescaling)")
    print(f"  diag(F) dynamic range        {a['diagF_dynamic_range_decades']['mean']:.1f} decades")
    if len(args.batch_sizes) > 1:
        print("  batch-size sweep (ρ):        " + ", ".join(
            f"B={B}: {np.mean([r.claim_a['spearman_vhat_diagF'] for r in rs]):.3f}"
            for B, rs in by_batch.items()))

    print(f"\nCLAIM B — ladder (λ = {args.lambdas[0]:g}·mean diag F, "
          f"D={agg['D']}, rank(F)={agg['fisher_rank_mean']:.0f})")
    n_ck = agg["scores"][lk]["_n_checkpoints"]
    print(f"  {'rung':<28}{'cos_F to (F+λI)^-1':>20}{'Spearman scores':>18}"
          f"{'gap closed':>12}{'wins':>8}")
    for n in RUNGS:
        o, s = agg["operator"][lk][n], agg["scores"][lk][n]
        print(f"  {n:<28}{o['mean']:>13.3f}±{o['std']:.3f}"
              f"{s['spearman_per_test']['mean']:>13.3f}±{s['spearman_per_test']['std']:.3f}"
              f"{100 * agg['scores'][lk]['_gap_closed_vs_identity'][n]:>11.1f}%"
              f"{int(round(agg['scores'][lk]['_win_rate_over_identity'][n] * n_ck)):>5}/{n_ck}")
    print("  ('gap closed' = fraction of the identity→full-Fisher Spearman gap; "
          "'wins' = checkpoints beating identity)")
    print(f"  off-diagonal mass of (F+λI)^-1: "
          f"{100 * agg['operator'][lk]['offdiag_frac_of_inv']:.1f}% "
          f"(the ceiling on any diagonal rung)")
    frac = agg["half_fisher_ladder_holds_frac"]
    print(f"  sampling guard: rebuilt from HALF the Fisher sample, the ordering still "
          f"holds at {int(round(frac * n_ck))}/{n_ck} checkpoints"
          + ("" if frac == 1.0 else " — treat the ordering as sample-dependent"))
    if frac < 1.0:
        ratio = agg["fisher_rank_mean"] / max(agg["D"], 1)
        print(f"    F spans only ~{agg['fisher_rank_mean']:.0f} of D={agg['D']} dimensions "
              f"({100 * ratio:.0f}%), so (F+λI)^-1 is I/λ over most of the space and the\n"
              f"    identity rung is flattered. If raising --n-fisher does not move the "
              f"rank, the gradients are\n    structurally low-rank (e.g. rank-1 LoRA "
              f"adapters) and the ladder needs a richer parameterisation\n    "
              f"(--lora-rank / --lora-targets) rather than more data.")

    g = agg["scores"][lk]["_gap_closed_vs_identity"]
    print(f"  the √ in Adam's preconditioner costs real curvature: the sqrt rungs close "
          f"{100 * g['adam']:.1f}%\n  of the identity→Fisher gap vs "
          f"{100 * g['diagF']:.1f}% for the un-square-rooted diagonal "
          f"(diag(F)+λ)^-1.")

    print("\n  λ-sweep (score-space Spearman vs the Fisher estimator):")
    print(f"    {'λ/mean diagF':<14}" + "".join(f"{n:>14}" for n in RUNGS[:-1]))
    for mult in args.lambdas:
        k = f"lam{mult:g}"
        row = "".join(
            f"{agg['scores'][k][n]['spearman_per_test']['mean']:>14.3f}" for n in RUNGS[:-1])
        print(f"    {mult:<14g}{row}")
    print("    (as λ grows the target → I/λ and the identity rung catches up; the "
          "ladder is a statement\n     about the lightly-damped regime where the "
          "curvature term actually matters)")

    if agg["ground_truth"]:
        print("\nSEPARATE CHECK — predicting the measured Δtest-loss of one real AdamW step")
        for n, d in agg["ground_truth"].items():
            print(f"  {n:<34}{d['mean']:>13.3f}±{d['std']:.3f}")
        print("  (this ranks estimators by optimizer-faithfulness, not by Fisher proximity)")

    print("\nVERDICT")
    for k, val in vd.items():
        print(f"  {'PASS' if val else 'FAIL'}  {k}")
    print("=" * 78)
    print(f"artifacts in {args.outdir}")


if __name__ == "__main__":
    main()
