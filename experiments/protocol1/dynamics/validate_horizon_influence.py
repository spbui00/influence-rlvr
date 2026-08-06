"""Numerical validation of the finite-horizon (dynamics-based) influence derivation.

Checks the closed forms against BOTH exact optimizer unrolling and real "retraining"
on controlled quadratic problems where the derivation's approximations are EXACT:

  shared per-prompt Hessian A  ->  frozen linearization (A5) and affine-path (A7) hold
  linear target f              ->  no A5 error in g_test  (isolates the resolvent, Steps 4-8)
  quadratic target f           ->  tests the eps=1 interaction kernel (Step 9)

If the derivation is right, closed form == unrolling == retrain to machine precision.
"""
import numpy as np


def build(D=6, N=20, K=8, eta=0.05, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((D, D)); A = M @ M.T / D + 0.5 * np.eye(D)   # shared PSD Hessian
    c = rng.standard_normal((N, D)); S = np.sort(rng.choice(N, K, replace=False))
    P = np.diag(0.5 + rng.random(D)); theta0 = rng.standard_normal(D)
    d = A @ (c.mean(0) - c[S].mean(0))                                   # g_S - gbar (constant)
    barg = lambda th: A @ (th - c.mean(0))
    return dict(A=A, P=P, eta=eta, theta0=theta0, d=d, barg=barg, D=D, rng=rng)


def unroll(env, eps, k):
    th = env["theta0"].copy()
    for _ in range(k):
        th = th - env["eta"] * (env["P"] @ (env["barg"](th) + eps * env["d"]))
    return th


def Rk(env, k):
    D, eta, P, A = env["D"], env["eta"], env["P"], env["A"]
    Am = np.eye(D) - eta * (P @ A)
    return (np.eye(D) - np.linalg.matrix_power(Am, k)) @ np.linalg.inv(A)


def main():
    env = build()
    D, A, P, eta, d = env["D"], env["A"], env["P"], env["eta"], env["d"]
    rng = env["rng"]
    ok = True

    print("=== Test A: resolvent vs exact unrolling & retrain (linear f) ===")
    gtest = rng.standard_normal(D)
    for k in (1, 10, 50, 200):
        Icf = -gtest @ Rk(env, k) @ d
        Ifd = (gtest @ unroll(env, 1e-6, k) - gtest @ unroll(env, 0, k)) / 1e-6
        dF = gtest @ unroll(env, 1.0, k) - gtest @ unroll(env, 0.0, k)
        m = np.allclose(Icf, Ifd) and np.allclose(Icf, dF)
        ok &= m
        print(f"  k={k:3d}  closed={Icf:+.6f}  FD_deriv={Ifd:+.6f}  retrain_dF={dF:+.6f}  match={m}")

    print("\n=== limits ===")
    l1 = np.allclose(Rk(env, 1), eta * P)
    linf = np.allclose(Rk(env, 4000), np.linalg.inv(A), atol=1e-4)
    ok &= l1 and linf
    print(f"  R_1 == eta*P: {l1}   R_inf == A^-1: {linf}")

    print("\n=== diagonal [R_k]_jj closed form ===")
    Ad = np.diag(0.3 + rng.random(D)); Pd = np.diag(1 / (np.sqrt(np.diag(Ad)) + 1e-8))
    k = 37
    Rk_mat = (np.eye(D) - np.linalg.matrix_power(np.eye(D) - eta * (Pd @ Ad), k)) @ np.linalg.inv(Ad)
    Rk_diag = (1 - (1 - eta * np.diag(Pd @ Ad)) ** k) / np.diag(Ad)
    md = np.allclose(np.diag(Rk_mat), Rk_diag); ok &= md
    print(f"  diagonal formula matches matrix R_k: {md}")

    print("\n=== Test B: eps=1 interaction kernel (quadratic f) ===")
    Mb = rng.standard_normal((D, D)); B = Mb @ Mb.T / D + 0.2 * np.eye(D)
    ctest = rng.standard_normal(D)
    f = lambda th: 0.5 * (th - ctest) @ B @ (th - ctest)
    for k in (10, 80):
        th0k = unroll(env, 0, k); thk1 = unroll(env, 1, k)
        Rd = Rk(env, k) @ d
        affine = np.allclose(thk1, th0k - Rd)
        gtrue = B @ (th0k - ctest)
        pred = -gtrue @ Rd + 0.5 * (Rd @ B @ Rd)          # Step-9 formula w/ end-point g
        dF = f(thk1) - f(th0k)
        m = np.allclose(dF, pred); ok &= (m and affine)
        print(f"  k={k:3d}  retrain_dF={dF:+.6f}  step9_pred={pred:+.6f}  affine_path={affine}  match={m}")

    print(f"\nALL CHECKS PASS: {ok}")
    return ok


if __name__ == "__main__":
    main()
