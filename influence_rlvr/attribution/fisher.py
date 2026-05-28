import numpy as np
import torch

from .base import BaseInfluenceMethod
from .tracin import _stack_train_weights


def _influence_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _geometry_weight_vector(infos):
    if not infos:
        return torch.zeros(0, dtype=torch.float32)
    raw_weights = torch.tensor(
        [float(info.get("historical_weight", 0.0)) for info in infos],
        dtype=torch.float32,
    )
    if torch.any(raw_weights > 0):
        total = raw_weights.sum().clamp(min=1e-12)
        return raw_weights / total
    return torch.full(
        (len(infos),),
        1.0 / max(len(infos), 1),
        dtype=torch.float32,
    )


class MeanScoreFisherInfluence(BaseInfluenceMethod):
    """
    Outer-of-means Fisher with a selectable solver.

        F = Σ_i w_i · x_i x_i^T + λI,    x_i = ∇log mean_y π(y|z_i)

    `x_i` is one *averaged* score vector per training prompt, so F is rank
    ≤ n_train plus damping. Two solvers consume the same operator:

      - `solver="woodbury"` (default): Woodbury identity, inverts an n×n
        matrix instead of D×D. O(n² · D) space; use when D >> n_train
        (typical LLM scale).
      - `solver="full"`: materialize and `torch.linalg.inv` the explicit
        D×D Fisher. Use when D is small enough that the explicit inverse
        is cheaper than the Woodbury structure (toy / small-feature regime).

    For the *true* policy Fisher F = Σ_i Σ_y π(y|z_i) g g^T, use
    `TrueFisherInfluence` (direct Cholesky) or `CGInfluence` (CG+FVP).

    Required train_info keys:
      train_info["grad"]              — flat D-vector training-side gradient.
      train_info["geometry_feature"]  — flat D-vector ∇log mean_y π(y|z_i).
      train_info["historical_weight"] — optional, defaults to uniform 1/n.
    """

    _SOLVERS = ("woodbury", "full")

    def __init__(
        self,
        train_infos: list,
        lambda_damp: float = 0.1,
        normalize: bool = False,
        solver: str = "woodbury",
    ):
        if solver not in self._SOLVERS:
            raise ValueError(
                f"Unsupported solver: {solver!r}. Use one of {self._SOLVERS}."
            )
        with torch.no_grad():
            self.device = _influence_device()
            self.lambda_damp = float(lambda_damp)
            self.normalize = bool(normalize)
            self.solver = solver
            self._train_grad_list = [info["grad"] for info in train_infos]
            self._geometry_list = []
            for info in train_infos:
                gf = info.get("geometry_feature")
                if gf is None:
                    raise ValueError(
                        "MeanScoreFisherInfluence requires 'geometry_feature' on each train info."
                    )
                self._geometry_list.append(gf)
            n = len(self._train_grad_list)
            if n != len(self._geometry_list):
                raise ValueError("train_infos grad and geometry_feature counts differ.")

            self._grad_norms = None
            self._geometry_norms = None
            if n > 0 and normalize:
                grad_norms = torch.empty(n, dtype=torch.float32, device=self.device)
                geom_norms = torch.empty(n, dtype=torch.float32, device=self.device)
                for i in range(n):
                    gi = self._train_grad_list[i].to(device=self.device, dtype=torch.float32)
                    xi = self._geometry_list[i].to(device=self.device, dtype=torch.float32)
                    grad_norms[i] = gi.norm().clamp(min=1e-12)
                    geom_norms[i] = xi.norm().clamp(min=1e-12)
                self._grad_norms = grad_norms
                self._geometry_norms = geom_norms

            self.geometry_weights = _geometry_weight_vector(train_infos).to(self.device)

            # Solver-specific state: at most one of these is populated.
            self._inverse_small: torch.Tensor | None = None   # (n × n), Woodbury
            self._inverse_fisher: torch.Tensor | None = None  # (D × D), full

            if n == 0:
                return
            if solver == "woodbury":
                self._setup_woodbury(n)
            else:
                self._setup_full(n)

    def _geometry_weighted_chunk(self, start: int, end: int) -> torch.Tensor:
        chunk = torch.stack([
            self._geometry_list[k].to(device=self.device, dtype=torch.float32)
            for k in range(start, end)
        ])
        if self.normalize:
            chunk = chunk / self._geometry_norms[start:end].unsqueeze(1)  # type: ignore[index]
        chunk = chunk * self.geometry_weights[start:end].sqrt().unsqueeze(1)
        return chunk

    def _setup_woodbury(self, n: int) -> None:
        chunk_size = 10
        gram = torch.empty(n, n, dtype=torch.float32, device="cpu")
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            chunk_i = self._geometry_weighted_chunk(i, i_end)
            for j in range(0, n, chunk_size):
                j_end = min(j + chunk_size, n)
                chunk_j = self._geometry_weighted_chunk(j, j_end)
                gram[i:i_end, j:j_end] = (chunk_i @ chunk_j.T).cpu()
                del chunk_j
            del chunk_i
        identity = torch.eye(n, dtype=torch.float32, device="cpu")
        self._inverse_small = torch.linalg.inv(identity + gram / self.lambda_damp)

    def _setup_full(self, n: int) -> None:
        features = torch.stack([
            self._geometry_list[i].to(device=self.device, dtype=torch.float32)
            for i in range(n)
        ])
        if self.normalize:
            features = features / self._geometry_norms.unsqueeze(1)  # type: ignore[union-attr]
        features = features * self.geometry_weights.sqrt().unsqueeze(1)
        dim = int(features.shape[1])
        fisher = features.T @ features
        fisher = fisher + self.lambda_damp * torch.eye(dim, dtype=torch.float32, device=self.device)
        self._inverse_fisher = torch.linalg.inv(fisher)

    def _precondition(self, g_test: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g = g_test.to(device=self.device, dtype=torch.float32)
            if self.normalize:
                g = g / (g.norm() + 1e-12)
            n = len(self._geometry_list)
            if n == 0:
                return (1.0 / self.lambda_damp) * g
            if self.solver == "full":
                assert self._inverse_fisher is not None
                return self._inverse_fisher @ g
            # Woodbury: h = (1/λ) g − (1/λ²) X^T (I + XX^T/λ)^{-1} X g
            assert self._inverse_small is not None
            chunk_size = 10
            features_g = torch.empty(n, dtype=torch.float32, device=self.device)
            for i in range(0, n, chunk_size):
                i_end = min(i + chunk_size, n)
                chunk = self._geometry_weighted_chunk(i, i_end)
                features_g[i:i_end] = chunk @ g
                del chunk
            temp = self._inverse_small.to(self.device) @ features_g
            del features_g
            correction = torch.zeros_like(g)
            for i in range(0, n, chunk_size):
                i_end = min(i + chunk_size, n)
                chunk = self._geometry_weighted_chunk(i, i_end)
                correction += chunk.T @ temp[i:i_end]
                del chunk
            return (1.0 / self.lambda_damp) * g - (1.0 / self.lambda_damp**2) * correction

    def compute_all_scores(self, test_info: dict) -> np.ndarray:
        with torch.no_grad():
            n = len(self._train_grad_list)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            h_inv_g = self._precondition(test_info["grad"])
            chunk_size = 10
            out = np.empty(n, dtype=np.float32)
            for i in range(0, n, chunk_size):
                i_end = min(i + chunk_size, n)
                chunk = torch.stack([
                    self._train_grad_list[k].to(device=self.device, dtype=torch.float32)
                    for k in range(i, i_end)
                ])
                if self.normalize:
                    chunk = chunk / self._grad_norms[i:i_end].unsqueeze(1)  # type: ignore[index]
                out[i:i_end] = (chunk @ h_inv_g).detach().cpu().numpy()
                del chunk
            return out

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        with torch.no_grad():
            g_train = train_info["grad"].to(device=self.device, dtype=torch.float32)
            if self.normalize:
                g_train = g_train / (g_train.norm() + 1e-12)
            h_inv_g = self._precondition(test_info["grad"])
            return torch.dot(h_inv_g, g_train).item()


class TrueFisherInfluence(BaseInfluenceMethod):
    """
    Direct (Cholesky) inversion of the *true* policy Fisher.

        F = (1/n) Σ_i Σ_y π(y|z_i) ∇log π(y|z_i) ∇log π(y|z_i)^T + λI

    Consumes the same `grad_cache` / `prob_cache` that
    `policy_fisher_fvp_from_grad_cache` (and thus `CGInfluence`) uses, so this
    class and CG should produce the same `h` to machine precision. Use this
    as the gold-standard reference when validating CG, or when D is small
    enough that an explicit D×D Cholesky is cheaper than CG.

    Required train_info keys:
      train_info["grad"] — flat D-vector training-side gradient.

    grad_cache[i]: (n_y, D) tensor of ∇log π(y|z_i) for every rolled-out y.
    prob_cache[i]: (n_y,)  tensor of π(y|z_i) under the current policy.
    """

    def __init__(
        self,
        train_infos: list,
        grad_cache: list,
        prob_cache: list,
        lambda_damp: float = 0.1,
    ):
        with torch.no_grad():
            self.device = _influence_device()
            self.lambda_damp = float(lambda_damp)
            self._train_grad_list = [info["grad"] for info in train_infos]
            n = len(grad_cache)
            if n != len(prob_cache):
                raise ValueError(
                    f"grad_cache (len {n}) and prob_cache (len {len(prob_cache)}) must match."
                )
            if n == 0:
                self._L = None
                self._D = 0
                return

            D = int(grad_cache[0].shape[1])
            F = torch.zeros(D, D, dtype=torch.float32, device=self.device)
            for G_i, pi_i in zip(grad_cache, prob_cache):
                G = G_i.to(device=self.device, dtype=torch.float32)
                pi = pi_i.to(device=self.device, dtype=torch.float32)
                # G^T diag(π) G — per-prompt expected outer product
                F = F + G.T @ (pi.unsqueeze(1) * G)
            F = F / max(n, 1)
            F = F + self.lambda_damp * torch.eye(D, dtype=torch.float32, device=self.device)
            self._L = torch.linalg.cholesky(F)
            self._D = D

    def _precondition(self, g_test: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g = g_test.to(device=self.device, dtype=torch.float32)
            if self._L is None:
                return (1.0 / self.lambda_damp) * g
            return torch.cholesky_solve(g.unsqueeze(-1), self._L).squeeze(-1)

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        with torch.no_grad():
            h = self._precondition(test_info["grad"])
            g_train = train_info["grad"].to(device=self.device, dtype=torch.float32)
            return float(torch.dot(h, g_train).item())

    def compute_all_scores(self, test_info: dict) -> np.ndarray:
        with torch.no_grad():
            n = len(self._train_grad_list)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            h = self._precondition(test_info["grad"])
            G = torch.stack(
                [
                    self._train_grad_list[i].to(device=self.device, dtype=torch.float32)
                    for i in range(n)
                ]
            )
            return (G @ h).detach().cpu().numpy().astype(np.float32, copy=False)


class TrajectoryFisherInfluence:
    def __init__(
        self,
        lambda_damp: float = 0.1,
        normalize: bool = False,
        solver: str = "woodbury",
    ):
        self.lambda_damp = lambda_damp
        self.normalize = normalize
        self.solver = str(solver).strip().lower()
        if self.solver not in {"woodbury", "full"}:
            raise ValueError(f"Unsupported Fisher solver: {solver!r}. Use 'woodbury' or 'full'.")

    def _build_fisher(self, train_infos: list) -> BaseInfluenceMethod:
        return MeanScoreFisherInfluence(
            train_infos,
            lambda_damp=self.lambda_damp,
            normalize=self.normalize,
            solver=self.solver,
        )

    def compute_matrix(self, checkpoint_infos: list, return_breakdown: bool = False):
        if not checkpoint_infos:
            empty = np.zeros((0, 0), dtype=np.float32)
            return (empty, []) if return_breakdown else empty

        total_matrix = None
        breakdown = []

        for checkpoint in checkpoint_infos:
            n_test = len(checkpoint["test_infos"])
            n_train = len(checkpoint["train_infos"])
            if checkpoint.get("fisher_checkpoint_matrix") is not None:
                # precomputed matrix for this checkpoint, just load it
                matrix = np.asarray(
                    checkpoint["fisher_checkpoint_matrix"],
                    dtype=np.float32,
                )
            else:
                fisher = self._build_fisher(checkpoint["train_infos"])
                matrix = np.zeros((n_test, n_train), dtype=np.float32)
                for idx, test_info in enumerate(checkpoint["test_infos"]):
                    matrix[idx] = fisher.compute_all_scores(test_info) # IF of all train examples for test example test_info

            train_weights = _stack_train_weights(checkpoint["train_infos"]).numpy()
            learning_rate = float(checkpoint.get("learning_rate", 1.0))
            weighted_matrix = learning_rate * matrix * train_weights[None, :]

            if total_matrix is None:
                total_matrix = weighted_matrix
            else:
                total_matrix = total_matrix + weighted_matrix

            if return_breakdown:
                breakdown.append({
                    "step": checkpoint["step"],
                    "learning_rate": learning_rate,
                    "matrix": matrix,
                    "weighted_matrix": weighted_matrix,
                    "train_weights": train_weights,
                    "solver": self.solver,
                })

        return (total_matrix, breakdown) if return_breakdown else total_matrix
