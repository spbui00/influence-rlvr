from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseInfluenceMethod(ABC):
    """
    Strategy interface for influence function methods.

    All strategies receive generic dicts (test_info, train_info) rather than
    bare tensors.  Each strategy documents which keys it expects:
      - "grad"          : flattened LoRA gradient vector (torch.Tensor)
      - "logits"        : model output logit distributions
      - "hidden_states" : last-layer hidden state embeddings
    This keeps the interface stable as we add methods that need non-gradient data.
    """

    @abstractmethod
    def compute_score(self, test_info: dict, train_info: dict) -> float:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — TracIn (First-Order, multi-checkpoint ready)
# ─────────────────────────────────────────────────────────────────────────────
class TracInInfluence(BaseInfluenceMethod):
    """
    First-order influence via learning-rate-weighted gradient dot product.

    When normalize=True (default), gradients are L2-normalized before the
    dot product, giving cosine similarity scaled by lr:

        score = lr * (g_test / ||g_test||) . (g_train / ||g_train||)

    When normalize=False, uses the raw dot product:

        score = lr * g_test^T @ g_train

    Normalization is recommended when g_test and g_train come from different
    loss functions (e.g. SFT vs GRPO) whose gradient magnitudes are not
    directly comparable.

    To aggregate over K checkpoints, call compute_score once per checkpoint
    and sum the results.  The caller controls the loop so we can lazily load
    checkpoint-specific gradients without holding them all in memory.

    Required keys: test_info["grad"], train_info["grad"]
    """

    def __init__(self, learning_rate: float = 1e-4, normalize: bool = True):
        self.learning_rate = learning_rate
        self.normalize = normalize

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        g_test = test_info["grad"]
        g_train = train_info["grad"]
        if self.normalize:
            norm_test = g_test.norm() + 1e-12
            norm_train = g_train.norm() + 1e-12
            g_test = g_test / norm_test
            g_train = g_train / norm_train
        return (self.learning_rate * torch.dot(g_test, g_train)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — DataInf (Second-Order, Woodbury approximation)
# ─────────────────────────────────────────────────────────────────────────────
class DataInfInfluence(BaseInfluenceMethod):
    """
    Second-order influence via the Woodbury Matrix Identity, avoiding
    the D×D Hessian entirely.

    Given the Empirical Fisher / Gauss-Newton Hessian:
        H = λI_D + (1/n) J^T J
    the Woodbury identity yields:
        H^{-1} = (1/λ)I_D - (1/λ²) J^T (nλI_n + JJ^T)^{-1} J

    For a single test gradient g_test, ALL n training scores are:
        v       = J @ g_test          — (n,) first-order dot products
        scores  = -(1/λ) v + (1/λ²) v @ M @ K
    where K = JJ^T (n×n) and M = (nλI_n + K)^{-1} (n×n).

    The largest matrix ever inverted is n×n (number of training examples),
    which is instant even on Apple MPS.

    When normalize=True, each training gradient row in J and each g_test
    are L2-normalized before scoring, isolating directional alignment.

    Required keys: test_info["grad"]
    """

    def __init__(self, g_train_list: list, lambda_damp: float = 0.1,
                 normalize: bool = True):
        self.lambda_damp = lambda_damp
        self.normalize = normalize
        self.n = len(g_train_list)

        J = torch.stack(g_train_list).float()
        if normalize:
            norms = J.norm(dim=1, keepdim=True).clamp(min=1e-12)
            J = J / norms

        self.J = J
        K = J @ J.T
        self.K = K
        I_n = torch.eye(self.n, dtype=torch.float32)
        self.M = torch.linalg.inv(self.n * lambda_damp * I_n + K)

        del I_n

    def _normalize_test(self, g_test: torch.Tensor) -> torch.Tensor:
        g = g_test.float()
        if self.normalize:
            g = g / (g.norm() + 1e-12)
        return g

    def compute_all_scores(self, test_info: dict) -> np.ndarray:
        g_test = self._normalize_test(test_info["grad"])
        lam = self.lambda_damp
        v = self.J @ g_test
        scores = -(1.0 / lam) * v + (1.0 / lam**2) * (v @ self.M @ self.K)
        return scores.numpy()

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        g_test = self._normalize_test(test_info["grad"])
        g_train = train_info["grad"].float()
        if self.normalize:
            g_train = g_train / (g_train.norm() + 1e-12)
        lam = self.lambda_damp
        Jg = self.J @ g_test
        h_inv_g = (1.0 / lam) * g_test - (1.0 / lam**2) * (self.J.T @ self.M @ Jg)
        return -(torch.dot(h_inv_g, g_train)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 — PBRF (Proximal Bregman Response Function)
# ─────────────────────────────────────────────────────────────────────────────
class PBRFInfluence(BaseInfluenceMethod):
    """
    Influence under KL-divergence / Bregman geometry (Proximal Bregman
    Response Function).

    Unlike gradient-only methods, PBRF operates on the model's output
    *distributions* rather than parameter gradients.  It measures how
    removing a training point shifts the predicted distribution under
    a KL-divergence proximal objective.

    This will likely require:
      - test_info["logits"]  : logit distribution on the test input
      - train_info["logits"] : logit distribution on the train input
      - Possibly the reference (pre-training) logits for the KL anchor.

    Required keys: test_info["logits"], train_info["logits"]
    """

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        # TODO: Implement PBRF scoring
        #   1. Convert logits to probability distributions (softmax).
        #   2. Compute KL-divergence based influence under Bregman proximal framework.
        #   3. May require a reference distribution (pre-fine-tune) as the KL anchor.
        raise NotImplementedError(
            "PBRF (Proximal Bregman Response Function) is not yet implemented. "
            "Requires storing full policy logit distributions, not just parameter gradients."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Method 4 — RepSim (Representation Similarity)
# ─────────────────────────────────────────────────────────────────────────────
class RepSimInfluence(BaseInfluenceMethod):
    """
    Influence via cosine similarity of last hidden state embeddings.

    Instead of comparing parameter gradients, this measures how similar
    the model's internal representations are for a test input vs. a
    training input—capturing "representational overlap" in activation space.

    Required keys: test_info["hidden_states"], train_info["hidden_states"]
    """

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        # TODO: Implement representation similarity
        #   1. Extract last hidden state vectors from test_info and train_info.
        #   2. Compute cosine similarity: cos(h_test, h_train).
        #   3. Optionally track the *shift* in embeddings before/after training.
        raise NotImplementedError(
            "RepSim (Representation Similarity) is not yet implemented. "
            "Requires last hidden state embeddings, not parameter gradients."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — builds the full N×M influence matrix
# ─────────────────────────────────────────────────────────────────────────────
class InfluenceCalculator:
    """
    Applies a chosen BaseInfluenceMethod strategy to every (test, train) pair
    and returns the N×M influence matrix as a numpy array.

    Usage:
        calc = InfluenceCalculator(TracInInfluence(learning_rate=1e-4))
        matrix = calc.compute_matrix(test_infos, train_infos)
    """

    def __init__(self, method: BaseInfluenceMethod):
        self.method = method

    def compute_matrix(self, test_infos: list, train_infos: list) -> np.ndarray:
        n_test = len(test_infos)
        n_train = len(train_infos)
        matrix = np.zeros((n_test, n_train))

        for i, t_info in enumerate(test_infos):
            for j, tr_info in enumerate(train_infos):
                matrix[i, j] = self.method.compute_score(t_info, tr_info)

        return matrix
