import torch
import numpy as np

from .base import BaseInfluenceMethod
from .tracin import _stack_train_weights


class DataInfInfluence(BaseInfluenceMethod):
    """
    Second-order influence via the Woodbury Matrix Identity, avoiding
    the D x D Hessian entirely.

    Given H = lambda * I_D + (1/n) J^T J, the Woodbury identity yields:
        H^{-1} = (1/lambda) I_D - (1/lambda^2) J^T (n*lambda*I_n + J J^T)^{-1} J

    The largest matrix ever inverted is n x n (number of training examples).

    Required keys: test_info["grad"]

    Sign convention:
    Returns helpful-positive scores, aligned with the lr-weighted TracIn
    convention used elsewhere in this project. Positive values mean the
    train gradient points in a direction that should decrease the test
    loss under the local second-order approximation.
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
        scores = (1.0 / lam) * v - (1.0 / lam**2) * (v @ self.M @ self.K)
        return scores.numpy()

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        g_test = self._normalize_test(test_info["grad"])
        g_train = train_info["grad"].float()
        if self.normalize:
            g_train = g_train / (g_train.norm() + 1e-12)
        lam = self.lambda_damp
        Jg = self.J @ g_test
        h_inv_g = (1.0 / lam) * g_test - (1.0 / lam**2) * (self.J.T @ self.M @ Jg)
        weight = float(train_info.get("historical_weight", 1.0))
        return (weight * torch.dot(h_inv_g, g_train)).item()


class TrajectoryDataInfInfluence:
    def __init__(self, lambda_damp: float = 0.1, normalize: bool = False):
        self.lambda_damp = lambda_damp
        self.normalize = normalize

    def compute_matrix(self, checkpoint_infos: list, return_breakdown: bool = False):
        if not checkpoint_infos:
            empty = np.zeros((0, 0), dtype=np.float32)
            return (empty, []) if return_breakdown else empty

        total_matrix = None
        breakdown = []

        for checkpoint in checkpoint_infos:
            g_train_list = [info["grad"] for info in checkpoint["train_infos"]]
            datainf = DataInfInfluence(
                g_train_list,
                lambda_damp=self.lambda_damp,
                normalize=self.normalize,
            )

            n_test = len(checkpoint["test_infos"])
            n_train = len(checkpoint["train_infos"])
            matrix = np.zeros((n_test, n_train), dtype=np.float32)
            for idx, test_info in enumerate(checkpoint["test_infos"]):
                matrix[idx] = datainf.compute_all_scores(test_info)

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
                })

        return (total_matrix, breakdown) if return_breakdown else total_matrix
