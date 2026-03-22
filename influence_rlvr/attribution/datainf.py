import torch
import numpy as np

from .base import BaseInfluenceMethod


class DataInfInfluence(BaseInfluenceMethod):
    """
    Second-order influence via the Woodbury Matrix Identity, avoiding
    the D x D Hessian entirely.

    Given H = lambda * I_D + (1/n) J^T J, the Woodbury identity yields:
        H^{-1} = (1/lambda) I_D - (1/lambda^2) J^T (n*lambda*I_n + J J^T)^{-1} J

    The largest matrix ever inverted is n x n (number of training examples).

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
