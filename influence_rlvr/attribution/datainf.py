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
        self.g_train_list = g_train_list
        chunk_size = 10

        if self.n == 0:
            self._grad_norms = None
            self.K = torch.empty(0, 0, dtype=torch.float32)
            self.M = torch.empty(0, 0, dtype=torch.float32)
            return

        dev = g_train_list[0].device
        if self.normalize:
            norms = torch.empty(self.n, dtype=torch.float32, device=dev)
            for i in range(self.n):
                norms[i] = g_train_list[i].float().norm().clamp(min=1e-12)
            self._grad_norms = norms
        else:
            self._grad_norms = None

        K = torch.empty(self.n, self.n, dtype=torch.float32, device="cpu")
        for i in range(0, self.n, chunk_size):
            i_end = min(i + chunk_size, self.n)
            chunk_i = torch.stack(
                [g_train_list[k].float() for k in range(i, i_end)]
            ).to("cuda")
            if self.normalize:
                chunk_i = chunk_i / self._grad_norms[i:i_end].unsqueeze(1).to(
                    "cuda"
                )
            for j in range(0, self.n, chunk_size):
                j_end = min(j + chunk_size, self.n)
                chunk_j = torch.stack(
                    [g_train_list[k].float() for k in range(j, j_end)]
                ).to("cuda")
                if self.normalize:
                    chunk_j = chunk_j / self._grad_norms[j:j_end].unsqueeze(1).to(
                        "cuda"
                    )
                block = chunk_i @ chunk_j.T
                K[i:i_end, j:j_end] = block.cpu()
                del chunk_j
                del block
            del chunk_i

        self.K = K
        I_n = torch.eye(self.n, dtype=torch.float32, device="cpu")
        self.M = torch.linalg.inv(self.n * lambda_damp * I_n + K)
        del I_n

    def _normalize_test(self, g_test: torch.Tensor) -> torch.Tensor:
        g = g_test.float()
        if self.normalize:
            g = g / (g.norm() + 1e-12)
        return g

    def compute_all_scores(self, test_info: dict) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0, dtype=np.float32)
        g_test = self._normalize_test(test_info["grad"]).to("cuda")
        lam = self.lambda_damp
        chunk_size = 10
        v = torch.empty(self.n, dtype=torch.float32, device="cuda")
        for i in range(0, self.n, chunk_size):
            i_end = min(i + chunk_size, self.n)
            chunk_i = torch.stack(
                [self.g_train_list[k].float() for k in range(i, i_end)]
            ).to("cuda")
            if self.normalize:
                chunk_i = chunk_i / self._grad_norms[i:i_end].unsqueeze(1).to(
                    "cuda"
                )
            v[i:i_end] = chunk_i @ g_test
            del chunk_i
        v_cpu = v.cpu()
        del v
        scores = (1.0 / lam) * v_cpu - (1.0 / lam**2) * (v_cpu @ self.M @ self.K)
        return scores.detach().cpu().numpy()

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        if self.n == 0:
            return 0.0
        g_test = self._normalize_test(test_info["grad"]).to("cuda")
        g_train = train_info["grad"].float().to("cuda")
        if self.normalize:
            g_train = g_train / (g_train.norm() + 1e-12)
        lam = self.lambda_damp
        chunk_size = 10
        Jg = torch.empty(self.n, dtype=torch.float32, device="cuda")
        for i in range(0, self.n, chunk_size):
            i_end = min(i + chunk_size, self.n)
            chunk_i = torch.stack(
                [self.g_train_list[k].float() for k in range(i, i_end)]
            ).to("cuda")
            if self.normalize:
                chunk_i = chunk_i / self._grad_norms[i:i_end].unsqueeze(1).to(
                    "cuda"
                )
            Jg[i:i_end] = chunk_i @ g_test
            del chunk_i
        My = self.M @ Jg.cpu()
        del Jg
        acc = torch.zeros_like(g_test)
        for i in range(0, self.n, chunk_size):
            i_end = min(i + chunk_size, self.n)
            chunk_i = torch.stack(
                [self.g_train_list[k].float() for k in range(i, i_end)]
            ).to("cuda")
            if self.normalize:
                chunk_i = chunk_i / self._grad_norms[i:i_end].unsqueeze(1).to(
                    "cuda"
                )
            acc += chunk_i.T @ My[i:i_end].to("cuda")
            del chunk_i
        h_inv_g = (1.0 / lam) * g_test - (1.0 / lam**2) * acc
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
            n_test = len(checkpoint["test_infos"])
            n_train = len(checkpoint["train_infos"])
            if checkpoint.get("datainf_checkpoint_matrix") is not None:
                matrix = np.asarray(
                    checkpoint["datainf_checkpoint_matrix"],
                    dtype=np.float32,
                )
            else:
                g_train_list = [info["grad"] for info in checkpoint["train_infos"]]
                datainf = DataInfInfluence(
                    g_train_list,
                    lambda_damp=self.lambda_damp,
                    normalize=self.normalize,
                )
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
