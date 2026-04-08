import numpy as np
import torch

from .base import BaseInfluenceMethod


def tracin_base_matrix_from_infos(test_infos, train_infos, normalize: bool) -> np.ndarray:
    with torch.no_grad():
        n_test = len(test_infos)
        n_train = len(train_infos)
        matrix = np.zeros((n_test, n_train), dtype=np.float32)

        if n_test == 0 or n_train == 0:
            return matrix

        if normalize:
            test_norms = []
            for t in test_infos:
                x = t["grad"].to(device="cuda", dtype=torch.float32)
                test_norms.append(x.norm().item())
                del x
            train_norms = []
            for t in train_infos:
                x = t["grad"].to(device="cuda", dtype=torch.float32)
                train_norms.append(x.norm().item())
                del x
        else:
            test_norms = [1.0] * n_test
            train_norms = [1.0] * n_train

        for i, t_info in enumerate(test_infos):
            t_grad_gpu = t_info["grad"].to(device="cuda", dtype=torch.float32)
            for j, tr_info in enumerate(train_infos):
                tr_grad_gpu = tr_info["grad"].to(
                    device="cuda", dtype=torch.float32
                )
                dot_val = torch.dot(t_grad_gpu, tr_grad_gpu).item()
                if normalize:
                    dot_val = dot_val / (test_norms[i] * train_norms[j])
                matrix[i, j] = dot_val
                del tr_grad_gpu
            del t_grad_gpu

        return matrix


def _stack_train_weights(infos):
    return torch.tensor(
        [float(info.get("historical_weight", 1.0)) for info in infos],
        dtype=torch.float32,
    )


class TracInInfluence(BaseInfluenceMethod):
    """
    First-order influence via learning-rate-weighted gradient dot product.

    When normalize=True (default), gradients are L2-normalized before the
    dot product, giving cosine similarity scaled by lr.

    When normalize=False, uses the raw dot product.

    Required keys: test_info["grad"], train_info["grad"]
    """

    def __init__(self, learning_rate: float = 1e-4, normalize: bool = True):
        self.learning_rate = learning_rate
        self.normalize = normalize

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        g_test = test_info["grad"]
        g_train = train_info["grad"]
        if self.normalize:
            g_test = g_test / (g_test.norm() + 1e-12)
            g_train = g_train / (g_train.norm() + 1e-12)
        weight = float(train_info.get("historical_weight", 1.0))
        return (self.learning_rate * weight * torch.dot(g_test, g_train)).item()


class TrajectoryTracInInfluence:
    def __init__(self, normalize: bool = False):
        self.normalize = normalize

    def compute_matrix(self, checkpoint_infos: list, return_breakdown: bool = False):
        if not checkpoint_infos:
            empty = np.zeros((0, 0), dtype=np.float32)
            return (empty, []) if return_breakdown else empty

        total_matrix = None
        breakdown = []

        for checkpoint in checkpoint_infos:
            train_weights = _stack_train_weights(checkpoint["train_infos"])
            if checkpoint.get("checkpoint_matrix") is not None:
                matrix = np.asarray(
                    checkpoint["checkpoint_matrix"],
                    dtype=np.float32,
                )
            else:
                matrix = tracin_base_matrix_from_infos(
                    checkpoint["test_infos"],
                    checkpoint["train_infos"],
                    self.normalize,
                )
            learning_rate = float(checkpoint.get("learning_rate", 1.0))
            weighted_matrix = learning_rate * matrix * train_weights.numpy()[None, :]

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
                    "train_weights": train_weights.numpy(),
                })

        return (total_matrix, breakdown) if return_breakdown else total_matrix
