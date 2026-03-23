import numpy as np
import torch

from .base import BaseInfluenceMethod


def _stack_grads(infos, normalize):
    grads = torch.stack([info["grad"].float() for info in infos])
    if normalize:
        norms = grads.norm(dim=1, keepdim=True).clamp(min=1e-12)
        grads = grads / norms
    return grads


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
        return (self.learning_rate * torch.dot(g_test, g_train)).item()


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
            test_grads = _stack_grads(checkpoint["test_infos"], self.normalize)
            train_grads = _stack_grads(checkpoint["train_infos"], self.normalize)
            matrix = (test_grads @ train_grads.T).cpu().numpy()
            learning_rate = float(checkpoint.get("learning_rate", 1.0))
            weighted_matrix = learning_rate * matrix

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
                })

        return (total_matrix, breakdown) if return_breakdown else total_matrix
