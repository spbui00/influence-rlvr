import numpy as np
import torch

from .base import BaseInfluenceMethod


def _stack_grads(infos, normalize):
    grads = torch.stack([info["grad"].float() for info in infos])
    if normalize:
        norms = grads.norm(dim=1, keepdim=True).clamp(min=1e-12)
        grads = grads / norms
    return grads


def tracin_base_matrix_from_infos(test_infos, train_infos, normalize: bool) -> np.ndarray:
    if not test_infos or not train_infos:
        return np.zeros(
            (len(test_infos), len(train_infos)),
            dtype=np.float32,
        )
    test_grads = _stack_grads(test_infos, normalize)
    train_grads = _stack_grads(train_infos, normalize)
    matrix = (test_grads @ train_grads.T).cpu().numpy().astype(np.float32, copy=False)
    del test_grads, train_grads
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
                test_grads = _stack_grads(checkpoint["test_infos"], self.normalize)
                train_grads = _stack_grads(checkpoint["train_infos"], self.normalize)
                matrix = (test_grads @ train_grads.T).cpu().numpy()
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
