import numpy as np
import torch

from .base import BaseInfluenceMethod
from .tracin import _stack_train_weights


def _stack_geometry_features(infos, normalize):
    features = []
    for info in infos:
        feature = info.get("geometry_feature")
        if feature is None:
            raise ValueError(
                "Fisher influence requires 'geometry_feature' on each train info."
            )
        features.append(feature.float())
    matrix = torch.stack(features)
    if normalize:
        norms = matrix.norm(dim=1, keepdim=True).clamp(min=1e-12)
        matrix = matrix / norms
    return matrix


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


class FisherInfluence(BaseInfluenceMethod):
    def __init__(self, train_infos: list, lambda_damp: float = 0.1, normalize: bool = False):
        self.lambda_damp = lambda_damp
        self.normalize = normalize
        self.train_grads = torch.stack([info["grad"].float() for info in train_infos])
        if normalize:
            grad_norms = self.train_grads.norm(dim=1, keepdim=True).clamp(min=1e-12)
            self.train_grads = self.train_grads / grad_norms

        geometry_features = _stack_geometry_features(train_infos, normalize)
        geometry_weights = _geometry_weight_vector(train_infos)
        self.geometry_matrix = geometry_features * geometry_weights.sqrt().unsqueeze(1)
        gram = self.geometry_matrix @ self.geometry_matrix.T
        identity = torch.eye(gram.shape[0], dtype=torch.float32)
        self.inverse_small = torch.linalg.inv(identity + gram / lambda_damp)

    def _normalize_test(self, g_test: torch.Tensor) -> torch.Tensor:
        g = g_test.float()
        if self.normalize:
            g = g / (g.norm() + 1e-12)
        return g

    def _precondition(self, g_test: torch.Tensor) -> torch.Tensor:
        g = self._normalize_test(g_test)
        features_g = self.geometry_matrix @ g
        correction = self.geometry_matrix.T @ (self.inverse_small @ features_g)
        return (1.0 / self.lambda_damp) * g - (1.0 / self.lambda_damp**2) * correction

    def compute_all_scores(self, test_info: dict) -> np.ndarray:
        h_inv_g = self._precondition(test_info["grad"])
        return (self.train_grads @ h_inv_g).numpy()

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        g_train = train_info["grad"].float()
        if self.normalize:
            g_train = g_train / (g_train.norm() + 1e-12)
        h_inv_g = self._precondition(test_info["grad"])
        return torch.dot(h_inv_g, g_train).item()


class TrajectoryFisherInfluence:
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
            if checkpoint.get("fisher_checkpoint_matrix") is not None:
                matrix = np.asarray(
                    checkpoint["fisher_checkpoint_matrix"],
                    dtype=np.float32,
                )
            else:
                fisher = FisherInfluence(
                    checkpoint["train_infos"],
                    lambda_damp=self.lambda_damp,
                    normalize=self.normalize,
                )
                matrix = np.zeros((n_test, n_train), dtype=np.float32)
                for idx, test_info in enumerate(checkpoint["test_infos"]):
                    matrix[idx] = fisher.compute_all_scores(test_info)

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
