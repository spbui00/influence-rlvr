import torch

from .base import BaseInfluenceMethod


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
