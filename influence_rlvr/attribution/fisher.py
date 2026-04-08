import numpy as np
import torch

from .base import BaseInfluenceMethod
from .tracin import _stack_train_weights


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
        with torch.no_grad():
            self.lambda_damp = lambda_damp
            self.normalize = normalize
            self._train_grad_list = [info["grad"] for info in train_infos]
            self._geometry_list = []
            for info in train_infos:
                gf = info.get("geometry_feature")
                if gf is None:
                    raise ValueError(
                        "Fisher influence requires 'geometry_feature' on each train info."
                    )
                self._geometry_list.append(gf)
            n = len(self._train_grad_list)
            if n != len(self._geometry_list):
                raise ValueError("train_infos grad and geometry_feature counts differ.")

            if n > 0 and normalize:
                norms = torch.empty(n, dtype=torch.float32, device="cuda")
                for i in range(n):
                    gi = self._train_grad_list[i].to(
                        device="cuda", dtype=torch.float32
                    )
                    norms[i] = gi.norm().clamp(min=1e-12)
                    del gi
                self._grad_norms = norms
            else:
                self._grad_norms = None

            if n > 0 and normalize:
                geom_norms = torch.empty(n, dtype=torch.float32, device="cuda")
                for i in range(n):
                    xi = self._geometry_list[i].to(
                        device="cuda", dtype=torch.float32
                    )
                    geom_norms[i] = xi.norm().clamp(min=1e-12)
                    del xi
                self._geometry_norms = geom_norms
            else:
                self._geometry_norms = None

            self.geometry_weights = _geometry_weight_vector(train_infos).to("cuda")

            chunk_size = 10
            if n == 0:
                gram = torch.empty(0, 0, dtype=torch.float32, device="cpu")
                self.inverse_small = torch.empty(0, 0, dtype=torch.float32)
            else:
                gram = torch.empty(n, n, dtype=torch.float32, device="cpu")
                for i in range(0, n, chunk_size):
                    i_end = min(i + chunk_size, n)
                    chunk_i = self._geometry_weighted_chunk(i, i_end)
                    for j in range(0, n, chunk_size):
                        j_end = min(j + chunk_size, n)
                        chunk_j = self._geometry_weighted_chunk(j, j_end)
                        block = chunk_i @ chunk_j.T
                        gram[i:i_end, j:j_end] = block.cpu()
                        del chunk_j
                        del block
                    del chunk_i
                identity = torch.eye(n, dtype=torch.float32, device="cpu")
                self.inverse_small = torch.linalg.inv(
                    identity + gram / lambda_damp
                )
                del identity

    def _geometry_weighted_chunk(self, start: int, end: int) -> torch.Tensor:
        chunk = torch.stack(
            [
                self._geometry_list[k].to(device="cuda", dtype=torch.float32)
                for k in range(start, end)
            ]
        )
        if self.normalize:
            chunk = chunk / self._geometry_norms[start:end].unsqueeze(1)
        chunk = chunk * self.geometry_weights[start:end].sqrt().unsqueeze(1)
        return chunk

    def _precondition(self, g_test: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g = g_test.to(device="cuda", dtype=torch.float32)
            if self.normalize:
                g = g / (g.norm() + 1e-12)
            n = len(self._geometry_list)
            chunk_size = 10
            if n == 0:
                return (1.0 / self.lambda_damp) * g
            features_g = torch.empty(n, dtype=torch.float32, device="cuda")
            for i in range(0, n, chunk_size):
                i_end = min(i + chunk_size, n)
                chunk = self._geometry_weighted_chunk(i, i_end)
                features_g[i:i_end] = chunk @ g
                del chunk
            temp = self.inverse_small.to("cuda") @ features_g
            del features_g
            correction = torch.zeros_like(g)
            for i in range(0, n, chunk_size):
                i_end = min(i + chunk_size, n)
                chunk = self._geometry_weighted_chunk(i, i_end)
                correction += chunk.T @ temp[i:i_end]
                del chunk
            del temp
            return (1.0 / self.lambda_damp) * g - (
                1.0 / self.lambda_damp**2
            ) * correction

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
                chunk = torch.stack(
                    [
                        self._train_grad_list[k].to(
                            device="cuda", dtype=torch.float32
                        )
                        for k in range(i, i_end)
                    ]
                )
                if self.normalize:
                    chunk = chunk / self._grad_norms[i:i_end].unsqueeze(1)
                out[i:i_end] = (chunk @ h_inv_g).detach().cpu().numpy()
                del chunk
            return out

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        with torch.no_grad():
            g_train = train_info["grad"].to(
                device="cuda", dtype=torch.float32
            )
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
