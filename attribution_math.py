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
    First-order influence via learning-rate-weighted gradient dot product:

        score = lr * g_test^T @ g_train

    To aggregate over K checkpoints, call compute_score once per checkpoint
    and sum the results.  The caller controls the loop so we can lazily load
    checkpoint-specific gradients without holding them all in memory.

    Required keys: test_info["grad"], train_info["grad"]
    """

    def __init__(self, learning_rate: float = 1e-4):
        self.learning_rate = learning_rate

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        g_test = test_info["grad"]
        g_train = train_info["grad"]
        return (self.learning_rate * torch.dot(g_test, g_train)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — DataInf (Second-Order, Woodbury approximation)
# ─────────────────────────────────────────────────────────────────────────────
class DataInfInfluence(BaseInfluenceMethod):
    """
    Second-order parameter-space influence via the Woodbury Matrix Identity
    approximation of the inverse Hessian:

        score = g_test^T  H^{-1}  g_train

    where H^{-1} is approximated from the outer products of all training
    gradients using the Woodbury identity.

    Either pass a pre-computed inverse_hessian tensor or the full list of
    training gradient vectors (all_train_grads) so H^{-1} can be built
    on the fly.

    Required keys: test_info["grad"], train_info["grad"]
    """

    def __init__(self, inverse_hessian=None, all_train_grads=None, damping=1e-3):
        self.inverse_hessian = inverse_hessian
        self.all_train_grads = all_train_grads
        self.damping = damping

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        # TODO: Implement Woodbury identity approximation
        #   1. If self.inverse_hessian is None, build it from self.all_train_grads:
        #      H ≈ (1/N) * sum_i(g_i g_i^T) + damping * I
        #      H^{-1} via Woodbury: (A + UCV)^{-1}
        #   2. score = g_test^T @ H^{-1} @ g_train
        raise NotImplementedError(
            "DataInf (Woodbury inverse-Hessian approximation) is not yet implemented. "
            "Provide inverse_hessian or all_train_grads and implement the Woodbury update."
        )


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
