from .base import BaseInfluenceMethod


class RepSimInfluence(BaseInfluenceMethod):
    """
    Influence via cosine similarity of last hidden state embeddings.
    Not yet implemented.

    Required keys: test_info["hidden_states"], train_info["hidden_states"]
    """

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        raise NotImplementedError(
            "RepSim is not yet implemented. "
            "Requires last hidden state embeddings."
        )
