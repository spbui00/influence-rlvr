from .base import BaseInfluenceMethod


class PBRFInfluence(BaseInfluenceMethod):
    """
    Influence under KL-divergence / Bregman geometry (Proximal Bregman
    Response Function).  Not yet implemented.

    Required keys: test_info["logits"], train_info["logits"]
    """

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        raise NotImplementedError(
            "PBRF is not yet implemented. "
            "Requires storing full policy logit distributions."
        )
