from .base import BaseInfluenceMethod, InfluenceCalculator
from .tracin import TracInInfluence, TrajectoryTracInInfluence
from .datainf import DataInfInfluence, TrajectoryDataInfInfluence
from .fisher import (
    MeanScoreFisherInfluence,
    MeanScoreFisherWoodburyInfluence,
    TrueFisherInfluence,
    TrajectoryFisherInfluence,
    # Deprecated aliases (point to MeanScore* — outer-of-means operator):
    FisherInfluence,
    FisherWoodburyInfluence,
)
from .pbrf import PBRFInfluence
from .repsim import RepSimInfluence
from .cg import CGInfluence, policy_fisher_fvp_from_grad_cache

__all__ = [
    "BaseInfluenceMethod",
    "InfluenceCalculator",
    "TracInInfluence",
    "DataInfInfluence",
    "MeanScoreFisherInfluence",
    "MeanScoreFisherWoodburyInfluence",
    "TrueFisherInfluence",
    "TrajectoryTracInInfluence",
    "TrajectoryDataInfInfluence",
    "TrajectoryFisherInfluence",
    "PBRFInfluence",
    "RepSimInfluence",
    "CGInfluence",
    "policy_fisher_fvp_from_grad_cache",
    # Deprecated aliases:
    "FisherInfluence",
    "FisherWoodburyInfluence",
]
