"""
Prediction aggregator models f_lambda for FENS (Section 2.2, Section 3.6, Appendix B.3 & B.4).
"""

from fens.aggregators.models import (
    MLPAggregator,
    MoEAggregator,
    PerClassLinearAggregator,
    PolychotomousVotingAggregator,
    ScalarLinearAggregator,
    UniformAveragingAggregator,
    WeightedAveragingAggregator,
    build_aggregator,
)

__all__ = [
    "UniformAveragingAggregator",
    "WeightedAveragingAggregator",
    "ScalarLinearAggregator",
    "PerClassLinearAggregator",
    "MLPAggregator",
    "PolychotomousVotingAggregator",
    "MoEAggregator",
    "build_aggregator",
]
