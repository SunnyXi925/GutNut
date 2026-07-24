"""GMNPS validation interfaces.

The new Nature Food article framework uses preservation, heterogeneity and
digital-gut-twin benchmarks. Legacy independent-cohort scripts remain available
as direct module imports.
"""

from gmnps.validation.preservation import (
    fcs_category,
    heterogeneity_metrics,
    nps_preservation_metrics,
)
from gmnps.validation.synthetic_twin import (
    SyntheticTwinBundle,
    run_synthetic_benchmark,
    simulate_synthetic_twin,
)

__all__ = [
    "SyntheticTwinBundle",
    "fcs_category",
    "heterogeneity_metrics",
    "nps_preservation_metrics",
    "run_synthetic_benchmark",
    "simulate_synthetic_twin",
]
