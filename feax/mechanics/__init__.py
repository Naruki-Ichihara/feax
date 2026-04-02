"""Mechanics building blocks for feax.

This package provides reusable mechanics components (cohesive zones,
contact, etc.) that can be composed with feax's matrix-free solver.
"""

from .cohesive import (
    CohesiveInterface,
    exponential_potential,
    bilinear_potential,
    compute_trapezoidal_weights,
    compute_lumped_area_weights,
)
from .tmc import (
    ThirdMediumContact,
    classify_medium_cells,
)

__all__ = [
    "CohesiveInterface",
    "exponential_potential",
    "bilinear_potential",
    "compute_trapezoidal_weights",
    "compute_lumped_area_weights",
    "ThirdMediumContact",
    "classify_medium_cells",
]
