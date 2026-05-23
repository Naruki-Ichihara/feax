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
from .shell import (
    isotropic_in_plane_stiffness,
    orthotropic_in_plane_stiffness,
    plate_stiffness,
    z_layer_coordinates,
    rotate_in_plane_stiffness,
    rotate_shear_stiffness,
    laminate_stiffness,
    thermal_expansion_from_orientation,
    rotate_thermal_expansion,
    laminate_thermal_loads,
    mindlin_strains,
    mindlin_resultants,
    mindlin_weak_form,
    MindlinPlate,
    make_mindlin_plate,
)
from .orientation import (
    box_to_triangle,
    box_to_tetrahedron,
    smooth_sgn,
    orientation_tensor_2d,
    orientation_tensor_3d,
    principal_direction,
    quadratic_closure,
    linear_closure,
    hybrid_closure,
    orientation_averaged_stiffness,
    orientation_averaged_stiffness_3d,
)

__all__ = [
    "CohesiveInterface",
    "exponential_potential",
    "bilinear_potential",
    "compute_trapezoidal_weights",
    "compute_lumped_area_weights",
    "ThirdMediumContact",
    "classify_medium_cells",
    "isotropic_in_plane_stiffness",
    "orthotropic_in_plane_stiffness",
    "plate_stiffness",
    "z_layer_coordinates",
    "rotate_in_plane_stiffness",
    "rotate_shear_stiffness",
    "laminate_stiffness",
    "thermal_expansion_from_orientation",
    "rotate_thermal_expansion",
    "laminate_thermal_loads",
    "mindlin_strains",
    "mindlin_resultants",
    "mindlin_weak_form",
    "MindlinPlate",
    "make_mindlin_plate",
    "box_to_triangle",
    "box_to_tetrahedron",
    "smooth_sgn",
    "orientation_tensor_2d",
    "orientation_tensor_3d",
    "principal_direction",
    "quadratic_closure",
    "linear_closure",
    "hybrid_closure",
    "orientation_averaged_stiffness",
    "orientation_averaged_stiffness_3d",
]
