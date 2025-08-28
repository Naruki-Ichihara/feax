"""FEAX Lattice Toolkit - Periodic structures and homogenization utilities.

This subpackage provides tools for working with periodic lattice structures,
unit cell analysis, and computational homogenization.

Key modules:
    pbc: Periodic boundary condition utilities
    unitcell: Unit cell definition and mesh generation
    solver: Specialized solvers for homogenization problems
"""

from .solver import (
    create_homogenization_solver,
    create_unit_cell_solver, 
    create_macro_displacement_field
)

__all__ = [
    'create_homogenization_solver',
    'create_unit_cell_solver',
    'create_macro_displacement_field'
]