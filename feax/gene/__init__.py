"""
FEAX Gene - Generative Design in FEAX

A comprehensive toolkit for generative design and topology optimization.
Provides efficient, JIT-compiled implementations of common topology optimization
components that work seamlessly with the FEAX finite element framework.

Components:
    Response Functions:
        - create_compliance_fn: Universal compliance (strain energy) calculation
        - create_volume_fn: Material volume fraction computation

    Filtering & Projections:
        - create_helmholtz_filter: PDE-based Helmholtz filtering
        - create_density_filter: Distance-based weighted averaging filter
        - helmholtz_filter: One-shot Helmholtz filtering (for non-gradient use)
        - density_filter: One-shot density filtering (for non-gradient use)
        - heaviside_projection: Sharp void/solid boundary projection
        - compute_volume_fraction_threshold: Threshold for target volume fraction

    Adaptive Remeshing:
        - adaptive: Gmsh-based adaptive remeshing for topology optimization

Example:
    >>> import feax
    >>> from feax.gene import (create_compliance_fn, create_volume_fn,
    ...                        create_density_filter, mdmm)
    >>>
    >>> # Create response functions
    >>> compliance_fn = create_compliance_fn(problem)
    >>> volume_fn = create_volume_fn(problem)
    >>>
    >>> # Create density filter
    >>> filter_fn = create_density_filter(mesh, radius=3.0)
    >>>
    >>> # Define objective with filtering
    >>> def objective(rho):
    ...     rho_filtered = filter_fn(rho)
    ...     sol = solve(rho_filtered)
    ...     return compliance_fn(sol)
    >>>
    >>> # Add volume constraint with MDMM
    >>> constraint = mdmm.ineq(
    ...     lambda rho: 0.4 - volume_fn(filter_fn(rho)),
    ...     damping=10.0
    ... )
"""

from . import adaptive
from . import mesh_extraction
from . import optimizer
from . import sdf
from .mesh_extraction import extract_surface, extract_volume_mesh
from .filters import (
    create_density_filter,
    create_helmholtz_filter,
    density_filter,
    helmholtz_filter,
    heaviside_projection,
    compute_volume_fraction_threshold,
)
from .responses import create_compliance_fn, create_dynamic_compliance_fn, create_volume_fn
from .sdf import DensityField, from_opt_result, from_vtk

__all__ = [
    # Response functions
    'create_compliance_fn',
    'create_dynamic_compliance_fn',
    'create_volume_fn',

    # Filtering and transformations
    'create_helmholtz_filter',
    'helmholtz_filter',
    'create_density_filter',
    'density_filter',

    # Projection and threshold
    'heaviside_projection',
    'compute_volume_fraction_threshold',

    # SDF conversion
    'sdf',
    'DensityField',
    'from_opt_result',
    'from_vtk',

    # Mesh extraction (pyvista)
    'mesh_extraction',
    'extract_surface',
    'extract_volume_mesh',

    # Adaptive remeshing
    'adaptive',

    # Optimization driver
    'optimizer',
]

__version__ = "1.0.0"
