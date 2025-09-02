"""
Type definitions for FEAX finite element analysis framework.

This module provides centralized type aliases for enhanced code readability,
maintainability, and consistency across the FEAX codebase.
"""

import jax.numpy as np
from jax.experimental import sparse
from typing import Callable, List, Tuple, Union, Protocol, TYPE_CHECKING, Any

# Python 3.10+ compatibility for TypeAlias
try:
    from typing import TypeAlias
except ImportError:
    # Fallback for Python < 3.10
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        # Final fallback - just use assignment
        def TypeAlias(x): return x  # type: ignore

if TYPE_CHECKING:
    from feax.problem import Problem
    from feax.internal_vars import InternalVars
    from feax.DCboundary import DirichletBC

# =============================================================================
# Array Types
# =============================================================================

# Basic array types
Array = np.ndarray
SolutionArray = np.ndarray  # (num_nodes, vec) - solution at nodes
CoordinateArray = np.ndarray  # (num_points, dim) - physical coordinates
ElementArray = np.ndarray  # (num_cells, num_nodes) - element connectivity

# Specialized arrays
GradientArray = np.ndarray  # (vec, dim) - gradient tensor at quadrature point
StressTensor = np.ndarray  # (vec, dim) - stress/flux tensor
MassTerm = np.ndarray  # (vec,) - mass/reaction term
TractionVector = np.ndarray  # (vec,) - surface traction


# =============================================================================
# Physics Kernel Types
# =============================================================================

# Physics maps (user-defined constitutive laws)
TensorMap = Callable[[GradientArray], StressTensor]
"""Function that maps gradient to stress/flux tensor.
Signature: (u_grad: (vec,dim), *internal_vars) -> stress_tensor: (vec,dim)
Note: Additional internal_vars parameters are passed as *args
"""

MassMap = Callable[[np.ndarray, CoordinateArray], MassTerm]  
"""Function that computes mass/reaction terms.
Signature: (u: (vec,), x: (dim,), *internal_vars) -> mass_term: (vec,)
Note: Additional internal_vars parameters are passed as *args
"""

SurfaceMap = Callable[[np.ndarray, CoordinateArray], TractionVector]
"""Function that computes surface traction/flux.
Signature: (u: (vec,), x: (dim,), *internal_vars) -> traction: (vec,)
Note: Additional internal_vars parameters are passed as *args
"""

# Element-level kernel functions
LaplaceKernel = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""Element kernel for gradient-based physics.
Signature: (cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *internal_vars) -> element_residual
Note: Additional internal_vars parameters are passed as *args
"""

MassKernel = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""Element kernel for mass/reaction terms.
Signature: (cell_sol_flat, x, cell_JxW, *internal_vars) -> element_residual
Note: Additional internal_vars parameters are passed as *args
"""

SurfaceKernel = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""Element kernel for surface integrals.
Signature: (cell_sol_flat, x, face_shape_vals, face_shape_grads, nanson_scale, *internal_vars) -> element_residual
Note: Additional internal_vars parameters are passed as *args
"""

# Combined kernels - using Protocol for flexibility
class VolumeKernelProtocol(Protocol):
    """Protocol for volume kernels with flexible arguments."""
    def __call__(self, *args: Any) -> np.ndarray: ...

class UniversalKernelProtocol(Protocol):
    """Protocol for universal kernels with full FE data access."""
    def __call__(self, *args: Any) -> np.ndarray: ...

VolumeKernel = VolumeKernelProtocol
UniversalKernel = UniversalKernelProtocol


# =============================================================================
# Assembly and Solver Types
# =============================================================================

# Sparse matrix type
SparseMatrix = sparse.BCOO

# Function types for FE assembly and solving
JacobianFunction = Callable[[np.ndarray, 'InternalVars'], SparseMatrix]
"""Function that computes Jacobian matrix with BC applied.
Signature: (sol_flat, internal_vars) -> jacobian_matrix
"""

ResidualFunction = Callable[[np.ndarray, 'InternalVars'], np.ndarray]
"""Function that computes residual vector with BC applied.
Signature: (sol_flat, internal_vars) -> residual_vector
"""

SolverFunction = Callable[['InternalVars', np.ndarray], np.ndarray]
"""Differentiable solver function.
Signature: (internal_vars, initial_guess) -> solution
"""

# Boundary condition types
LocationFunction = Callable[[CoordinateArray], Union[bool, np.ndarray]]
"""Function that identifies boundary locations.
Signature: (point_coords) -> is_on_boundary
"""

ValueFunction = Callable[[CoordinateArray], Union[float, np.ndarray]]
"""Function that provides boundary values.
Signature: (point_coords) -> boundary_value
"""


# =============================================================================
# Protocol Types for Extensibility
# =============================================================================

class ProblemProtocol(Protocol):
    """Protocol for Problem-like objects."""
    
    def get_tensor_map(self) -> TensorMap:
        """Return tensor map for gradient-based physics."""
        ...
    
    def get_mass_map(self) -> MassMap:
        """Return mass map for reaction/inertia terms."""
        ...
    
    def get_surface_maps(self) -> List[SurfaceMap]:
        """Return surface maps for boundary loads."""
        ...


class InternalVarsProtocol(Protocol):
    """Protocol for InternalVars-like objects."""
    
    volume_vars: Tuple[np.ndarray, ...]
    surface_vars: List[Tuple[np.ndarray, ...]]


# =============================================================================
# Utility Type Unions
# =============================================================================

# Common parameter types
MaterialParameter = Union[float, np.ndarray]
LoadParameter = Union[float, np.ndarray, Callable[[CoordinateArray], np.ndarray]]

# Solution types
SolutionList = List[SolutionArray]
ResidualList = List[np.ndarray]

# Optimization types  
ObjectiveFunction = Callable[['InternalVars'], float]
GradientFunction = Callable[['InternalVars'], 'InternalVars']