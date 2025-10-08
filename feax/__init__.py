import jax
jax.config.update("jax_enable_x64", True)

# Main API
from .problem import Problem
from .internal_vars import InternalVars
from .assembler import get_J, get_res, create_J_bc_function, create_res_bc_function
from .mesh import Mesh
from .DCboundary import DirichletBC, apply_boundary_to_J, apply_boundary_to_res, DirichletBCSpec, DirichletBCConfig, dirichlet_bc_config
from .solver import newton_solve, SolverOptions, create_solver, linear_solve, newton_solve_fori, newton_solve_py
from .utils import zero_like_initial_guess
# Memory-efficient utilities
# Common types for convenience (full type library available in feax.types)
from .types import Array, TensorMap, MassMap, SurfaceMap

__all__ = [
    'Problem', 'InternalVars', 
    'get_J', 'get_res', 'create_J_bc_function', 'create_res_bc_function',
    'Mesh', 'DirichletBC', 'newton_solve', 'SolverOptions', 'create_solver',
    'zero_like_initial_guess', 'DirichletBCSpec', 'DirichletBCConfig', 'dirichlet_bc_config',
    'apply_boundary_to_J', 'apply_boundary_to_res',
    'Array', 'TensorMap', 'MassMap', 'SurfaceMap'
]