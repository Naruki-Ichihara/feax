"""feax - Finite Element Analysis with JAX

This is the main feax API with modular separation between:
- Problem: FE structure (mesh, elements, quadrature)  
- InternalVars: Material properties and loading parameters

For legacy compatibility, use: from feax.old import Problem, get_J, get_res
"""
import jax
jax.config.update("jax_enable_x64", True)
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Main API
from .problem import Problem
from .internal_vars import InternalVars
from .assembler import get_J, get_res, create_J_bc_function, create_res_bc_function
from .mesh import Mesh
from .DCboundary import DirichletBC, apply_boundary_to_J, apply_boundary_to_res
from .solver import newton_solve, SolverOptions, create_solver, linear_solve, newton_solve_fori, newton_solve_py
from .utils import zero_like_initial_guess
from .bc_spec import DirichletBCSpec, DirichletBCConfig, dirichlet_bc_config

__all__ = [
    'Problem', 'InternalVars', 
    'get_J', 'get_res', 'create_J_bc_function', 'create_res_bc_function',
    'Mesh', 'DirichletBC', 'newton_solve', 'SolverOptions', 'create_solver',
    'zero_like_initial_guess', 'DirichletBCSpec', 'DirichletBCConfig', 'dirichlet_bc_config'
]

# MDMM moved to topopt_toolkit for better organization
# from feax.topopt_toolkit import mdmm