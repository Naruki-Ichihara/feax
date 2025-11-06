"""
Experimental features for FEAX.

This module contains experimental features that are functional but may have
API changes in future versions. Use with caution in production code.

Features:
- Symbolic DSL for defining weak forms using mathematical notation
- FEAX Form Compiler: Compiles symbolic expressions to JAX kernels
- Name-based InternalVars for symbolic problems
- Topology optimization toolkit: filtering, response functions, and optimization
"""

from .feax_form_compiler import SymbolicProblem, create_internal_vars_from_dict
from . import topopt_toolkit

__all__ = ['SymbolicProblem', 'create_internal_vars_from_dict', 'topopt_toolkit']
