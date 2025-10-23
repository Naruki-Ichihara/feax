"""
Experimental features for FEAX.

This module contains experimental features that are functional but may have
API changes in future versions. Use with caution in production code.

Features:
- Symbolic DSL for defining weak forms using mathematical notation
- Name-based InternalVars for symbolic problems
"""

from .symbolic_problem import SymbolicProblem, create_internal_vars_from_dict

__all__ = ['SymbolicProblem', 'create_internal_vars_from_dict']
