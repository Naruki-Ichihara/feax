"""
Legacy feax interface with internal_vars embedded in Problem class.
This interface is maintained for backward compatibility.

Usage:
    from feax.old.problem import Problem
    from feax.old.assembler import get_J, get_res
"""

# Import legacy components
from .problem import Problem
from .assembler import get_J, get_res

__all__ = ['Problem', 'get_J', 'get_res']