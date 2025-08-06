"""
FEAX Topology Optimization Toolkit - Specialized tools for topology optimization.

This module provides topology optimization functionality:
- Universal response functions (compliance, volume)  
- Filtering techniques (Helmholtz, projections)
- Material interpolation schemes (SIMP, sigmoid)
- MDMM (Modified Differential Multiplier Method) optimizer
- JIT-compiled performance optimizations
"""

from .responses import create_compliance_fn, create_volume_fn
from .filter import (
    create_helmholtz_filter,
    create_helmholtz_transform,
    create_box_projection_transform,
    create_sigmoid_transform
)
from . import mdmm

__all__ = [
    # Response functions
    'create_compliance_fn',
    'create_volume_fn',
    # Filtering
    'create_helmholtz_filter',
    'create_helmholtz_transform',
    'create_box_projection_transform',
    'create_sigmoid_transform',
    # Optimization
    'mdmm',
]