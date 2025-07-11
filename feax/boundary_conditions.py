"""
Ultra-simple API for boundary condition handling in FEAX.

This module provides the simplest possible API for applying boundary conditions.
"""

import jax
import jax.numpy as np
from typing import Callable, List, Union, Optional, Dict, Any
from .apply_bcs import apply_dirichletBC, create_bc_info


def prepare_bc_info(mesh, boundary_conditions, vec_dim: int = 1):
    """
    Prepare boundary condition info from mesh and boundary condition specifications.
    This function is NOT JIT-compatible due to mesh dependency.
    
    Args:
        mesh: Mesh object with points attribute
        boundary_conditions: Can be:
            - Single dict: {'location': fn, 'value': val, 'components': [0,1,2]}
            - List of dicts: [{'location': fn1, 'value': val1, 'components': [0]}, ...]
            - String shorthand: 'fixed_left', 'fixed_right', etc.
        vec_dim: Vector dimension (1 for scalar, 3 for 3D vector field)
    
    Returns:
        bc_data: Dict with pre-computed DOF indices and values for JAX-optimized application
    """
    # Handle different input formats
    if isinstance(boundary_conditions, str):
        boundary_conditions = _parse_shorthand_bc(boundary_conditions, mesh)
    elif isinstance(boundary_conditions, dict):
        boundary_conditions = [boundary_conditions]
    
    # Process all boundary conditions
    all_dof_indices = []
    all_prescribed_values = []
    
    for bc in boundary_conditions:
        location_fn = bc['location']
        value = bc['value']
        components = bc.get('components', list(range(vec_dim)))
        
        # Ensure components is a list
        if isinstance(components, int):
            components = [components]
        
        # Handle constant vs function values
        if callable(value):
            value_fn = value
        else:
            value_fn = lambda point: value
        
        for comp in components:
            # Find nodes where boundary condition applies
            node_flags = jax.vmap(location_fn)(mesh.points)
            bc_nodes = np.where(node_flags)[0]
            
            if len(bc_nodes) > 0:
                # Evaluate boundary values
                bc_vals = jax.vmap(value_fn)(mesh.points[bc_nodes])
                
                # Convert to DOF indices
                dof_indices = bc_nodes * vec_dim + comp
                
                all_dof_indices.append(dof_indices)
                all_prescribed_values.append(bc_vals)
    
    # Create pre-computed boundary condition data
    if all_dof_indices:  # Only if we have boundary conditions
        return {
            'dof_indices': np.concatenate(all_dof_indices),
            'prescribed_values': np.concatenate(all_prescribed_values),
            'vec_dim': vec_dim,
            'has_bc': True
        }
    else:
        return {
            'dof_indices': np.array([], dtype=int),
            'prescribed_values': np.array([]),
            'vec_dim': vec_dim,
            'has_bc': False
        }




def apply_bc_sparse_direct(A, b, dof_indices, prescribed_values):
    """
    Apply boundary conditions directly to sparse matrix with proper duplicate handling.
    
    Args:
        A: Sparse matrix (BCOO format)
        b: Right-hand side vector
        dof_indices: DOF indices where boundary conditions apply
        prescribed_values: Prescribed values at those DOFs
    
    Returns:
        A_bc, b_bc: Modified sparse system with boundary conditions applied
    """
    from jax.experimental import sparse
    
    if len(dof_indices) == 0:
        return A, b
    
    # Get matrix data
    A_data = A.data
    A_indices = A.indices
    A_shape = A.shape
    
    # Create masks
    row_mask = np.isin(A_indices[:, 0], dof_indices)
    col_mask = np.isin(A_indices[:, 1], dof_indices)
    
    # Identify diagonal entries at BC DOFs that need to be removed
    is_bc_diag = row_mask & col_mask & (A_indices[:, 0] == A_indices[:, 1])
    
    # Keep non-BC-diagonal entries, but zero out BC rows
    keep_mask = ~is_bc_diag
    A_data_filtered = np.where(keep_mask & row_mask, 0.0, A_data)
    A_data_filtered = np.where(keep_mask, A_data_filtered, 0.0)  # Remove BC diagonals
    A_indices_filtered = A_indices[keep_mask]
    
    # Now add fresh diagonal entries for BC DOFs
    n_constrained = len(dof_indices)
    diag_indices = np.stack([dof_indices, dof_indices], axis=1)
    diag_data = np.ones(n_constrained)
    
    # Combine
    combined_indices = np.concatenate([A_indices_filtered, diag_indices], axis=0)
    combined_data = np.concatenate([A_data_filtered[keep_mask], diag_data], axis=0)
    
    # Create new sparse matrix
    A_bc = sparse.BCOO((combined_data, combined_indices), shape=A_shape)
    
    # Set RHS to prescribed values
    b_bc = b.at[dof_indices].set(prescribed_values)
    
    return A_bc, b_bc


def apply_bc(A, b, bc_data):
    """
    Ultra-simple boundary condition application using pre-computed bc_data.
    This function is fully JAX-compatible with no Python control flow.
    
    Args:
        A: Sparse matrix (LHS)
        b: Right-hand side vector
        bc_data: Pre-computed boundary condition data from prepare_bc_info
    
    Returns:
        A_bc, b_bc: Modified sparse system with boundary conditions applied
    """
    # Always apply the boundary condition function, but with potentially empty arrays
    dof_indices = bc_data['dof_indices']
    prescribed_values = bc_data['prescribed_values']
    
    return apply_bc_sparse_direct(A, b, dof_indices, prescribed_values)


def _parse_shorthand_bc(shorthand: str, mesh):
    """Parse shorthand boundary condition strings."""
    # Extract domain dimensions from mesh bounds
    points = mesh.points
    Lx = np.max(points[:, 0]) - np.min(points[:, 0])
    Ly = np.max(points[:, 1]) - np.min(points[:, 1])
    Lz = np.max(points[:, 2]) - np.min(points[:, 2]) if points.shape[1] > 2 else None
    
    boundary_fns = create_boundary_functions(Lx, Ly, Lz)
    
    shorthand_map = {
        'fixed_left': {'location': boundary_fns['left'], 'value': 0.0},
        'fixed_right': {'location': boundary_fns['right'], 'value': 0.0},
        'fixed_bottom': {'location': boundary_fns['bottom'], 'value': 0.0},
        'fixed_top': {'location': boundary_fns['top'], 'value': 0.0},
    }
    
    if Lz is not None:
        shorthand_map.update({
            'fixed_front': {'location': boundary_fns['front'], 'value': 0.0},
            'fixed_back': {'location': boundary_fns['back'], 'value': 0.0},
        })
    
    return shorthand_map.get(shorthand, {'location': lambda x: False, 'value': 0.0})


# Convenience functions for common patterns
def DirichletBC(location, value=0.0, components=None):
    """Create a Dirichlet boundary condition specification."""
    return {'location': location, 'value': value, 'components': components}


def FixedBC(location, components=None):
    """Create a fixed (zero) boundary condition specification."""
    return DirichletBC(location, 0.0, components)








# Common boundary condition patterns
def create_boundary_functions(Lx: float, Ly: float, Lz: Optional[float] = None, 
                             atol: float = 1e-5):
    """
    Create common boundary location functions for box domains.
    
    Args:
        Lx, Ly, Lz: Domain dimensions
        atol: Tolerance for boundary detection
    
    Returns:
        Dictionary of boundary functions
    """
    functions = {
        'left': lambda point: np.isclose(point[0], 0., atol=atol),
        'right': lambda point: np.isclose(point[0], Lx, atol=atol),
        'bottom': lambda point: np.isclose(point[1], 0., atol=atol),
        'top': lambda point: np.isclose(point[1], Ly, atol=atol),
    }
    
    if Lz is not None:  # 3D case
        functions.update({
            'front': lambda point: np.isclose(point[2], 0., atol=atol),
            'back': lambda point: np.isclose(point[2], Lz, atol=atol),
        })
    
    return functions


def create_cantilever_bc(A, b, mesh, vec_dim: int = 3):
    """
    Create boundary conditions for a cantilever beam problem.
    Assumes beam extends in x-direction, fixed at x=0.
    
    Args:
        A: Sparse matrix (LHS)
        b: Right-hand side vector
        mesh: Mesh object with points attribute
        vec_dim: Vector dimension (should be 3 for 3D elasticity)
    
    Returns:
        A_bc, b_bc: Modified sparse system with cantilever boundary conditions
    """
    # Fixed boundary at x=0 (left end)
    left_boundary = lambda point: np.isclose(point[0], 0., atol=1e-5)
    
    return apply_fixed_bc(A, b, mesh, left_boundary, 
                         components=list(range(vec_dim)), vec_dim=vec_dim)