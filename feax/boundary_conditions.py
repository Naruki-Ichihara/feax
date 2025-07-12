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

def _apply_bc_sparse_direct_jit(A, b, dof_indices, prescribed_values):
    """
    JIT-compatible version of apply_bc_sparse_direct.
    
    This version avoids boolean indexing and uses JAX-compatible operations.
    The key insight: we need to filter out BC diagonal entries and add fresh ones,
    just like the original implementation, but in a JIT-compatible way.
    
    Args:
        A: Sparse matrix (BCOO format)
        b: Right-hand side vector
        dof_indices: DOF indices where boundary conditions apply
        prescribed_values: Prescribed values at those DOFs
    
    Returns:
        A_bc, b_bc: Modified sparse system with boundary conditions applied
    """
    from jax.experimental import sparse
    
    # Handle empty BC case
    if dof_indices.shape[0] == 0:
        return A, b
    
    # First, modify RHS using original matrix A
    u_bc = np.zeros_like(b)
    u_bc = u_bc.at[dof_indices].set(prescribed_values)
    b_bc = b - A @ u_bc
    # Set BC DOF values to prescribed values  
    b_bc = b_bc.at[dof_indices].set(prescribed_values)
    
    # Now modify the matrix A
    A_data = A.data
    A_indices = A.indices
    A_shape = A.shape
    
    # Create a lookup array for fast checking if index is in dof_indices
    max_dof = A_shape[0]
    is_bc_dof = np.zeros(max_dof, dtype=bool)
    is_bc_dof = is_bc_dof.at[dof_indices].set(True)
    
    # Check which entries to zero out (BC rows or columns)
    rows = A_indices[:, 0]
    cols = A_indices[:, 1]
    row_is_bc = is_bc_dof[rows]
    col_is_bc = is_bc_dof[cols]
    
    # Identify diagonal entries at BC DOFs that need to be completely removed
    is_diag = rows == cols
    is_bc_diag = row_is_bc & col_is_bc & is_diag
    
    # Create the filtering approach in a JIT-compatible way
    # Instead of boolean indexing, we'll use where to set values to NaN for filtering
    # and then use isfinite to keep only non-NaN values
    
    # Zero out BC rows and columns
    A_data_zeroed = np.where(row_is_bc | col_is_bc, 0.0, A_data)
    
    # Mark BC diagonal entries for removal by setting them to NaN
    A_data_marked = np.where(is_bc_diag, np.nan, A_data_zeroed)
    
    # Filter out NaN entries (this is JIT-compatible)
    valid_mask = np.isfinite(A_data_marked)
    
    # Use a clever trick: create indices for the valid entries
    # We'll build new arrays by concatenating the filtered entries with new diagonal entries
    num_valid = np.sum(valid_mask)
    
    # This is tricky in JIT - let me use a different approach
    # Instead of filtering, let's just set BC diagonals to 1 and rely on BCOO behavior
    # But first zero them out completely
    A_data_modified = np.where(is_bc_diag, 0.0, A_data_zeroed)
    
    # Create new diagonal entries for BC DOFs
    n_constrained = dof_indices.shape[0]
    diag_indices = np.stack([dof_indices, dof_indices], axis=1)
    diag_data = np.ones(n_constrained)
    
    # Combine with original (modified) matrix
    combined_indices = np.concatenate([A_indices, diag_indices], axis=0)
    combined_data = np.concatenate([A_data_modified, diag_data], axis=0)
    
    # Create modified sparse matrix
    A_bc = sparse.BCOO((combined_data, combined_indices), shape=A_shape)
    
    return A_bc, b_bc


def apply_bc(A, b, bc_data):
    """
    JIT-compatible version of apply_bc.
    
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
    
    return _apply_bc_sparse_direct_jit(A, b, dof_indices, prescribed_values)


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