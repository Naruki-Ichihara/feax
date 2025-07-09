"""
Pure functional boundary condition application module for feax.

This module provides JAX-compatible functions for applying Dirichlet boundary conditions
to sparse systems. Based on the "row elimination" method from jax-fem.
"""

import jax
import jax.numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, NamedTuple
from jax.experimental.sparse import BCOO


@dataclass
class DirichletBC:
    """JAX-compatible dataclass for Dirichlet boundary condition information."""
    node_indices: np.ndarray  # Node indices where BC is applied
    vec_indices: np.ndarray   # Vector component indices (for multi-component fields)
    values: np.ndarray        # BC values
    
    def __post_init__(self):
        """Ensure all fields are JAX arrays."""
        self.node_indices = np.asarray(self.node_indices)
        self.vec_indices = np.asarray(self.vec_indices)
        self.values = np.asarray(self.values)


class BCInfo(NamedTuple):
    """Container for multiple boundary conditions."""
    bcs: Tuple[DirichletBC, ...]  # Tuple of DirichletBC objects
    vec_dim: int                  # Vector dimension (e.g., 3 for 3D displacement)
    num_nodes: int                # Total number of nodes
    

# Register BCInfo as a PyTree for JAX compatibility
def _bcinfo_tree_flatten(bcinfo):
    """Flatten BCInfo into children and aux_data."""
    # Extract all BC data as children
    children = []
    bc_data = []
    
    for bc in bcinfo.bcs:
        children.extend([bc.node_indices, bc.vec_indices, bc.values])
        bc_data.append((len(bc.node_indices), len(bc.vec_indices), len(bc.values)))
    
    aux_data = (bc_data, bcinfo.vec_dim, bcinfo.num_nodes)
    return children, aux_data


def _bcinfo_tree_unflatten(aux_data, children):
    """Unflatten BCInfo from children and aux_data."""
    bc_data, vec_dim, num_nodes = aux_data
    
    bcs = []
    child_idx = 0
    
    for node_len, vec_len, val_len in bc_data:
        node_indices = children[child_idx]
        vec_indices = children[child_idx + 1]
        values = children[child_idx + 2]
        
        bcs.append(DirichletBC(node_indices, vec_indices, values))
        child_idx += 3
    
    return BCInfo(tuple(bcs), vec_dim, num_nodes)


jax.tree_util.register_pytree_node(
    BCInfo,
    _bcinfo_tree_flatten,
    _bcinfo_tree_unflatten
)


def apply_dirichletBC(A: BCOO, b: np.ndarray, bc_info: BCInfo) -> Tuple[BCOO, np.ndarray]:
    """
    Apply Dirichlet boundary conditions to a sparse linear system using row elimination.
    
    This function modifies the sparse matrix A and RHS vector b to enforce Dirichlet
    boundary conditions. The method zeros out rows corresponding to constrained DOFs
    and sets the diagonal to 1, while modifying the RHS to the prescribed values.
    
    Parameters
    ----------
    A : BCOO
        Sparse coefficient matrix in BCOO format
    b : np.ndarray
        Right-hand side vector
    bc_info : BCInfo
        Boundary condition information containing node indices, vector components,
        and prescribed values
        
    Returns
    -------
    A_modified : BCOO
        Modified sparse matrix with BC applied
    b_modified : np.ndarray
        Modified RHS vector with BC applied
        
    Notes
    -----
    The row elimination method works by:
    1. Identifying constrained DOFs from bc_info
    2. Zeroing out rows in A corresponding to constrained DOFs
    3. Setting diagonal entries to 1 for constrained DOFs
    4. Setting RHS entries to prescribed values for constrained DOFs
    
    This approach is mathematically equivalent to:
    res(u) = D*r(u) + (I - D)*u - u_b
    where D is a diagonal matrix with 1s at constrained DOFs and 0s elsewhere.
    """
    # Convert A to dense for modification (only efficient for small systems)
    # For large systems, we would need a more sophisticated approach
    if A.nse > 10000:  # Threshold for large systems
        return _apply_dirichletBC_sparse(A, b, bc_info)
    else:
        return _apply_dirichletBC_dense(A, b, bc_info)


def _apply_dirichletBC_dense(A: BCOO, b: np.ndarray, bc_info: BCInfo) -> Tuple[BCOO, np.ndarray]:
    """Apply BC using dense matrix operations (for small systems)."""
    # Convert to dense for easier manipulation
    A_dense = A.todense()
    b_modified = b.copy()
    
    # Apply boundary conditions
    for bc in bc_info.bcs:
        # Convert node and vector indices to global DOF indices
        dof_indices = bc.node_indices * bc_info.vec_dim + bc.vec_indices
        
        # Zero out rows corresponding to constrained DOFs
        A_dense = A_dense.at[dof_indices, :].set(0.0)
        
        # Set diagonal entries to 1
        A_dense = A_dense.at[dof_indices, dof_indices].set(1.0)
        
        # Set RHS to prescribed values
        b_modified = b_modified.at[dof_indices].set(bc.values)
    
    # Convert back to sparse format with explicit nse
    # Count non-zero entries manually to avoid concretization error
    nse = np.sum(A_dense != 0.0)
    A_modified = BCOO.fromdense(A_dense, nse=nse)
    
    return A_modified, b_modified


def _apply_dirichletBC_sparse(A: BCOO, b: np.ndarray, bc_info: BCInfo) -> Tuple[BCOO, np.ndarray]:
    """Apply BC using sparse matrix operations (for large systems)."""
    # Extract sparse matrix components
    data = A.data.copy()
    indices = A.indices.copy()
    b_modified = b.copy()
    
    # Collect all constrained DOFs
    constrained_dofs = []
    prescribed_values = []
    
    for bc in bc_info.bcs:
        dof_indices = bc.node_indices * bc_info.vec_dim + bc.vec_indices
        constrained_dofs.extend(dof_indices)
        prescribed_values.extend(bc.values)
    
    constrained_dofs = np.array(constrained_dofs)
    prescribed_values = np.array(prescribed_values)
    
    # Create mask for constrained DOFs
    constrained_mask = np.isin(indices[:, 0], constrained_dofs)
    
    # Zero out rows corresponding to constrained DOFs
    data = np.where(constrained_mask, 0.0, data)
    
    # Find diagonal entries for constrained DOFs
    diagonal_mask = (indices[:, 0] == indices[:, 1]) & constrained_mask
    data = np.where(diagonal_mask, 1.0, data)
    
    # Set RHS to prescribed values
    b_modified = b_modified.at[constrained_dofs].set(prescribed_values)
    
    # Create modified sparse matrix
    A_modified = BCOO((data, indices), shape=A.shape)
    
    return A_modified, b_modified


def apply_bc_to_vector(vec: np.ndarray, bc_info: BCInfo) -> np.ndarray:
    """
    Apply boundary conditions to a vector (e.g., for initial conditions).
    
    Parameters
    ----------
    vec : np.ndarray
        Vector to modify
    bc_info : BCInfo
        Boundary condition information
        
    Returns
    -------
    vec_modified : np.ndarray
        Modified vector with BC values applied
    """
    vec_modified = vec.copy()
    
    for bc in bc_info.bcs:
        # Convert node and vector indices to global DOF indices
        dof_indices = bc.node_indices * bc_info.vec_dim + bc.vec_indices
        
        # Set prescribed values
        vec_modified = vec_modified.at[dof_indices].set(bc.values)
    
    return vec_modified


def create_bc_info(node_indices_list: List[np.ndarray], 
                   vec_indices_list: List[np.ndarray],
                   values_list: List[np.ndarray],
                   vec_dim: int,
                   num_nodes: int) -> BCInfo:
    """
    Convenience function to create BCInfo from lists.
    
    Parameters
    ----------
    node_indices_list : List[np.ndarray]
        List of node indices for each BC
    vec_indices_list : List[np.ndarray]
        List of vector component indices for each BC
    values_list : List[np.ndarray]
        List of prescribed values for each BC
    vec_dim : int
        Vector dimension
    num_nodes : int
        Total number of nodes
        
    Returns
    -------
    BCInfo
        Boundary condition information
    """
    bcs = []
    for node_inds, vec_inds, vals in zip(node_indices_list, vec_indices_list, values_list):
        bcs.append(DirichletBC(node_inds, vec_inds, vals))
    
    return BCInfo(tuple(bcs), vec_dim, num_nodes)


def create_fixed_bc(node_indices: np.ndarray, vec_dim: int, num_nodes: int) -> BCInfo:
    """
    Create BC info for fixed (zero displacement) boundary conditions.
    
    Parameters
    ----------
    node_indices : np.ndarray
        Node indices to fix
    vec_dim : int
        Vector dimension
    num_nodes : int
        Total number of nodes
        
    Returns
    -------
    BCInfo
        Boundary condition information for fixed nodes
    """
    # Fix all components of the specified nodes
    all_node_indices = np.repeat(node_indices, vec_dim)
    all_vec_indices = np.tile(np.arange(vec_dim), len(node_indices))
    all_values = np.zeros(len(all_node_indices))
    
    bc = DirichletBC(all_node_indices, all_vec_indices, all_values)
    return BCInfo((bc,), vec_dim, num_nodes)