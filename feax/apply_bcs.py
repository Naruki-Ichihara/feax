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
from jax.experimental import sparse


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
    """
    return _apply_dirichletBC_sparse(A, b, bc_info)




def _apply_dirichletBC_sparse(A: BCOO, b: np.ndarray, bc_info: BCInfo) -> Tuple[BCOO, np.ndarray]:
    """Apply BC using sparse matrix operations with proper duplicate handling."""
    # Collect all constrained DOFs
    constrained_dofs = []
    prescribed_values = []
    
    for bc in bc_info.bcs:
        dof_indices = bc.node_indices * bc_info.vec_dim + bc.vec_indices
        constrained_dofs.extend(dof_indices)
        prescribed_values.extend(bc.values)
    
    constrained_dofs = np.array(constrained_dofs)
    prescribed_values = np.array(prescribed_values)
    
    if len(constrained_dofs) == 0:
        return A, b
    
    # Get matrix data
    A_data = A.data
    A_indices = A.indices
    A_shape = A.shape
    
    # Create masks
    row_mask = np.isin(A_indices[:, 0], constrained_dofs)
    col_mask = np.isin(A_indices[:, 1], constrained_dofs)
    
    # Identify diagonal entries at BC DOFs that need to be removed
    is_bc_diag = row_mask & col_mask & (A_indices[:, 0] == A_indices[:, 1])
    
    # Keep non-BC-diagonal entries, but zero out BC rows
    keep_mask = ~is_bc_diag
    A_data_filtered = np.where(keep_mask & row_mask, 0.0, A_data)
    A_data_filtered = np.where(keep_mask, A_data_filtered, 0.0)  # Remove BC diagonals
    A_indices_filtered = A_indices[keep_mask]
    
    # Now add fresh diagonal entries for BC DOFs
    n_constrained = len(constrained_dofs)
    diag_indices = np.stack([constrained_dofs, constrained_dofs], axis=1)
    diag_data = np.ones(n_constrained)
    
    # Combine
    combined_indices = np.concatenate([A_indices_filtered, diag_indices], axis=0)
    combined_data = np.concatenate([A_data_filtered[keep_mask], diag_data], axis=0)
    
    # Create new sparse matrix
    A_modified = BCOO((combined_data, combined_indices), shape=A_shape)
    
    # Set RHS to prescribed values
    b_modified = b.at[constrained_dofs].set(prescribed_values)
    
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