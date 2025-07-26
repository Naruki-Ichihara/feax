import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO


def apply_boundary_to_J(problem, J):
    """Apply Dirichlet boundary conditions to Jacobian matrix J using row elimination.
    
    This is a pure JAX implementation equivalent to get_A in jax-fem/solver.py.
    For each Dirichlet boundary condition, this function zeros out the corresponding
    rows and sets the diagonal elements to 1.
    
    Parameters
    ----------
    problem : Problem
        The problem instance containing boundary condition information
    J : jax.experimental.sparse.BCOO
        The sparse Jacobian matrix in BCOO format
        
    Returns
    -------
    J_bc : jax.experimental.sparse.BCOO
        The Jacobian matrix with boundary conditions applied
    """
    # Get the data and indices from the BCOO matrix
    data = J.data
    indices = J.indices
    shape = J.shape
    
    # Collect all boundary condition row indices using vectorized operations
    bc_rows_list = [
        fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
        for ind, fe in enumerate(problem.fes)
        for i in range(len(fe.node_inds_list))
    ]
    
    # Handle empty BC case
    if not bc_rows_list:
        return J
        
    bc_rows = np.concatenate(bc_rows_list)
    bc_rows_set = np.unique(bc_rows)
    
    # Get row and column indices from sparse matrix
    row_indices = indices[..., 0]
    col_indices = indices[..., 1]
    
    # Create mask for BC rows using vectorized comparison
    is_bc_row = np.sum(row_indices[:, None] == bc_rows_set[None, :], axis=1) > 0
    
    # Check diagonal entries
    is_diagonal = row_indices == col_indices
    
    # Mask for diagonal entries in BC rows
    is_bc_diagonal = is_bc_row & is_diagonal
    
    # Handle duplicate diagonal entries by keeping only first occurrence
    bc_diag_indices = indices[is_bc_diagonal]
    
    # Create first occurrence mask
    is_first_bc_diag = np.zeros_like(is_bc_diagonal)
    
    if bc_diag_indices.shape[0] > 0:
        unique_bc_diag, first_occurrence_idx = np.unique(bc_diag_indices[:, 0], return_index=True)
        bc_diag_positions = np.where(is_bc_diagonal)[0]
        is_first_bc_diag = is_first_bc_diag.at[bc_diag_positions[first_occurrence_idx]].set(True)
    
    # Create keep mask: non-BC rows or first BC diagonal
    keep_mask = ~is_bc_row | is_first_bc_diag
    
    # Filter indices and data
    filtered_indices = indices[keep_mask]
    filtered_data = data[keep_mask]
    
    # Update BC diagonal values to 1.0
    is_bc_diagonal_filtered = is_first_bc_diag[keep_mask]
    filtered_data = np.where(is_bc_diagonal_filtered, 1.0, filtered_data)
    
    # Find missing diagonal entries
    bc_diag_row_indices = row_indices[is_bc_diagonal]
    missing_diag_rows = np.setdiff1d(bc_rows_set, np.unique(bc_diag_row_indices))
    
    # Create diagonal entries for missing rows
    missing_diag_indices = np.stack([missing_diag_rows, missing_diag_rows], axis=-1)
    missing_diag_data = np.ones_like(missing_diag_rows, dtype=data.dtype)
    
    # Concatenate all data
    all_indices = np.concatenate([filtered_indices, missing_diag_indices], axis=0)
    all_data = np.concatenate([filtered_data, missing_diag_data], axis=0)
    
    # Create final BCOO matrix
    J_bc = BCOO((all_data, all_indices), shape=shape).sort_indices()
    
    return J_bc