import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from dataclasses import dataclass
from jax.tree_util import register_pytree_node


@dataclass(frozen=True)
class DirichletBC:
    """JAX-compatible dataclass for Dirichlet boundary conditions.
    
    This class pre-computes and stores all BC information as static JAX arrays,
    making it suitable for JIT compilation.
    """
    bc_rows: np.ndarray  # All boundary condition row indices
    bc_mask: np.ndarray  # Boolean mask for BC rows (size: total_dofs)
    bc_vals: np.ndarray  # Boundary condition values for each BC row
    total_dofs: int
    
    @staticmethod
    def from_problem(problem):
        """Create DirichletBC from a problem instance.
        
        Extracts boundary condition information from problem.fes and converts
        it to static JAX arrays.
        """
        bc_rows_list = []
        bc_vals_list = []
        
        for ind, fe in enumerate(problem.fes):
            for i in range(len(fe.node_inds_list)):
                bc_indices = fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
                bc_rows_list.append(bc_indices)
                # Extract BC values - expand to match the shape of bc_indices
                bc_values = np.full_like(bc_indices, fe.vals_list[i], dtype=np.float64)
                bc_vals_list.append(bc_values)
        
        if bc_rows_list:
            bc_rows = np.concatenate(bc_rows_list)
            bc_vals = np.concatenate(bc_vals_list)
            
            # Sort by row indices to maintain consistency
            sort_idx = np.argsort(bc_rows)
            bc_rows = bc_rows[sort_idx]
            bc_vals = bc_vals[sort_idx]
            
            # Handle duplicates by keeping first occurrence
            unique_rows, unique_idx = np.unique(bc_rows, return_index=True)
            bc_rows = unique_rows
            bc_vals = bc_vals[unique_idx]
        else:
            bc_rows = np.array([], dtype=np.int32)
            bc_vals = np.array([], dtype=np.float64)
        
        # Create a boolean mask for faster lookup
        # Get total_dofs from problem to ensure consistency
        total_dofs = problem.num_total_dofs_all_vars
        bc_mask = np.zeros(total_dofs, dtype=bool)
        if bc_rows.shape[0] > 0:
            bc_mask = bc_mask.at[bc_rows].set(True)
        
        return DirichletBC(
            bc_rows=bc_rows,
            bc_mask=bc_mask,
            bc_vals=bc_vals,
            total_dofs=total_dofs
        )


# Register DirichletBC as a JAX pytree
def _dirichletbc_flatten(bc):
    """Flatten DirichletBC into a list of arrays and auxiliary data."""
    # Arrays go in the first return value
    arrays = (bc.bc_rows, bc.bc_mask, bc.bc_vals)
    # Static data goes in the second return value
    aux_data = bc.total_dofs
    return arrays, aux_data


def _dirichletbc_unflatten(aux_data, arrays):
    """Reconstruct DirichletBC from flattened representation."""
    bc_rows, bc_mask, bc_vals = arrays
    total_dofs = aux_data
    return DirichletBC(bc_rows=bc_rows, bc_mask=bc_mask, bc_vals=bc_vals, total_dofs=total_dofs)


# Register the pytree
register_pytree_node(
    DirichletBC,
    _dirichletbc_flatten,
    _dirichletbc_unflatten
)

def apply_boundary_to_J(bc: DirichletBC, J: BCOO) -> BCOO:
    """Apply Dirichlet boundary conditions to Jacobian matrix J using row elimination.
    
    Parameters
    ----------
    bc : DirichletBC
        Pre-computed boundary condition information
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
    
    # Get row and column indices from sparse matrix
    row_indices = indices[..., 0]
    
    # Create mask for BC rows using pre-computed bc_mask
    is_bc_row = bc.bc_mask[row_indices]
    
    # The algorithm:
    # 1. Zero out all BC row entries 
    # 2. Add diagonal entries for ALL BC rows with value 1.0
    
    # Step 1: Zero out all BC row entries
    bc_row_mask = is_bc_row
    data_modified = np.where(bc_row_mask, 0.0, data)
    
    # Step 2: Add diagonal entries for ALL BC rows
    # Simple approach that works with JIT: always add all BC diagonal entries
    # This may create duplicates, but most JAX sparse solvers handle this correctly
    
    bc_diag_indices = np.stack([bc.bc_rows, bc.bc_rows], axis=-1)
    bc_diag_data = np.ones_like(bc.bc_rows, dtype=data.dtype)
    
    # Concatenate all data
    all_indices = np.concatenate([indices, bc_diag_indices], axis=0)
    all_data = np.concatenate([data_modified, bc_diag_data], axis=0)
    
    # Create final BCOO matrix
    J_bc = BCOO((all_data, all_indices), shape=shape)
    
    # Skip sorting for large matrices to avoid slow compilation
    # Most JAX sparse solvers can handle unsorted matrices with duplicates
    # The duplicates will be handled correctly by summing during solve
    # (BC diagonal entries: 0 + 1 = 1, which is what we want)
    
    return J_bc

def apply_boundary_to_res(bc: DirichletBC, res_vec: np.ndarray, sol_vec: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Apply Dirichlet boundary conditions to residual vector using row elimination.
    
    This is a JAX-JIT compatible implementation that applies boundary conditions
    to a residual vector: res[bc_dof] = sol[bc_dof] - bc_val * scale
    
    Parameters
    ----------
    bc : DirichletBC
        Pre-computed boundary condition information
    res_vec : np.ndarray
        The residual vector (flattened)
    sol_vec : np.ndarray  
        The solution vector (flattened)
    scale : float, optional
        Scaling factor for boundary condition values, by default 1.0
        
    Returns
    -------
    np.ndarray
        The residual vector with boundary conditions applied
    """
    # Create a copy of the residual vector to modify
    res_modified = res_vec.copy()
    
    # For each boundary condition row:
    # res[bc_row] = sol[bc_row] - bc_val * scale
    # This is equivalent to the reference implementation
    
    # Apply BC: set residual at BC nodes to solution minus BC values
    bc_residual_values = sol_vec[bc.bc_rows] - bc.bc_vals * scale
    res_modified = res_modified.at[bc.bc_rows].set(bc_residual_values)
    
    return res_modified


@jax.jit
def update_J(bc, precomputed_J):
    """Update Jacobian matrix values using new boundary conditions.
    
    This function assumes the sparse matrix structure (indices, shape) remains identical
    to precomputed_J and only updates the data values. This provides maximum optimization
    by avoiding any structural operations and is JIT-compatible.
    
    Parameters
    ----------
    bc : DirichletBC
        Updated boundary condition information
    precomputed_J : BCOO
        Pre-computed Jacobian matrix with identical structure to be maintained
        
    Returns
    -------
    bc_applied_J : BCOO
        Updated Jacobian matrix with boundary conditions applied
    """
    # Reuse exact structure - only update data values
    data = precomputed_J.data
    indices = precomputed_J.indices
    shape = precomputed_J.shape
    
    # Apply boundary conditions by modifying values only
    row_indices = indices[..., 0]
    col_indices = indices[..., 1]
    
    # Simple value updates in data array V - just update specific values
    def update_bc_row(i, data):
        bc_row = bc.bc_rows[i]
        # Zero all entries in this BC row  
        bc_row_mask = (row_indices == bc_row)
        data = np.where(bc_row_mask, 0.0, data)
        # Set diagonal to 1.0 if it exists
        diagonal_mask = bc_row_mask & (col_indices == bc_row)
        data = np.where(diagonal_mask, 1.0, data)
        return data
    
    data = jax.lax.fori_loop(0, len(bc.bc_rows), update_bc_row, data)
    
    # Create updated matrix with same structure
    bc_applied_J = BCOO((data, indices), shape=shape)
    
    return bc_applied_J