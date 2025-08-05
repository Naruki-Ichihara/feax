import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO, eye, bcoo_multiply_sparse
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
    def from_bc_info(problem, bc_info):
        """Create DirichletBC from boundary condition info directly.
        
        This method allows creating BC objects with custom BC information
        without modifying the problem's dirichlet_bc_info.
        
        Parameters
        ----------
        problem : Problem
            The problem instance containing mesh and finite element information
        bc_info : list
            Boundary condition information in format [location_fns, vecs, value_fns]
            - location_fns: list of functions that identify boundary nodes
            - vecs: list of vector component indices (0, 1, 2 for x, y, z)
            - value_fns: list of functions that return BC values at each point
        
        Returns
        -------
        DirichletBC
            A new DirichletBC instance with the specified boundary conditions
        
        Examples
        --------
        >>> def left_boundary(point):
        ...     return np.isclose(point[0], 0.0)
        >>> def zero_disp(point):
        ...     return 0.0
        >>> bc_info = [[left_boundary], [0], [zero_disp]]
        >>> bc = DirichletBC.from_bc_info(problem, bc_info)
        """
        if bc_info is None:
            return DirichletBC(
                bc_rows=np.array([], dtype=np.int32),
                bc_mask=np.zeros(problem.num_total_dofs_all_vars, dtype=bool),
                bc_vals=np.array([], dtype=np.float64),
                total_dofs=problem.num_total_dofs_all_vars
            )
        
        location_fns, vecs, value_fns = bc_info
        assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs), \
            "location_fns, vecs, and value_fns must have the same length"
        
        bc_rows_list = []
        bc_vals_list = []
        
        for ind, fe in enumerate(problem.fes):
            for i in range(len(location_fns)):
                # Handle location functions with 1 or 2 arguments
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:
                    location_fn = lambda point, ind_unused: location_fns[i](point)
                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, got {num_args}")
                
                # Find nodes that satisfy the boundary condition
                node_inds = np.argwhere(
                    jax.vmap(location_fn)(fe.mesh.points, np.arange(fe.num_total_nodes))
                ).reshape(-1)
                
                if len(node_inds) > 0:
                    # Create vector component indices
                    vec_inds = np.ones_like(node_inds, dtype=np.int32) * vecs[i]
                    
                    # Calculate DOF indices
                    bc_indices = node_inds * fe.vec + vec_inds + problem.offset[ind]
                    bc_rows_list.append(bc_indices)
                    
                    # Calculate BC values at the nodes
                    values = jax.vmap(value_fns[i])(fe.mesh.points[node_inds].reshape(-1, fe.dim)).reshape(-1)
                    bc_vals_list.append(values)
        
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
    row_indices = indices[:, 0]
    
    # Create mask for BC rows using pre-computed bc_mask
    is_bc_row = bc.bc_mask[row_indices]
    
    # The algorithm:
    # 1. Zero out all BC row entries 
    # 2. Add diagonal entries for ALL BC rows with value 1.0
    
    # Step 1: Zero out all BC row entries
    bc_row_mask = is_bc_row
    data_modified = np.where(bc_row_mask, 0.0, data)
    
    # Step 2: Add diagonal entries for ALL BC rows
    # Direct approach that works with JIT: always add all BC diagonal entries
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


def create_J_bc_updater(problem, bc: DirichletBC):
    """Create a function that updates Jacobian values while preserving BC structure.
    
    This function generator creates an updater that:
    1. Pre-computes the BC structure (which rows to zero, where to add diagonals)
    2. Returns a function that only updates values based on new DOFs
    3. Avoids reapplying row elimination each time
    
    Parameters
    ----------
    problem : Problem
        The problem instance for computing Jacobian
    bc : DirichletBC
        Pre-computed boundary condition information
        
    Returns
    -------
    update_J_bc : function
        Function that takes DOFs and returns updated Jacobian with BC structure
    """
    from feax.assembler import get_J
    
    # Pre-compute structure information
    # Get a reference Jacobian to understand structure
    zero_sol = np.zeros(problem.num_total_dofs_all_vars)
    zero_sol_unflat = problem.unflatten_fn_sol_list(zero_sol)
    J_ref = get_J(problem, zero_sol_unflat)
    
    # Store original indices and shape
    original_indices = J_ref.indices
    original_shape = J_ref.shape
    num_original_entries = original_indices.shape[0]
    
    # Pre-compute BC diagonal indices to add
    bc_diag_indices = np.stack([bc.bc_rows, bc.bc_rows], axis=-1)
    bc_diag_data = np.ones_like(bc.bc_rows, dtype=J_ref.data.dtype)
    
    # Concatenate indices (structure remains fixed)
    all_indices = np.concatenate([original_indices, bc_diag_indices], axis=0)
    
    # Pre-compute mask for BC rows in original data
    row_indices = original_indices[:, 0]
    is_bc_row = bc.bc_mask[row_indices]
    
    def update_J_bc(dofs):
        """Update Jacobian values based on new DOFs while preserving BC structure.
        
        Parameters
        ----------
        dofs : np.ndarray
            Current degrees of freedom vector
            
        Returns
        -------
        J_bc : BCOO
            Updated Jacobian with BC structure preserved
        """
        # Compute new Jacobian values
        sol_unflat = problem.unflatten_fn_sol_list(dofs)
        J_new = get_J(problem, sol_unflat)
        
        # Apply BC structure: zero out BC rows
        data_modified = np.where(is_bc_row, 0.0, J_new.data)
        
        # Concatenate with BC diagonal entries
        all_data = np.concatenate([data_modified, bc_diag_data], axis=0)
        
        # Create updated BCOO matrix with pre-computed structure
        J_bc = BCOO((all_data, all_indices), shape=original_shape)
        
        return J_bc
    
    return update_J_bc


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