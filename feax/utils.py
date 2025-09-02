"""
Utility functions for FEAX finite element analysis framework.

This module provides utility functions for file I/O, solution initialization,
and data processing operations commonly needed in finite element analysis.
"""

import jax
import jax.numpy as np
import numpy as onp
import meshio
import os

from feax.mesh import get_meshio_cell_type, Mesh
from feax.DCboundary import DirichletBC
from typing import Optional, List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from feax.problem import Problem


def save_sol(
    mesh: Mesh,
    sol_file: str,
    cell_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
    point_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
) -> None:
    """Save mesh and solution data to VTK format.

    Args:
        mesh: feax mesh object containing nodes and elements
        sol_file: Output file path for VTK file
        cell_infos: List of (name, data) tuples for cell-based data.
            Data shape should be (n_elements, ...) where ... can be:
            - () or (1,) for scalar data
            - (n,) for vector data
            - (3, 3) for tensor data (will be flattened to (9,))
        point_infos: List of (name, data) tuples for point-based data.
            Data shape should be (n_nodes, ...)

    Raises:
        ValueError: If neither cell_infos nor point_infos is provided.
    """
    if cell_infos is None and point_infos is None:
        raise ValueError("At least one of cell_infos or point_infos must be provided.")
    
    # Get meshio cell type from mesh element type
    # We need to infer element type from the mesh structure
    n_nodes_per_element = mesh.cells.shape[1]
    if n_nodes_per_element == 4:
        element_type = 'TET4'
    elif n_nodes_per_element == 10:
        element_type = 'TET10'
    elif n_nodes_per_element == 8:
        element_type = 'HEX8'
    elif n_nodes_per_element == 20:
        element_type = 'HEX20'
    else:
        raise ValueError(f"Unsupported element type with {n_nodes_per_element} nodes per element")
    
    cell_type = get_meshio_cell_type(element_type)
    
    # Create output directory if needed
    sol_dir = os.path.dirname(sol_file)
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
    
    # Convert JAX arrays to numpy for meshio
    nodes_np = onp.array(mesh.points)
    elements_np = onp.array(mesh.cells)
    
    # Create meshio mesh
    out_mesh = meshio.Mesh(
        points=nodes_np,
        cells={cell_type: elements_np}
    )
    
    # Process cell data
    if cell_infos is not None:
        out_mesh.cell_data = {}
        for name, data in cell_infos:
            # Convert to numpy if it's a JAX array
            data = onp.array(data, dtype=onp.float32)
            
            # Validate shape
            if data.shape[0] != mesh.cells.shape[0]:
                raise ValueError(
                    f"Cell data '{name}' has wrong shape: got {data.shape}, "
                    f"expected first dimension = {mesh.cells.shape[0]}"
                )
            
            # Handle different data dimensions
            if data.ndim == 3:
                # Tensor (n_elements, 3, 3) -> flatten to (n_elements, 9)
                data = data.reshape(mesh.cells.shape[0], -1)
            elif data.ndim == 2:
                # Vector (n_elements, n) is OK as is
                pass
            else:
                # Scalar (n_elements,) -> (n_elements, 1)
                data = data.reshape(mesh.cells.shape[0], 1)
            
            out_mesh.cell_data[name] = [data]
    
    # Process point data
    if point_infos is not None:
        out_mesh.point_data = {}
        for name, data in point_infos:
            # Convert to numpy if it's a JAX array
            data = onp.array(data, dtype=onp.float32)
            
            # Validate shape
            if data.shape[0] != mesh.points.shape[0]:
                raise ValueError(
                    f"Point data '{name}' has wrong shape: got {data.shape}, "
                    f"expected first dimension = {mesh.points.shape[0]}"
                )
            
            out_mesh.point_data[name] = data
    
    # Write the mesh
    out_mesh.write(sol_file)


def zero_like_initial_guess(problem: 'Problem', bc: DirichletBC) -> np.ndarray:
    """Create a zero initial guess with boundary condition values set.
    
    This is the standard initial guess for FE problems: zeros everywhere
    except at Dirichlet boundary condition locations where the prescribed
    values are set.
    
    Parameters
    ----------
    problem : Problem
        The FE problem instance containing DOF information
    bc : DirichletBC
        Boundary conditions with rows and values to set
        
    Returns
    -------
    initial_guess : jax.numpy.ndarray
        Initial guess vector of shape (num_total_dofs,) with zeros
        everywhere except BC locations which have prescribed values
        
    Examples
    --------
    >>> from feax.utils import zero_like_initial_guess
    >>> initial_guess = zero_like_initial_guess(problem, bc)
    >>> solution = solver(internal_vars, initial_guess)
    
    For time-dependent problems:
    >>> # First timestep
    >>> solution = solver(internal_vars_t0, zero_like_initial_guess(problem, bc))
    >>> # Subsequent timesteps use previous solution
    >>> for t in timesteps[1:]:
    >>>     solution = solver(internal_vars_t, solution)
    """
    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    return initial_guess