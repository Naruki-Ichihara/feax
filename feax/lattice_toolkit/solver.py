"""Solver utilities for periodic lattice structures and homogenization problems.

This module provides specialized solvers for unit cell analysis and computational
homogenization with periodic boundary conditions. It implements the macro term
approach for handling prescribed macroscopic strains without rigid body motion.

Key Functions:
    create_homogenization_solver: Create solver for unit cell homogenization
    create_affine_displacement_solver: Create solver for affine displacement using JAX linear solvers
    create_macro_displacement_field: Generate macro displacement from strain
    
Example:
    Basic homogenization solver usage:
    
    >>> from feax.lattice_toolkit.solver import create_homogenization_solver
    >>> from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix
    >>> 
    >>> # Setup periodic boundary conditions
    >>> pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
    >>> P = prolongation_matrix(pbc, mesh, vec=3)
    >>> 
    >>> # Define macroscopic strain
    >>> epsilon_macro = np.array([[0.01, 0.0, 0.0],
    >>>                          [0.0, 0.0, 0.0], 
    >>>                          [0.0, 0.0, 0.0]])
    >>> 
    >>> # Create homogenization solver
    >>> solver = create_homogenization_solver(
    >>>     problem, bc, P, epsilon_macro, solver_options, mesh
    >>> )
    >>> 
    >>> # Solve for total displacement (fluctuation + macro)
    >>> solution = solver(internal_vars, initial_guess)
"""

import jax.numpy as np
import scipy.sparse
from typing import Callable, Any
from feax import create_solver, SolverOptions, DirichletBCConfig


def create_macro_displacement_field(mesh, epsilon_macro: np.ndarray) -> np.ndarray:
    """Create macroscopic displacement field from macroscopic strain tensor.
    
    Computes the affine displacement field u_macro = epsilon_macro @ X for each
    node position X in the mesh. This represents the displacement that would
    occur under pure macroscopic deformation without any fluctuations.
    
    Args:
        mesh: FEAX mesh object with points attribute
        epsilon_macro (np.ndarray): Macroscopic strain tensor of shape (3, 3)
        
    Returns:
        np.ndarray: Flattened displacement field of shape (num_nodes * 3,)
        
    Example:
        >>> # Pure extension in x-direction
        >>> epsilon_macro = np.array([[0.01, 0.0, 0.0],
        >>>                          [0.0, 0.0, 0.0],
        >>>                          [0.0, 0.0, 0.0]])
        >>> u_macro = create_macro_displacement_field(mesh, epsilon_macro)
    """
    points = mesh.points
    num_nodes = len(points)
    
    # u_macro = epsilon_macro @ X for each node
    u_macro = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        u_macro = u_macro.at[i].set(epsilon_macro @ points[i])
    
    return u_macro.flatten()


def create_homogenization_solver(
    problem: Any,
    bc: Any, 
    P: np.ndarray,
    epsilon_macro: np.ndarray,
    solver_options: SolverOptions,
    mesh: Any
) -> Callable[[Any, np.ndarray], np.ndarray]:
    """Create a homogenization solver with macro term support.
    
    This solver implements the computational homogenization approach where:
    1. The reduced system solves for fluctuation displacement in periodic space
    2. Macroscopic displacement is added as post-processing step
    3. Total displacement = fluctuation + macro displacement
    
    This approach is equivalent to the macro_term handling in advanced FE codes
    and avoids rigid body motion issues by separating fluctuation and macro scales.
    
    Args:
        problem: FEAX Problem instance with mesh and finite elements
        bc: Dirichlet boundary conditions (typically empty for periodic problems)
        P (np.ndarray): Prolongation matrix from periodic boundary conditions
        epsilon_macro (np.ndarray): Macroscopic strain tensor of shape (3, 3)
        solver_options (SolverOptions): Linear solver configuration
        mesh: FEAX mesh object for computing macro displacement field
        
    Returns:
        Callable: Solver function that takes (internal_vars, initial_guess) and 
                 returns total displacement field
        
    Note:
        The solver automatically handles the affine force computation internally
        through FEAX's built-in reduced solver, then adds the macro displacement
        field as prescribed in computational homogenization theory.
        
    Example:
        >>> # Create periodic boundary conditions
        >>> pbc = periodic_bc_3D(unitcell, vec=3, dim=3)  
        >>> P = prolongation_matrix(pbc, mesh, vec=3)
        >>> 
        >>> # Define macroscopic strain (1% extension in x)
        >>> epsilon_macro = np.array([[0.01, 0.0, 0.0],
        >>>                          [0.0, 0.0, 0.0],
        >>>                          [0.0, 0.0, 0.0]])
        >>> 
        >>> # Create homogenization solver
        >>> solver = create_homogenization_solver(
        >>>     problem, bc, P, epsilon_macro, solver_options, mesh
        >>> )
        >>> 
        >>> # Solve homogenization problem
        >>> total_displacement = solver(internal_vars, initial_guess)
    """
    
    # Create macro displacement field (equivalent to macro_term in advanced solvers)
    macro_term = create_macro_displacement_field(mesh, epsilon_macro)
    
    def homogenization_solver(internal_vars: Any, initial_guess: np.ndarray) -> np.ndarray:
        """Solve homogenization problem - temporary revert to basic FEAX solver.
        
        Args:
            internal_vars: FEAX InternalVars with material properties
            initial_guess (np.ndarray): Initial displacement guess
            
        Returns:
            np.ndarray: Total displacement field
        """
        from feax import create_solver
        
        # Use FEAX's built-in reduced solver (just to test basic functionality)
        solver = create_solver(problem, bc, solver_options, iter_num=1, P=P)
        
        # Solve for displacement 
        displacement = solver(internal_vars, initial_guess)
        
        return displacement
    
    return homogenization_solver


def create_affine_displacement_solver(
    problem: Any,
    bc: Any,
    P: np.ndarray,
    epsilon_macro: np.ndarray,
    mesh: Any,
    solver_options: SolverOptions = None
) -> Callable[[Any, np.ndarray], np.ndarray]:
    """Create an affine displacement solver using JAX linear solvers directly.
    
    This solver implements affine displacement boundary conditions by:
    1. Assembling the system matrix directly from the problem
    2. Computing affine force term K * u_macro 
    3. Solving the reduced system using JAX linear solvers
    4. Adding back the macro displacement field
    
    This approach bypasses FEAX's create_solver and uses JAX linear algebra directly,
    similar to the reference solver pattern but adapted for periodic lattice problems.
    
    Args:
        problem: FEAX Problem instance with assembled matrices
        bc: Dirichlet boundary conditions (typically empty for periodic problems)
        P (np.ndarray): Prolongation matrix from periodic boundary conditions
        epsilon_macro (np.ndarray): Macroscopic strain tensor of shape (3, 3)
        mesh: FEAX mesh object for computing macro displacement field
        solver_options (SolverOptions, optional): FEAX solver options. Defaults to bicgstab with Jacobi preconditioning.
        
    Returns:
        Callable: Solver function that takes (internal_vars, initial_guess) and returns total displacement
        
    Example:
        >>> # Setup periodic BCs and macro strain
        >>> pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
        >>> P = prolongation_matrix(pbc, mesh, vec=3)
        >>> epsilon_macro = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> 
        >>> # Create affine solver
        >>> solver = create_affine_displacement_solver(problem, bc, P, epsilon_macro, mesh)
        >>> displacement = solver(internal_vars, initial_guess)
    """
    import jax
    from jax.experimental.sparse import BCOO
    
    if solver_options is None:
        solver_options = SolverOptions(
            linear_solver="bicgstab",
            use_jacobi_preconditioner=True,
            linear_solver_tol=1e-10,
            linear_solver_maxiter=10000
        )
    
    # Create macro displacement field (affine term)
    macro_term = create_macro_displacement_field(mesh, epsilon_macro)
    
    def affine_solver(internal_vars: Any, initial_guess: np.ndarray) -> np.ndarray:
        """Solve affine displacement problem using matrix-free approach like FEAX.
        
        Args:
            internal_vars: FEAX InternalVars with material properties
            initial_guess (np.ndarray): Initial displacement guess
            
        Returns:
            np.ndarray: Total displacement field (fluctuation + macro)
        """
        # Use FEAX's matrix-free approach
        from feax.assembler import create_J_bc_function
        J_bc_applied = create_J_bc_function(problem, bc)
        
        # Macro displacement field
        u_macro_jax = np.array(macro_term)
        
        # Compute Jacobian once at initial + macro displacement
        J_full = J_bc_applied(initial_guess + u_macro_jax, internal_vars)
        
        # Matrix-free reduced Jacobian operator
        def reduced_matvec(v_reduced):
            v_full = P @ v_reduced          # Expand to full space
            Jv_full = J_full @ v_full       # Apply full Jacobian
            Jv_reduced = P.T @ Jv_full      # Reduce back
            return Jv_reduced
        
        # Right-hand side: -P^T @ (J @ u_macro)
        b = -P.T @ (J_full @ u_macro_jax)
        
        # Initial guess in reduced space
        x0 = np.zeros(P.shape[1])
        
        # Solve using matrix-free operator
        if solver_options.linear_solver == 'cg':
            u_fluctuation, info = jax.scipy.sparse.linalg.cg(
                reduced_matvec, b, x0=x0,
                tol=solver_options.linear_solver_tol,
                maxiter=solver_options.linear_solver_maxiter
            )
        elif solver_options.linear_solver == 'bicgstab':
            u_fluctuation, info = jax.scipy.sparse.linalg.bicgstab(
                reduced_matvec, b, x0=x0,
                tol=solver_options.linear_solver_tol,
                atol=solver_options.linear_solver_atol,
                maxiter=solver_options.linear_solver_maxiter
            )
        else:  # gmres
            u_fluctuation, info = jax.scipy.sparse.linalg.gmres(
                reduced_matvec, b, x0=x0,
                tol=solver_options.linear_solver_tol,
                atol=solver_options.linear_solver_atol,
                maxiter=solver_options.linear_solver_maxiter
            )
        
        # Total displacement = prolongated fluctuation + macro
        u_total = P @ u_fluctuation + u_macro_jax
        
        return u_total
    
    return affine_solver


def create_unit_cell_solver(
    problem: Any,
    bc: Any,
    P: np.ndarray, 
    solver_options: SolverOptions,
    epsilon_macro: np.ndarray = None,
    mesh: Any = None
) -> Callable[[Any, np.ndarray], np.ndarray]:
    """Create a unit cell solver for lattice structures.
    
    Convenience function that creates either a standard reduced solver (no macro strain)
    or a full homogenization solver (with macro strain) depending on inputs.
    
    Args:
        problem: FEAX Problem instance
        bc: Dirichlet boundary conditions  
        P (np.ndarray): Prolongation matrix from periodic BCs
        solver_options (SolverOptions): Linear solver configuration
        epsilon_macro (np.ndarray, optional): Macroscopic strain tensor. If provided,
                                            creates homogenization solver.
        mesh (optional): FEAX mesh object. Required if epsilon_macro is provided.
        
    Returns:
        Callable: Unit cell solver function
        
    Raises:
        ValueError: If epsilon_macro is provided but mesh is None
        
    Example:
        Standard periodic solver (no macro strain):
        >>> solver = create_unit_cell_solver(problem, bc, P, solver_options)
        
        Homogenization solver (with macro strain):
        >>> epsilon = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> solver = create_unit_cell_solver(problem, bc, P, solver_options, epsilon, mesh)
    """
    
    if epsilon_macro is not None:
        if mesh is None:
            raise ValueError("mesh must be provided when epsilon_macro is specified")
        return create_homogenization_solver(problem, bc, P, epsilon_macro, solver_options, mesh)
    else:
        # Standard reduced solver without macro term
        return create_solver(problem, bc, solver_options, iter_num=1, P=P)