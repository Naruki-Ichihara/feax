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

def create_affine_displacement_solver(
    problem: Any,
    bc: Any,
    P: np.ndarray,
    epsilon_macro: np.ndarray,
    mesh: Any,
    solver_options: SolverOptions = None
) -> Callable[[Any, np.ndarray], np.ndarray]:
    """Create an affine displacement solver for LINEAR elasticity homogenization.

    This solver computes the affine displacement problem for linear elasticity:
    1. Solves: K @ u_fluctuation = -K @ u_macro  (in reduced space)
    2. Returns: u_total = u_fluctuation + u_macro
    3. Fully differentiable via JAX's implicit differentiation

    **Note**: This solver is for LINEAR problems only (constant stiffness K).
    For nonlinear problems with periodic BCs, use create_solver(problem, bc, P=P) directly.
    Homogenization (computing C_hom) only makes sense for linear elasticity where
    the stiffness tensor is constant.

    Args:
        problem: FEAX Problem instance (must be linear elasticity)
        bc: Dirichlet boundary conditions (typically empty for periodic problems)
        P (np.ndarray): Prolongation matrix from periodic boundary conditions
        epsilon_macro (np.ndarray): Macroscopic strain tensor of shape (3, 3)
        mesh: FEAX mesh object for computing macro displacement field
        solver_options (SolverOptions, optional): Linear solver configuration

    Returns:
        Callable: Differentiable solver (internal_vars, initial_guess) -> total displacement

    Example:
        >>> pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
        >>> P = prolongation_matrix(pbc, mesh, vec=3)
        >>> epsilon_macro = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> solver = create_affine_displacement_solver(problem, bc, P, epsilon_macro, mesh)
        >>> u_total = solver(internal_vars, initial_guess)
    """
    import jax
    from feax.assembler import create_J_bc_function

    if solver_options is None:
        solver_options = SolverOptions(
            linear_solver="cg",
            linear_solver_tol=1e-10,
            linear_solver_maxiter=10000
        )

    def affine_solver(internal_vars: Any, initial_guess: np.ndarray) -> np.ndarray:
        """Linear affine displacement solver.

        Solves: K @ u_fluctuation = -K @ u_macro  (in reduced space)
        Returns: u_total = u_fluctuation + u_macro
        """
        # Compute macro displacement field
        u_macro = create_macro_displacement_field(mesh, epsilon_macro)

        # Get Jacobian function
        J_bc_func = create_J_bc_function(problem, bc)

        # Compute Jacobian at u_macro (for linear problems, K is constant)
        J_full = J_bc_func(u_macro, internal_vars)

        # Matrix-free reduced Jacobian operator
        def reduced_matvec(v_reduced):
            v_full = P @ v_reduced
            Jv_full = J_full @ v_full
            Jv_reduced = P.T @ Jv_full
            return Jv_reduced

        # RHS: -P^T @ (J @ u_macro)
        b = -P.T @ (J_full @ u_macro)

        # Solve in reduced space
        x0 = np.zeros(P.shape[1])

        if solver_options.linear_solver == 'cg':
            u_fluct_reduced, _ = jax.scipy.sparse.linalg.cg(
                reduced_matvec, b, x0=x0,
                tol=solver_options.linear_solver_tol,
                maxiter=solver_options.linear_solver_maxiter
            )
        elif solver_options.linear_solver == 'bicgstab':
            u_fluct_reduced, _ = jax.scipy.sparse.linalg.bicgstab(
                reduced_matvec, b, x0=x0,
                tol=solver_options.linear_solver_tol,
                atol=solver_options.linear_solver_atol,
                maxiter=solver_options.linear_solver_maxiter
            )
        else:
            u_fluct_reduced, _ = jax.scipy.sparse.linalg.gmres(
                reduced_matvec, b, x0=x0,
                tol=solver_options.linear_solver_tol,
                atol=solver_options.linear_solver_atol,
                maxiter=solver_options.linear_solver_maxiter
            )

        # Total displacement
        u_total = P @ u_fluct_reduced + u_macro

        return u_total

    return affine_solver


def create_homogenization_solver(
    problem: Any,
    bc: Any,
    P: np.ndarray,
    solver_options: SolverOptions,
    mesh: Any,
    dim: int = 3
) -> Callable[[Any], np.ndarray]:
    """Create a computational homogenization solver for LINEAR elasticity.

    This solver runs multiple macroscopic strain cases and computes the volume-averaged
    stress response to determine the homogenized stiffness matrix C_hom.

    For 2D: Runs 3 independent strain cases (11, 22, 12)
    For 3D: Runs 6 independent strain cases (11, 22, 33, 23, 13, 12)

    The homogenized stiffness relates average stress to average strain via:
    <σ> = C_hom : <ε>

    **Important**: This solver is for LINEAR elasticity only (constant C_hom).
    For nonlinear materials, C depends on strain state and homogenization requires
    tangent/secant stiffness computation at each strain level. For nonlinear periodic
    problems, use create_solver(problem, bc, P=P) directly.

    **Differentiability**:
    The homogenization solver is fully differentiable w.r.t. internal_vars (material
    properties) via JAX's built-in implicit differentiation. This enables topology
    optimization with homogenized properties as objectives.

    Args:
        problem: FEAX Problem instance (must be LINEAR elasticity)
        bc: Dirichlet boundary conditions (typically empty for periodic problems)
        P (np.ndarray): Prolongation matrix from periodic boundary conditions
        solver_options (SolverOptions): Linear solver configuration
        mesh: FEAX mesh object for computing macro displacement field
        dim (int): Problem dimension (2 or 3). Defaults to 3.

    Returns:
        Callable: Differentiable function that takes internal_vars and returns homogenized stiffness matrix
                 Shape: (3, 3) for 2D in Voigt notation, (6, 6) for 3D in Voigt notation

    Raises:
        ValueError: If dim is not 2 or 3
        
    Example:
        >>> # 3D homogenization
        >>> from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix
        >>> pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
        >>> P = prolongation_matrix(pbc, mesh, vec=3)
        >>> 
        >>> # Create homogenization solver
        >>> compute_C_hom = create_homogenization_solver(
        >>>     problem, bc, P, solver_options, mesh, dim=3
        >>> )
        >>> 
        >>> # Compute homogenized stiffness
        >>> C_hom = compute_C_hom(internal_vars)
        >>> print(f"Homogenized stiffness matrix: {C_hom.shape}")  # (6, 6) for 3D
    """
    import jax
    from feax import zero_like_initial_guess
    
    if dim not in [2, 3]:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    
    # Define unit strain cases based on dimension as a single array
    if dim == 2:
        # 2D: 3 independent strain components (ε11, ε22, 2*ε12)
        n_cases = 3
        unit_strains = np.array([
            [[1.0, 0.0, 0.0],  # ε11
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0],  # ε22
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0],  # γ12 = 2*ε12
             [0.5, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
    else:  # dim == 3
        # 3D: 6 independent strain components (ε11, ε22, ε33, 2*ε23, 2*ε13, 2*ε12)
        n_cases = 6
        unit_strains = np.array([
            [[1.0, 0.0, 0.0],  # ε11
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0],  # ε22
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0],  # ε33
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0],  # γ23 = 2*ε23
             [0.0, 0.0, 0.5],
             [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.5],  # γ13 = 2*ε13
             [0.0, 0.0, 0.0],
             [0.5, 0.0, 0.0]],
            [[0.0, 0.5, 0.0],  # γ12 = 2*ε12
             [0.5, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
    
    # Get tensor map from problem
    tensor_map = problem.get_tensor_map()
    
    # Precompute initial guess once
    initial_guess = zero_like_initial_guess(problem, bc)
    
    # Total volume for averaging
    total_volume = problem.JxW.sum()
    
    def compute_strain_case_stress(epsilon_macro, internal_vars):
        """Compute average stress for a single strain case - fully JIT-able."""
        # Create and apply affine displacement solver
        solver = create_affine_displacement_solver(
            problem, bc, P, epsilon_macro, mesh, solver_options
        )
        u_total = solver(internal_vars, initial_guess)
        
        # Get solution at cells
        sol_list = problem.unflatten_fn_sol_list(u_total)
        cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
        
        # Prepare internal variables for vmap over cells
        # Handle both cell-based (num_cells,) and quad-based (num_cells, num_quads)
        volume_vars_for_vmap = []

        # Determine number of quad points from JxW
        # JxW can have shape (num_cells, num_quads) or (num_cells, 1, num_quads)
        if problem.JxW.ndim == 3:
            num_quads = problem.JxW.shape[2]
        elif problem.JxW.ndim == 2:
            num_quads = problem.JxW.shape[1]
        else:
            num_quads = 1

        for var in internal_vars.volume_vars:
            if var.ndim == 1:
                # Cell-based: broadcast to (num_cells, num_quads)
                var_quad = np.tile(var[:, None], (1, num_quads))
                volume_vars_for_vmap.append(var_quad)
            else:
                # Already quad-based
                volume_vars_for_vmap.append(var)

        # Compute stress for all cells at once
        def compute_cell_stress(cell_sol_flat, cell_shape_grads, cell_JxW, *cell_internal_vars):
            """Compute volume-weighted stress for one cell."""
            cell_sol = problem.unflatten_fn_dof(cell_sol_flat)[0]
            vec = problem.fes[0].vec

            # Compute u_grads at all quad points
            u_grads = np.sum(
                cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :],
                axis=1
            )

            # Apply tensor_map at all quad points
            stresses = jax.vmap(tensor_map)(u_grads, *cell_internal_vars)

            # Weight by quadrature weights
            if cell_JxW.ndim == 2:
                weights = cell_JxW[0, :]
            else:
                weights = cell_JxW

            weighted_stress = stresses * weights[:, None, None]
            return np.sum(weighted_stress, axis=0)

        # Vectorize over all cells
        stress_per_cell = jax.vmap(compute_cell_stress)(
            cells_sol_flat,
            problem.shape_grads,
            problem.JxW,
            *volume_vars_for_vmap
        )
        
        # Average stress
        avg_stress = np.sum(stress_per_cell, axis=0) / total_volume
        
        # Pad to 3x3 if 2D
        if problem.dim == 2:
            avg_stress_3d = np.zeros((3, 3))
            avg_stress_3d = avg_stress_3d.at[:2, :2].set(avg_stress)
            avg_stress = avg_stress_3d
        
        # Convert to Voigt notation
        if dim == 2:
            stress_voigt = np.array([
                avg_stress[0, 0],  # σ11
                avg_stress[1, 1],  # σ22
                avg_stress[0, 1],  # σ12
            ])
        else:  # dim == 3
            stress_voigt = np.array([
                avg_stress[0, 0],  # σ11
                avg_stress[1, 1],  # σ22
                avg_stress[2, 2],  # σ33
                avg_stress[1, 2],  # σ23
                avg_stress[0, 2],  # σ13
                avg_stress[0, 1],  # σ12
            ])
        
        return stress_voigt
    
    def compute_homogenized_stiffness(internal_vars: Any) -> np.ndarray:
        """Compute homogenized stiffness matrix using vmapped strain cases.
        
        Args:
            internal_vars: FEAX InternalVars with material properties
            
        Returns:
            np.ndarray: Homogenized stiffness matrix in Voigt notation
        """
        # Use vmap to compute all strain cases in parallel
        compute_all_cases = jax.vmap(
            lambda eps: compute_strain_case_stress(eps, internal_vars)
        )
        
        # Compute all stress responses at once
        stress_responses = compute_all_cases(unit_strains)
        
        # Transpose to get stiffness matrix columns
        # stress_responses shape: (n_cases, n_stress_components)
        # C_hom[i,j] = stress_i for unit_strain_j
        C_hom = stress_responses.T
        
        return C_hom
    
    return compute_homogenized_stiffness


def create_unit_cell_solver(
    problem: Any,
    bc: Any,
    P: np.ndarray, 
    solver_options: SolverOptions = None,
    adjoint_solver_options: SolverOptions = None,
    epsilon_macro: np.ndarray = None,
    mesh: Any = None,
    iter_num: int = 1
) -> Callable[[Any, np.ndarray], np.ndarray]:
    """Create a differentiable unit cell solver for lattice structures with custom VJP.
    
    This solver handles periodic boundary conditions and optional macroscopic strains,
    with full gradient support through the adjoint method. It follows the same pattern
    as FEAX's main create_solver but specialized for unit cell problems.
    
    Args:
        problem: FEAX Problem instance
        bc: Dirichlet boundary conditions (typically empty for periodic problems)
        P (np.ndarray): Prolongation matrix from periodic BCs
        solver_options (SolverOptions, optional): Forward solver configuration
        adjoint_solver_options (SolverOptions, optional): Adjoint solver configuration
        epsilon_macro (np.ndarray, optional): Macroscopic strain tensor. If provided,
                                            adds affine displacement term.
        mesh (optional): FEAX mesh object. Required if epsilon_macro is provided.
        iter_num (int): Number of iterations (default 1 for linear problems)
        
    Returns:
        Callable: Differentiable unit cell solver function with signature
                 (internal_vars, initial_guess) -> solution
        
    Raises:
        ValueError: If epsilon_macro is provided but mesh is None
        
    Example:
        Standard periodic solver (no macro strain):
        >>> solver = create_unit_cell_solver(problem, bc, P, solver_options)
        >>> sol = solver(internal_vars, initial_guess)
        
        With macro strain (affine displacement):
        >>> epsilon = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> solver = create_unit_cell_solver(problem, bc, P, solver_options, 
        >>>                                  epsilon_macro=epsilon, mesh=mesh)
        >>> 
        >>> # Use in optimization with gradients
        >>> def loss_fn(internal_vars):
        >>>     sol = solver(internal_vars, initial_guess)
        >>>     return compute_compliance(sol)
        >>> grad_fn = jax.grad(loss_fn)
    """
    import jax
    from feax.assembler import create_J_bc_function, create_res_bc_function
    from feax import create_solver
    
    # Set default solver options
    if solver_options is None:
        solver_options = SolverOptions(
            linear_solver="cg",
            linear_solver_tol=1e-10,
            linear_solver_maxiter=1000
        )
    
    if adjoint_solver_options is None:
        adjoint_solver_options = SolverOptions(
            linear_solver="bicgstab",
            linear_solver_tol=1e-10,
            linear_solver_maxiter=1000
        )
    
    # If macro strain is provided, use affine displacement solver
    if epsilon_macro is not None:
        if mesh is None:
            raise ValueError("mesh must be provided when epsilon_macro is specified")
        
        # Create macro displacement field once
        macro_term = create_macro_displacement_field(mesh, epsilon_macro)
        
        # Get Jacobian and residual functions
        J_bc_func = create_J_bc_function(problem, bc)
        res_bc_func = create_res_bc_function(problem, bc)
        
        @jax.custom_vjp
        def differentiable_affine_solve(internal_vars, initial_guess):
            """Solve with affine displacement using matrix-free reduced space operations.
            
            Solves: P^T @ J @ P @ u_fluct = -P^T @ (J @ u_macro)
            Returns: u_total = P @ u_fluct + u_macro
            """
            # Macro displacement field
            u_macro_jax = np.array(macro_term)
            
            # Compute Jacobian at initial + macro displacement
            J_full = J_bc_func(initial_guess + u_macro_jax, internal_vars)
            
            # Right-hand side: -P^T @ (J @ u_macro)
            b = -P.T @ (J_full @ u_macro_jax)
            
            # Matrix-free reduced Jacobian operator
            def reduced_matvec(v_reduced):
                v_full = P @ v_reduced          # Expand to full space
                Jv_full = J_full @ v_full       # Apply full Jacobian
                Jv_reduced = P.T @ Jv_full      # Reduce back
                return Jv_reduced
            
            # Initial guess in reduced space
            x0 = np.zeros(P.shape[1])
            
            # Solve in reduced space
            if solver_options.linear_solver == 'cg':
                u_fluctuation, _ = jax.scipy.sparse.linalg.cg(
                    reduced_matvec, b, x0=x0,
                    tol=solver_options.linear_solver_tol,
                    maxiter=solver_options.linear_solver_maxiter
                )
            elif solver_options.linear_solver == 'bicgstab':
                u_fluctuation, _ = jax.scipy.sparse.linalg.bicgstab(
                    reduced_matvec, b, x0=x0,
                    tol=solver_options.linear_solver_tol,
                    atol=solver_options.linear_solver_atol,
                    maxiter=solver_options.linear_solver_maxiter
                )
            else:  # gmres
                u_fluctuation, _ = jax.scipy.sparse.linalg.gmres(
                    reduced_matvec, b, x0=x0,
                    tol=solver_options.linear_solver_tol,
                    atol=solver_options.linear_solver_atol,
                    maxiter=solver_options.linear_solver_maxiter
                )
            
            # Total displacement = prolongated fluctuation + macro
            u_total = P @ u_fluctuation + u_macro_jax
            return u_total
        
        def f_fwd(internal_vars, initial_guess):
            """Forward pass - compute solution and save for backward."""
            sol = differentiable_affine_solve(internal_vars, initial_guess)
            return sol, (internal_vars, sol, macro_term)
        
        def f_bwd(res, v):
            """Backward pass - compute VJP using adjoint method."""
            internal_vars, sol, macro_term = res
            u_macro_jax = np.array(macro_term)
            
            # Compute Jacobian at solution
            J_full = J_bc_func(sol, internal_vars)
            
            # Solve adjoint system in reduced space
            # (P^T @ J^T @ P) @ lambda = P^T @ v
            rhs_reduced = P.T @ v
            
            def adjoint_matvec(lambda_reduced):
                lambda_full = P @ lambda_reduced
                Jt_lambda = J_full.T @ lambda_full
                return P.T @ Jt_lambda
            
            # Solve adjoint equation
            if adjoint_solver_options.linear_solver == 'cg':
                adjoint_reduced, _ = jax.scipy.sparse.linalg.cg(
                    adjoint_matvec, rhs_reduced,
                    tol=adjoint_solver_options.linear_solver_tol,
                    maxiter=adjoint_solver_options.linear_solver_maxiter
                )
            elif adjoint_solver_options.linear_solver == 'bicgstab':
                adjoint_reduced, _ = jax.scipy.sparse.linalg.bicgstab(
                    adjoint_matvec, rhs_reduced,
                    tol=adjoint_solver_options.linear_solver_tol,
                    atol=adjoint_solver_options.linear_solver_atol,
                    maxiter=adjoint_solver_options.linear_solver_maxiter
                )
            else:
                adjoint_reduced, _ = jax.scipy.sparse.linalg.gmres(
                    adjoint_matvec, rhs_reduced,
                    tol=adjoint_solver_options.linear_solver_tol,
                    atol=adjoint_solver_options.linear_solver_atol,
                    maxiter=adjoint_solver_options.linear_solver_maxiter
                )
            
            # Expand adjoint to full space
            adjoint_full = P @ adjoint_reduced
            
            # Compute VJP w.r.t. internal_vars using constraint function approach
            def constraint_fn(dofs, internal_vars):
                # For affine problem: res = J(u) @ (u - u_macro) + J(u_macro) @ u_macro
                # We need the derivative of the residual w.r.t. internal_vars
                return res_bc_func(dofs, internal_vars)
            
            def constraint_fn_sol_to_sol(sol_list, internal_vars):
                dofs = jax.flatten_util.ravel_pytree(sol_list)[0] if hasattr(jax.flatten_util, 'ravel_pytree') else sol_list[0].flatten()
                con_vec = constraint_fn(dofs, internal_vars)
                return problem.unflatten_fn_sol_list(con_vec) if hasattr(problem, 'unflatten_fn_sol_list') else [con_vec]
            
            def get_partial_params_c_fn(sol_list):
                def partial_params_c_fn(internal_vars):
                    return constraint_fn_sol_to_sol(sol_list, internal_vars)
                return partial_params_c_fn
            
            def get_vjp_contraint_fn_params(internal_vars, sol_list):
                partial_c_fn = get_partial_params_c_fn(sol_list)
                def vjp_linear_fn(v_list):
                    _, f_vjp = jax.vjp(partial_c_fn, internal_vars)
                    val, = f_vjp(v_list)
                    return val
                return vjp_linear_fn
            
            sol_list = problem.unflatten_fn_sol_list(sol) if hasattr(problem, 'unflatten_fn_sol_list') else [sol]
            vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, sol_list)
            adjoint_list = problem.unflatten_fn_sol_list(adjoint_full) if hasattr(problem, 'unflatten_fn_sol_list') else [adjoint_full]
            vjp_result = vjp_linear_fn(adjoint_list)
            vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)
            
            return (vjp_result, None)  # No gradient w.r.t. initial_guess
        
        differentiable_affine_solve.defvjp(f_fwd, f_bwd)
        return differentiable_affine_solve
    
    else:
        # Standard reduced solver without macro term
        # Use FEAX's built-in create_solver with prolongation matrix
        return create_solver(problem, bc, solver_options, adjoint_solver_options, iter_num, P)