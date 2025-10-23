"""
Demonstration of InternalVars usage with symbolic DSL.

This example shows how to properly use material properties and parameters
with the symbolic DSL, enabling optimization and parameter studies.
"""

import sys
sys.path.insert(0, '/workspace')

import jax
import jax.numpy as np
from feax import Problem, InternalVars, SolverOptions, create_solver
from feax.mesh import box_mesh
from feax.DCboundary import DirichletBCConfig, DirichletBCSpec
from feax.experimental.symbolic import (TrialFunction, TestFunction, Constant,
                            grad, epsilon, tr, inner, Identity, dx, ds)
from feax.experimental import SymbolicProblem, create_internal_vars_from_dict
from feax.utils import save_sol

jax.config.update("jax_enable_x64", True)

print("="*70)
print(" Symbolic DSL with InternalVars - Demonstration")
print("="*70)

# ============================================================================
# EXAMPLE 1: Uniform material properties (simplest)
# ============================================================================
print("\n" + "="*70)
print("Example 1: Uniform material properties")
print("="*70)

def example1_uniform():
    """Uniform Young's modulus and Poisson's ratio."""

    # Create mesh
    mesh = box_mesh(size=1.0, mesh_size=0.2, element_type='HEX8')
    print(f"  Mesh: {mesh.cells.shape[0]} elements")

    # Define symbolic problem with material property constants
    u = TrialFunction(vec=3, name='displacement')
    v = TestFunction(vec=3, name='v')
    E = Constant(name='E', vec=1)      # Young's modulus
    nu = Constant(name='nu', vec=1)    # Poisson's ratio

    # Compute Lamé parameters from E and nu
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    # Constitutive relation
    eps_u = epsilon(u)
    sigma_u = 2 * mu * eps_u + lmbda * tr(eps_u) * Identity(3)

    # Weak form
    F = inner(sigma_u, epsilon(v)) * dx

    # Create problem (no surface integrals, so no location_fns needed)
    problem = SymbolicProblem(
        weak_form=F,
        mesh=mesh,
        dim=3,
        ele_type='HEX8',
        gauss_order=2
    )

    # Boundary conditions
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda x: np.isclose(x[0], 0.0, atol=1e-5), 'all', 0.0),
        DirichletBCSpec(lambda x: np.isclose(x[0], 1.0, atol=1e-5), 2, 0.01),  # 1% strain
    ])
    bc = bc_config.create_bc(problem)

    # Create InternalVars with uniform values
    E_value = 210e9      # Steel
    nu_value = 0.3

    # Recommended: Use name-based approach (order doesn't matter!)
    E_array = InternalVars.create_cell_var(problem, E_value)
    nu_array = InternalVars.create_cell_var(problem, nu_value)

    internal_vars = create_internal_vars_from_dict(
        problem,
        volume_dict={'nu': nu_array, 'E': E_array}  # Order doesn't matter!
    )

    print(f"  E = {E_value:.2e} Pa")
    print(f"  nu = {nu_value}")

    # Solve
    solver_options = SolverOptions(linear_solver="cg", tol=1e-8)
    solver = create_solver(problem, bc, solver_options, iter_num=1)

    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)

    solution = solver(internal_vars, initial_guess)
    sol_reshaped = solution.reshape(-1, 3)

    print(f"  Max displacement: {np.max(np.abs(sol_reshaped)):.6e} m")

    return problem, solution

example1_uniform()

# ============================================================================
# EXAMPLE 2: Spatially varying material (graded material)
# ============================================================================
print("\n" + "="*70)
print("Example 2: Functionally graded material (varying E)")
print("="*70)

def example2_graded():
    """Young's modulus varies linearly from 100 GPa to 300 GPa."""

    mesh = box_mesh(size=1.0, mesh_size=0.2, element_type='HEX8')
    print(f"  Mesh: {mesh.cells.shape[0]} elements")

    # Symbolic problem (same as example 1)
    u = TrialFunction(vec=3, name='displacement')
    v = TestFunction(vec=3, name='v')
    E = Constant(name='E', vec=1)
    nu = Constant(name='nu', vec=1)

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    eps_u = epsilon(u)
    sigma_u = 2 * mu * eps_u + lmbda * tr(eps_u) * Identity(3)
    F = inner(sigma_u, epsilon(v)) * dx

    problem = SymbolicProblem(F, mesh, dim=3, ele_type='HEX8', gauss_order=2)

    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda x: np.isclose(x[0], 0.0, atol=1e-5), 'all', 0.0),
        DirichletBCSpec(lambda x: np.isclose(x[0], 1.0, atol=1e-5), 2, 0.01),
    ])
    bc = bc_config.create_bc(problem)

    # Graded material: E varies from 100 GPa to 300 GPa along x
    def E_function(x):
        return 100e9 + 200e9 * x[0]  # Linear variation

    # Use node-based variables for smooth variation (most memory efficient!)
    E_array = InternalVars.create_node_var_from_fn(problem, E_function)
    nu_array = InternalVars.create_node_var(problem, 0.3)

    internal_vars = InternalVars(
        volume_vars=(E_array, nu_array),
        surface_vars=()
    )

    print(f"  E varies: 100 GPa -> 300 GPa along x")
    print(f"  E min: {np.min(E_array):.2e} Pa")
    print(f"  E max: {np.max(E_array):.2e} Pa")

    solver_options = SolverOptions(linear_solver="cg", tol=1e-8)
    solver = create_solver(problem, bc, solver_options, iter_num=1)

    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)

    solution = solver(internal_vars, initial_guess)
    sol_reshaped = solution.reshape(-1, 3)

    print(f"  Max displacement: {np.max(np.abs(sol_reshaped)):.6e} m")

    # Save with material field
    save_sol(mesh, "/tmp/graded_material.vtu",
             point_infos=[("displacement", sol_reshaped),
                          ("E_modulus", E_array.reshape(-1, 1))])
    print(f"  Saved to /tmp/graded_material.vtu")

    return problem, solution

example2_graded()

# ============================================================================
# EXAMPLE 3: Topology optimization setup (cell-based density)
# ============================================================================
print("\n" + "="*70)
print("Example 3: Topology optimization (cell-based density)")
print("="*70)

def example3_topopt():
    """Material density for topology optimization."""

    mesh = box_mesh(size=1.0, mesh_size=0.15, element_type='HEX8')
    print(f"  Mesh: {mesh.cells.shape[0]} elements")

    # Symbolic problem with SIMP penalty
    u = TrialFunction(vec=3, name='displacement')
    v = TestFunction(vec=3, name='v')
    rho = Constant(name='density', vec=1)  # Design variable
    E_base = Constant(name='E_base', vec=1)  # Base material property
    nu = Constant(name='nu', vec=1)

    # SIMP: E_eff = E_base * rho^p (p=3 for penalty)
    p = 3.0
    E_eff = E_base * rho * rho * rho  # rho^3

    mu = E_eff / (2.0 * (1.0 + nu))
    lmbda = E_eff * nu / ((1 + nu) * (1 - 2 * nu))

    eps_u = epsilon(u)
    sigma_u = 2 * mu * eps_u + lmbda * tr(eps_u) * Identity(3)
    F = inner(sigma_u, epsilon(v)) * dx

    problem = SymbolicProblem(F, mesh, dim=3, ele_type='HEX8', gauss_order=2)

    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda x: np.isclose(x[0], 0.0, atol=1e-5), 'all', 0.0),
        DirichletBCSpec(lambda x: np.isclose(x[0], 1.0, atol=1e-5), 2, 0.01),
    ])
    bc = bc_config.create_bc(problem)

    # Cell-based density (one value per element - best for topology optimization!)
    initial_density = 0.5  # Start with 50% material everywhere
    rho_array = InternalVars.create_cell_var(problem, initial_density)
    E_base_array = InternalVars.create_cell_var(problem, 210e9)
    nu_array = InternalVars.create_cell_var(problem, 0.3)

    internal_vars = InternalVars(
        volume_vars=(rho_array, E_base_array, nu_array),
        surface_vars=()
    )

    print(f"  Initial density: {initial_density}")
    print(f"  Density shape: {rho_array.shape} (cell-based)")
    print(f"  E_base: 210 GPa")

    solver_options = SolverOptions(linear_solver="cg", tol=1e-8)
    solver = create_solver(problem, bc, solver_options, iter_num=1)

    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)

    solution = solver(internal_vars, initial_guess)
    sol_reshaped = solution.reshape(-1, 3)

    print(f"  Max displacement: {np.max(np.abs(sol_reshaped)):.6e} m")

    # Demonstrate gradient computation for optimization
    def compliance(internal_vars):
        """Objective function: minimize compliance = u^T K u"""
        sol = solver(internal_vars, initial_guess)
        return np.sum(sol**2)

    # Compute gradient w.r.t. density
    grad_fn = jax.grad(lambda iv: compliance(
        InternalVars(volume_vars=(iv, E_base_array, nu_array), surface_vars=())
    ))

    gradient = grad_fn(rho_array)
    print(f"  Gradient shape: {gradient.shape}")
    print(f"  Gradient range: [{np.min(gradient):.2e}, {np.max(gradient):.2e}]")
    print(f"  ✓ Gradient computation works! Ready for optimization.")

    return problem, solution

example3_topopt()

# ============================================================================
# EXAMPLE 4: Vectorized parameter study with vmap
# ============================================================================
print("\n" + "="*70)
print("Example 4: Parameter study (vmap over Young's modulus)")
print("="*70)

def example4_vmap():
    """Solve for multiple E values in parallel."""

    mesh = box_mesh(size=1.0, mesh_size=0.25, element_type='HEX8')
    print(f"  Mesh: {mesh.cells.shape[0]} elements")

    u = TrialFunction(vec=3, name='displacement')
    v = TestFunction(vec=3, name='v')
    E = Constant(name='E', vec=1)
    nu = Constant(name='nu', vec=1)

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    eps_u = epsilon(u)
    sigma_u = 2 * mu * eps_u + lmbda * tr(eps_u) * Identity(3)
    F = inner(sigma_u, epsilon(v)) * dx

    problem = SymbolicProblem(F, mesh, dim=3, ele_type='HEX8', gauss_order=2)

    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda x: np.isclose(x[0], 0.0, atol=1e-5), 'all', 0.0),
        DirichletBCSpec(lambda x: np.isclose(x[0], 1.0, atol=1e-5), 2, 0.01),
    ])
    bc = bc_config.create_bc(problem)

    # Parameter sweep: different E values
    E_values = np.array([50e9, 100e9, 150e9, 200e9, 250e9])  # 5 cases
    print(f"  Testing {len(E_values)} different E values")

    # Create batched internal vars
    nu_base = InternalVars.create_cell_var(problem, 0.3)

    solver_options = SolverOptions(linear_solver="cg", tol=1e-8)
    solver = create_solver(problem, bc, solver_options, iter_num=1)

    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)

    # Vectorized solve function
    def solve_for_E(E_val):
        E_array = InternalVars.create_cell_var(problem, E_val)
        internal_vars = InternalVars(volume_vars=(E_array, nu_base), surface_vars=())
        return solver(internal_vars, initial_guess)

    # Use vmap to solve all cases in parallel
    print(f"  Solving with vmap...")
    solutions = jax.vmap(solve_for_E)(E_values)

    print(f"  Solutions shape: {solutions.shape} = (num_cases, num_dofs)")

    # Analyze results
    max_displacements = np.max(np.abs(solutions.reshape(len(E_values), -1, 3)), axis=(1, 2))

    print(f"\n  Results:")
    print(f"  {'E (GPa)':>10} | {'Max displacement (mm)':>20}")
    print(f"  {'-'*10}-+-{'-'*20}")
    for E_val, max_disp in zip(E_values, max_displacements):
        print(f"  {E_val/1e9:>10.0f} | {max_disp*1000:>20.6f}")

    print(f"\n  ✓ Vectorized parameter study complete!")

    return solutions

example4_vmap()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print(" SUMMARY: InternalVars Usage Patterns")
print("="*70)
print("""
1. **Uniform properties**: Use create_cell_var(problem, value)
   - Simplest approach
   - One value for entire domain
   - Example: E_array = InternalVars.create_cell_var(problem, 210e9)

2. **Graded materials**: Use create_node_var_from_fn(problem, func)
   - Spatially varying properties
   - Memory efficient (one value per node)
   - Smooth interpolation to quad points
   - Example: E_array = InternalVars.create_node_var_from_fn(problem, lambda x: 100e9 + 200e9*x[0])

3. **Topology optimization**: Use create_cell_var(problem, initial_value)
   - One design variable per element
   - Efficient for optimization
   - Example: rho_array = InternalVars.create_cell_var(problem, 0.5)

4. **Parameter studies**: Use vmap over create_cell_var
   - Solve multiple parameter values in parallel
   - Example: jax.vmap(solve_for_E)(E_values)

**Key points:**
- Order of volume_vars tuple MUST match order of Constant() in weak form
- Use Constant(name='param', vec=1) in symbolic expressions
- Pass values via InternalVars(volume_vars=(...), surface_vars=(...))
- Supports node-based, cell-based, or quad-based formats
- Fully differentiable for optimization!
""")

print("="*70)
print(" ✓ All examples completed successfully!")
print("="*70)
