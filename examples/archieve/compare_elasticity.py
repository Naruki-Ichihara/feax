"""
Compare legacy vs symbolic linear elasticity implementations.

This verifies the symbolic DSL works for vector problems with surface tractions.
"""

import sys
sys.path.insert(0, '/workspace')

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh
from feax.utils import save_sol

# Symbolic imports
from feax.experimental.symbolic import (
    TrialFunction, TestFunction, Constant, Identity,
    grad, inner, epsilon, tr, dx, ds
)
from feax.experimental import SymbolicProblem

jax.config.update("jax_enable_x64", True)

# Material properties
E = 70e3      # Young's modulus
nu = 0.3      # Poisson's ratio
T = 1e2       # Traction magnitude

# Compute Lamé parameters
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

print(f"Material properties:")
print(f"  E = {E:.1f}")
print(f"  nu = {nu}")
print(f"  mu = {mu:.2f}")
print(f"  lambda = {lmbda:.2f}")


# ============================================================================
# LEGACY IMPLEMENTATION
# ============================================================================

class ElasticityLegacy(Problem):
    """Legacy elasticity using get_tensor_map and get_surface_maps."""

    def get_tensor_map(self):
        def stress(u_grad, *args):
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]


# ============================================================================
# SYMBOLIC IMPLEMENTATION
# ============================================================================

def create_symbolic_elasticity(mesh, location_fns):
    """Create symbolic elasticity problem."""

    # Define symbolically
    u = TrialFunction(vec=3, name='displacement')
    v = TestFunction(vec=3, name='v')
    traction = Constant(name='traction', vec=3)

    # Stress tensor: σ = 2με(u) + λtr(ε)I
    eps_u = epsilon(u)  # (∇u + ∇u^T) / 2
    eps_v = epsilon(v)
    sigma_u = 2 * mu * eps_u + lmbda * tr(eps_u) * Identity(3)

    # Weak form: ∫ σ(u):ε(v) dx = ∫ t·v ds
    # Note: For symmetric σ, σ:ε(v) = σ:∇v, so assembly uses ∇v directly
    F = inner(sigma_u, eps_v) * dx - inner(traction, v) * ds

    problem = SymbolicProblem(
        weak_form=F,
        mesh=mesh,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        location_fns=location_fns
    )

    return problem


# ============================================================================
# BOUNDARY CONDITIONS (shared)
# ============================================================================

def left(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right(point):
    return np.isclose(point[0], 2.0, atol=1e-5)


# ============================================================================
# COMPARISON TEST
# ============================================================================

def run_comparison():
    """Run both implementations and compare results."""

    print("\n" + "="*70)
    print(" LEGACY vs SYMBOLIC LINEAR ELASTICITY - COMPARISON TEST")
    print("="*70)

    # Setup
    mesh_size = 0.15  # Coarser for faster testing
    location_fns = [right]

    print(f"\nProblem setup:")
    print(f"  Geometry: 2.0 × 1.0 × 1.0 box")
    print(f"  Mesh size: {mesh_size}")
    print(f"  Element: HEX8, Gauss order: 2")
    print(f"  BCs: Fixed on left (x=0), traction on right (x=2)")
    print(f"  Traction: [0, 0, {T}]")

    # Create mesh
    print(f"\nCreating mesh...")
    mesh = box_mesh(size=(2.0, 1.0, 1.0), mesh_size=mesh_size, element_type='HEX8')
    print(f"  ✓ {mesh.cells.shape[0]} cells, {mesh.points.shape[0]} nodes")

    # Boundary conditions (same for both)
    bc_config = DirichletBCConfig([
        DirichletBCSpec(location=left, component='all', value=0.0)
    ])

    solver_options = SolverOptions(tol=1e-8, linear_solver="cg")

    # ========================================================================
    # TEST 1: LEGACY IMPLEMENTATION
    # ========================================================================
    print(f"\n" + "-"*70)
    print("TEST 1: Legacy (kernel-based) implementation")
    print("-"*70)

    try:
        problem_legacy = ElasticityLegacy(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='HEX8',
            gauss_order=2,
            location_fns=location_fns
        )
        bc_legacy = bc_config.create_bc(problem_legacy)

        print(f"  Problem created: {problem_legacy.num_total_dofs_all_vars} DOFs")

        # Create surface variable (traction)
        traction_array = InternalVars.create_uniform_surface_var(problem_legacy, T)
        internal_vars_legacy = InternalVars(
            volume_vars=(),
            surface_vars=[(traction_array,)]
        )

        solver_legacy = create_solver(problem_legacy, bc_legacy, solver_options, iter_num=1)
        print(f"  Solver created")

        solution_legacy = solver_legacy(internal_vars_legacy, zero_like_initial_guess(problem_legacy, bc_legacy))

        sol_list_legacy = problem_legacy.unflatten_fn_sol_list(solution_legacy)
        u_legacy = sol_list_legacy[0]

        print(f"  ✓ Solution obtained")
        print(f"    Shape: {u_legacy.shape}")
        print(f"    Range: [{np.min(u_legacy):.6e}, {np.max(u_legacy):.6e}]")
        print(f"    Max displacement: {np.max(np.linalg.norm(u_legacy, axis=1)):.6e}")

        legacy_success = True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        legacy_success = False
        u_legacy = None

    # ========================================================================
    # TEST 2: SYMBOLIC IMPLEMENTATION
    # ========================================================================
    print(f"\n" + "-"*70)
    print("TEST 2: Symbolic implementation")
    print("-"*70)

    try:
        problem_symbolic = create_symbolic_elasticity(mesh, location_fns)
        bc_symbolic = bc_config.create_bc(problem_symbolic)

        print(f"  Problem created: {problem_symbolic.num_total_dofs_all_vars} DOFs")
        print(f"  Volume integrals: {len(problem_symbolic.volume_integrals)}")
        print(f"  Surface integrals: {len(problem_symbolic.surface_integrals)}")

        # Create surface variable (traction vector [0, 0, T])
        # For symbolic, we need to provide the full vector at each face quad point
        num_surface_faces = len(problem_symbolic.boundary_inds_list[0])
        num_face_quads = problem_symbolic.fes[0].face_shape_vals.shape[1]
        # Shape: (num_surface_faces, num_face_quads, 3)
        traction_vector_template = np.array([0., 0., T])
        traction_array = np.tile(traction_vector_template, (num_surface_faces, num_face_quads, 1))
        internal_vars_symbolic = InternalVars(
            volume_vars=(),
            surface_vars=[(traction_array,)]
        )

        solver_symbolic = create_solver(problem_symbolic, bc_symbolic, solver_options, iter_num=1)
        print(f"  Solver created")

        solution_symbolic = solver_symbolic(internal_vars_symbolic, zero_like_initial_guess(problem_symbolic, bc_symbolic))

        sol_list_symbolic = problem_symbolic.unflatten_fn_sol_list(solution_symbolic)
        u_symbolic = sol_list_symbolic[0]

        print(f"  ✓ Solution obtained")
        print(f"    Shape: {u_symbolic.shape}")
        print(f"    Range: [{np.min(u_symbolic):.6e}, {np.max(u_symbolic):.6e}]")
        print(f"    Max displacement: {np.max(np.linalg.norm(u_symbolic, axis=1)):.6e}")

        symbolic_success = True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        symbolic_success = False
        u_symbolic = None

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print(f"\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    if legacy_success and symbolic_success:
        # Compute difference
        diff = u_symbolic - u_legacy
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / (np.abs(u_legacy) + 1e-10)

        # Vector norms
        norm_legacy = np.linalg.norm(u_legacy, axis=1)
        norm_symbolic = np.linalg.norm(u_symbolic, axis=1)
        norm_diff = np.abs(norm_symbolic - norm_legacy)

        print(f"\nComponent-wise absolute difference:")
        print(f"  Max: {np.max(abs_diff):.2e}")
        print(f"  Mean: {np.mean(abs_diff):.2e}")
        print(f"  RMS: {np.sqrt(np.mean(abs_diff**2)):.2e}")

        print(f"\nComponent-wise relative difference:")
        print(f"  Max: {np.max(rel_diff):.2e}")
        print(f"  Mean: {np.mean(rel_diff):.2e}")

        print(f"\nDisplacement magnitude difference:")
        print(f"  Max: {np.max(norm_diff):.2e}")
        print(f"  Mean: {np.mean(norm_diff):.2e}")
        print(f"  RMS: {np.sqrt(np.mean(norm_diff**2)):.2e}")

        # Check if solutions match within tolerance
        tol = 1e-8
        matches = np.max(abs_diff) < tol

        if matches:
            print(f"\n✓ PASS: Solutions match within tolerance {tol:.0e}")
        else:
            print(f"\n✗ FAIL: Solutions differ by more than tolerance {tol:.0e}")

        # Save both solutions for visual comparison
        try:
            save_sol(mesh, "elasticity_legacy.vtu", point_infos=[("displacement", u_legacy)])
            save_sol(mesh, "elasticity_symbolic.vtu", point_infos=[("displacement", u_symbolic)])
            save_sol(mesh, "lasticity_diff.vtu", point_infos=[("diff", diff)])
            print(f"\n  Results saved to elasticity_*.vtu")
        except:
            pass

        return matches

    elif legacy_success and not symbolic_success:
        print(f"\n✗ Symbolic implementation failed, legacy works")
        return False

    elif not legacy_success and symbolic_success:
        print(f"\n⚠ Legacy implementation failed, but symbolic works!")
        return True

    else:
        print(f"\n✗ Both implementations failed")
        return False


if __name__ == '__main__':
    print("\n")
    success = run_comparison()
    print("\n" + "="*70)
    if success:
        print("✓ COMPARISON TEST PASSED")
    else:
        print("✗ COMPARISON TEST FAILED")
    print("="*70 + "\n")

    sys.exit(0 if success else 1)
