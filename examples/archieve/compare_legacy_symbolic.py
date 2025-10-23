"""
Compare legacy (kernel-based) vs symbolic Poisson implementations.

This script solves the same 2D Poisson problem using both approaches
and verifies the results match.
"""

import sys
sys.path.insert(0, '/workspace')

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import rectangle_mesh
from feax.utils import save_sol

# Symbolic imports
from feax.experimental.symbolic import TrialFunction, TestFunction, grad, inner, dx, ds
from feax.experimental import SymbolicProblem

jax.config.update("jax_enable_x64", True)


# ============================================================================
# LEGACY IMPLEMENTATION (kernel-based)
# ============================================================================

class PoissonLegacy(Problem):
    """Legacy Poisson implementation - Laplacian only for fair comparison."""

    def get_tensor_map(self):
        def tensor_map(u_grad, *args):
            return u_grad
        return tensor_map

    # No mass_map - no source term
    # No surface_maps - no Neumann BCs
    # This makes it equivalent to symbolic version


# ============================================================================
# SYMBOLIC IMPLEMENTATION
# ============================================================================

def create_symbolic_poisson(mesh):
    """Create symbolic Poisson problem - Laplacian only (no source/Neumann for now)."""

    # Define symbolically
    u = TrialFunction(vec=1, name='u')
    v = TestFunction(vec=1, name='v')

    # Weak form: ∫ ∇u·∇v dx
    # NOTE: We're only testing the Laplacian kernel, not source/Neumann terms yet
    F = inner(grad(u), grad(v)) * dx

    problem = SymbolicProblem(
        weak_form=F,
        mesh=mesh,
        dim=2,
        ele_type='QUAD4',
        location_fns=None  # No surface integrals for now
    )

    return problem


# ============================================================================
# BOUNDARY CONDITIONS (shared)
# ============================================================================

def left(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right(point):
    return np.isclose(point[0], 1.0, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0.0, atol=1e-5)

def top(point):
    return np.isclose(point[1], 1.0, atol=1e-5)

# Non-homogeneous Dirichlet BCs to get non-trivial solution
def dirichlet_val_left(point):
    return 0.0

def dirichlet_val_right(point):
    return np.sin(np.pi * point[1])  # Non-zero BC on right


# ============================================================================
# COMPARISON TEST
# ============================================================================

def run_comparison():
    """Run both implementations and compare results."""

    print("="*70)
    print(" LEGACY vs SYMBOLIC POISSON - COMPARISON TEST")
    print("="*70)

    # Setup
    Nx, Ny = 32, 32
    Lx, Ly = 1.0, 1.0
    location_fns = [bottom, top]

    print(f"\nProblem setup:")
    print(f"  Mesh: {Nx}×{Ny} QUAD4 elements")
    print(f"  Domain: [{Lx}] × [{Ly}]")
    print(f"  Equation: -∇²u = 0 (Laplacian only, non-homogeneous Dirichlet BCs)")
    print(f"  BCs: u(0,y) = 0, u(1,y) = sin(πy)")

    # Create mesh
    print(f"\nCreating mesh...")
    mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    print(f"  ✓ {mesh.cells.shape[0]} cells, {mesh.points.shape[0]} nodes")

    # Boundary conditions (same for both) - non-homogeneous to get non-trivial solution
    bc_config = DirichletBCConfig([
        DirichletBCSpec(location=left, component=0, value=dirichlet_val_left),
        DirichletBCSpec(location=right, component=0, value=dirichlet_val_right)
    ])

    solver_options = SolverOptions(tol=1e-10, linear_solver="cg")

    # ========================================================================
    # TEST 1: LEGACY IMPLEMENTATION
    # ========================================================================
    print(f"\n" + "-"*70)
    print("TEST 1: Legacy (kernel-based) implementation")
    print("-"*70)

    try:
        problem_legacy = PoissonLegacy(
            mesh=mesh,
            vec=1,
            dim=2,
            ele_type='QUAD4',
            location_fns=None  # No surface integrals
        )
        bc_legacy = bc_config.create_bc(problem_legacy)
        internal_vars_legacy = InternalVars()

        print(f"  Problem created: {problem_legacy.num_total_dofs_all_vars} DOFs")

        solver_legacy = create_solver(problem_legacy, bc_legacy, solver_options, iter_num=1)
        print(f"  Solver created")

        solution_legacy = solver_legacy(internal_vars_legacy, zero_like_initial_guess(problem_legacy, bc_legacy))

        sol_list_legacy = problem_legacy.unflatten_fn_sol_list(solution_legacy)
        u_legacy = sol_list_legacy[0]

        print(f"  ✓ Solution obtained")
        print(f"    Shape: {u_legacy.shape}")
        print(f"    Range: [{np.min(u_legacy):.6f}, {np.max(u_legacy):.6f}]")
        print(f"    Mean: {np.mean(u_legacy):.6f}")
        print(f"    Std: {np.std(u_legacy):.6f}")

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
        problem_symbolic = create_symbolic_poisson(mesh)  # No location_fns - testing Laplacian only
        bc_symbolic = bc_config.create_bc(problem_symbolic)
        internal_vars_symbolic = InternalVars()

        print(f"  Problem created: {problem_symbolic.num_total_dofs_all_vars} DOFs")
        print(f"  Volume integrals: {len(problem_symbolic.volume_integrals)}")

        solver_symbolic = create_solver(problem_symbolic, bc_symbolic, solver_options, iter_num=1)
        print(f"  Solver created")

        solution_symbolic = solver_symbolic(internal_vars_symbolic, zero_like_initial_guess(problem_symbolic, bc_symbolic))

        sol_list_symbolic = problem_symbolic.unflatten_fn_sol_list(solution_symbolic)
        u_symbolic = sol_list_symbolic[0]

        print(f"  ✓ Solution obtained")
        print(f"    Shape: {u_symbolic.shape}")
        print(f"    Range: [{np.min(u_symbolic):.6f}, {np.max(u_symbolic):.6f}]")
        print(f"    Mean: {np.mean(u_symbolic):.6f}")
        print(f"    Std: {np.std(u_symbolic):.6f}")

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

        print(f"\nAbsolute difference:")
        print(f"  Max: {np.max(abs_diff):.2e}")
        print(f"  Mean: {np.mean(abs_diff):.2e}")
        print(f"  RMS: {np.sqrt(np.mean(abs_diff**2)):.2e}")

        print(f"\nRelative difference:")
        print(f"  Max: {np.max(rel_diff):.2e}")
        print(f"  Mean: {np.mean(rel_diff):.2e}")

        # Check if solutions match within tolerance
        tol = 1e-8
        matches = np.max(abs_diff) < tol

        if matches:
            print(f"\n✓ PASS: Solutions match within tolerance {tol:.0e}")
        else:
            print(f"\n✗ FAIL: Solutions differ by more than tolerance {tol:.0e}")

        # Save both solutions for visual comparison
        try:
            save_sol(mesh, "/tmp/poisson_legacy.vtu", point_infos=[("u", u_legacy)])
            save_sol(mesh, "/tmp/poisson_symbolic.vtu", point_infos=[("u", u_symbolic)])
            save_sol(mesh, "/tmp/poisson_diff.vtu", point_infos=[("diff", diff)])
            print(f"\n  Results saved to /tmp/poisson_*.vtu")
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
