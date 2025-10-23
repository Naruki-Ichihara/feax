"""
Test symbolic DSL with surface integrals (Neumann boundary conditions).

This example demonstrates surface integrals using ds.
"""

import sys
sys.path.insert(0, '/workspace')

import jax.numpy as np
from feax.experimental.symbolic import TrialFunction, TestFunction, Constant, grad, inner, dx, ds
from feax.experimental import SymbolicProblem
from feax.mesh import rectangle_mesh
from feax.DCboundary import DirichletBC, DirichletBCSpec
from feax.solver import create_solver, SolverOptions
from feax.internal_vars import InternalVars
from feax.utils import zero_like_initial_guess, save_sol


def test_poisson_neumann():
    """Test Poisson with Neumann BC: -∇²u = 0, ∂u/∂n = g on boundary"""

    print("="*60)
    print("Symbolic Poisson with Neumann BC Test")
    print("="*60)

    # Step 1: Define physics symbolically
    print("\n1. Defining symbolic weak form...")
    u = TrialFunction(vec=1, name='u')
    v = TestFunction(vec=1, name='v')
    g = Constant(name='neumann_flux')  # Neumann BC flux

    # Weak form: ∫ ∇u·∇v dx = ∫ g·v ds
    F = inner(grad(u), grad(v)) * dx - g * v * ds

    print(f"   Trial function: u (scalar)")
    print(f"   Test function: v (scalar)")
    print(f"   Weak form: inner(grad(u), grad(v))*dx - g*v*ds")

    # Step 2: Create mesh
    print("\n2. Creating mesh...")
    mesh = rectangle_mesh(Nx=16, Ny=16, domain_x=1.0, domain_y=1.0)
    print(f"   Mesh: {mesh.cells.shape[0]} cells, {mesh.points.shape[0]} nodes")

    # Step 3: Define boundaries
    def left(point):
        return np.isclose(point[0], 0.0, atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1.0, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0.0, atol=1e-5)

    def top(point):
        return np.isclose(point[1], 1.0, atol=1e-5)

    # Location functions for Neumann boundaries (top and bottom)
    location_fns = [top]  # Apply Neumann on top only

    # Step 4: Compile symbolic form
    print("\n3. Compiling symbolic problem...")
    try:
        problem = SymbolicProblem(
            weak_form=F,
            mesh=mesh,
            dim=2,
            ele_type='QUAD4',
            location_fns=location_fns
        )
        print(f"   ✓ Problem compiled successfully")
        print(f"   Total DOFs: {problem.num_total_dofs_all_vars}")
        print(f"   Volume integrals: {len(problem.volume_integrals)}")
        print(f"   Surface integrals: {len(problem.surface_integrals)}")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Boundary conditions
    print("\n4. Setting up boundary conditions...")

    # Dirichlet BCs on left and right
    bc = DirichletBC.from_specs(problem, [
        DirichletBCSpec(left, 0, value=0.0),  # u = 0 on left
        DirichletBCSpec(right, 0, value=1.0), # u = 1 on right
    ])
    print(f"   ✓ Dirichlet BCs: u(0,y) = 0, u(1,y) = 1")
    print(f"   ✓ Neumann BC on top: ∂u/∂n = g")

    # Step 6: Create internal variables
    print("\n5. Creating internal variables...")
    # Neumann flux g - constant value on top boundary
    neumann_flux = InternalVars.create_cell_var(problem, 0.5)  # Flux value
    internal_vars = InternalVars(
        volume_vars=(),
        surface_vars=[(neumann_flux,)]  # Surface vars for each boundary
    )
    print(f"   ✓ Neumann flux: g = 0.5")

    # Step 7: Create solver
    print("\n6. Creating solver...")
    solver_options = SolverOptions(
        tol=1e-8,
        linear_solver="cg",
        use_jacobi_preconditioner=False
    )

    try:
        solver = create_solver(problem, bc, solver_options, iter_num=1)
        print(f"   ✓ Solver created (linear, CG)")
    except Exception as e:
        print(f"   ✗ Solver creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 8: Solve
    print("\n7. Solving...")
    initial_guess = zero_like_initial_guess(problem, bc)

    try:
        solution = solver(internal_vars, initial_guess)
        print(f"   ✓ Solution obtained")
        print(f"   Solution range: [{np.min(solution):.6f}, {np.max(solution):.6f}]")
        print(f"   Solution mean: {np.mean(solution):.6f}")
    except Exception as e:
        print(f"   ✗ Solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 9: Save results
    print("\n8. Saving results...")
    try:
        save_sol(mesh, "/tmp/symbolic_neumann.vtu",
                point_infos=[("solution", solution.reshape(-1, 1))])
        print(f"   ✓ Results saved to /tmp/symbolic_neumann.vtu")
    except Exception as e:
        print(f"   ⚠ Could not save results: {e}")

    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)

    return True


if __name__ == '__main__':
    success = test_poisson_neumann()
    sys.exit(0 if success else 1)
