"""
Test symbolic DSL with Poisson equation.

This example demonstrates the complete workflow:
1. Define physics symbolically
2. Compile to FEAX problem
3. Solve with standard solver
"""

import sys
sys.path.insert(0, '/workspace')

import jax.numpy as np
from feax.experimental.symbolic import TrialFunction, TestFunction, Constant, grad, inner, dx
from feax.experimental import SymbolicProblem
from feax.mesh import box_mesh
from feax.DCboundary import DirichletBC, DirichletBCSpec
from feax.solver import create_solver, SolverOptions
from feax.internal_vars import InternalVars
from feax.utils import zero_like_initial_guess, save_sol


def test_poisson():
    """Test Poisson equation: -∇²u = f"""

    print("="*60)
    print("Symbolic Poisson Equation Test")
    print("="*60)

    # Step 1: Define physics symbolically
    print("\n1. Defining symbolic weak form...")
    u = TrialFunction(vec=1, name='u')
    v = TestFunction(vec=1, name='v')
    f = Constant(name='source')

    # Weak form: ∫ ∇u·∇v dx = ∫ f·v dx
    F = inner(grad(u), grad(v)) * dx - f * v * dx

    print(f"   Trial function: u (scalar)")
    print(f"   Test function: v (scalar)")
    print(f"   Weak form: inner(grad(u), grad(v))*dx - f*v*dx")

    # Step 2: Create mesh
    print("\n2. Creating mesh...")
    mesh = box_mesh(size=1.0, mesh_size=0.2, element_type='HEX8')
    print(f"   Mesh: {mesh.cells.shape[0]} cells, {mesh.points.shape[0]} nodes")

    # Step 3: Compile symbolic form to FEAX problem
    print("\n3. Compiling symbolic problem...")
    try:
        problem = SymbolicProblem(
            weak_form=F,
            mesh=mesh,
            dim=3,
            ele_type='HEX8'
        )
        print(f"   ✓ Problem compiled successfully")
        print(f"   Variables: {problem.num_vars}")
        print(f"   Total DOFs: {problem.num_total_dofs_all_vars}")
        print(f"   Volume integrals: {len(problem.volume_integrals)}")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Set up boundary conditions
    print("\n4. Setting up boundary conditions...")

    def left_boundary(point):
        return np.isclose(point[0], 0.0, atol=1e-5)

    def right_boundary(point):
        return np.isclose(point[0], 1.0, atol=1e-5)

    bc = DirichletBC.from_specs(problem, [
        DirichletBCSpec(left_boundary, 0, value=0.0),
        DirichletBCSpec(right_boundary, 0, value=1.0)
    ])
    print(f"   ✓ Boundary conditions set")

    # Step 5: Create internal variables (source term)
    print("\n5. Creating internal variables...")
    source = InternalVars.create_cell_var(problem, 1.0)  # Constant source
    internal_vars = InternalVars(volume_vars=(source,), surface_vars=[])
    print(f"   ✓ Source term: constant = 1.0")

    # Step 6: Create solver
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

    # Step 7: Solve
    print("\n7. Solving...")
    initial_guess = zero_like_initial_guess(problem, bc)

    try:
        solution = solver(internal_vars, initial_guess)
        print(f"   ✓ Solution obtained")
        print(f"   Solution range: [{np.min(solution):.6f}, {np.max(solution):.6f}]")
    except Exception as e:
        print(f"   ✗ Solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 8: Save results
    print("\n8. Saving results...")
    try:
        save_sol(mesh, "/tmp/symbolic_poisson.vtu",
                point_infos=[("solution", solution.reshape(-1, 1))])
        print(f"   ✓ Results saved to /tmp/symbolic_poisson.vtu")
    except Exception as e:
        print(f"   ⚠ Could not save results: {e}")

    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)

    return True


if __name__ == '__main__':
    success = test_poisson()
    sys.exit(0 if success else 1)
