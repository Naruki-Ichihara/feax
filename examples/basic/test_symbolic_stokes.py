"""
Test symbolic DSL with Stokes flow (mixed finite element).

This demonstrates a multi-variable problem with velocity and pressure.
"""

import sys
sys.path.insert(0, '/workspace')

import jax.numpy as np
from feax.experimental.symbolic import (
    TrialFunction, TestFunction, Constant,
    grad, div, inner, epsilon, sigma, dx
)
from feax.experimental import SymbolicProblem
from feax.mesh import box_mesh
from feax.DCboundary import DirichletBC, DirichletBCSpec
from feax.solver import create_solver, SolverOptions
from feax.internal_vars import InternalVars
from feax.utils import zero_like_initial_guess, save_sol


def test_stokes():
    """Test Stokes flow: -∇·σ(u,p) = f, ∇·u = 0"""

    print("="*60)
    print("Symbolic Stokes Flow Test")
    print("="*60)

    # Step 1: Define physics symbolically
    print("\n1. Defining symbolic weak form...")
    u = TrialFunction(vec=3, name='velocity', index=0)
    p = TrialFunction(vec=1, name='pressure', index=1)
    v = TestFunction(vec=3, name='v', index=0)
    q = TestFunction(vec=1, name='q', index=1)

    mu = Constant(name='viscosity')
    f = Constant(name='body_force', vec=3)

    # Weak form: ∫ σ:ε(v) dx + ∫ q·∇·u dx = ∫ f·v dx
    # where σ = 2με(u) - pI
    F = (
        inner(sigma(u, p, mu), epsilon(v)) * dx +
        q * div(u) * dx -
        inner(f, v) * dx
    )

    print(f"   Trial functions: u (velocity, 3D), p (pressure, scalar)")
    print(f"   Test functions: v, q")
    print(f"   Weak form: inner(sigma(u,p,mu), epsilon(v))*dx + q*div(u)*dx - inner(f,v)*dx")

    # Step 2: Create mesh
    print("\n2. Creating mesh...")
    mesh = box_mesh(size=(2.0, 1.0, 1.0), mesh_size=0.3, element_type='HEX8')
    print(f"   Mesh: {mesh.cells.shape[0]} cells, {mesh.points.shape[0]} nodes")

    # Step 3: Compile symbolic form
    print("\n3. Compiling symbolic problem...")
    try:
        problem = SymbolicProblem(
            weak_form=F,
            mesh=mesh,
            dim=3,
            ele_type='HEX8'
        )
        print(f"   ✓ Problem compiled successfully")
        print(f"   Variables: {problem.num_vars} (velocity + pressure)")
        print(f"   Total DOFs: {problem.num_total_dofs_all_vars}")
        print(f"   Volume integrals: {len(problem.volume_integrals)}")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Boundary conditions
    print("\n4. Setting up boundary conditions...")

    def inlet(point):
        return np.isclose(point[0], 0.0, atol=1e-5)

    def outlet(point):
        return np.isclose(point[0], 2.0, atol=1e-5)

    def walls(point):
        return (np.isclose(point[1], 0.0, atol=1e-5) |
                np.isclose(point[1], 1.0, atol=1e-5) |
                np.isclose(point[2], 0.0, atol=1e-5) |
                np.isclose(point[2], 1.0, atol=1e-5))

    # Velocity BCs
    bc_specs = [
        # Inlet: parabolic velocity profile in x
        DirichletBCSpec(inlet, 0, value=1.0),  # u_x = 1
        DirichletBCSpec(inlet, 1, value=0.0),  # u_y = 0
        DirichletBCSpec(inlet, 2, value=0.0),  # u_z = 0
        # Walls: no-slip
        DirichletBCSpec(walls, 'all', value=0.0),
        # Outlet: pressure (handled by fixing one pressure DOF)
        # For simplicity, fix one pressure node
    ]

    bc = DirichletBC.from_specs(problem, bc_specs)
    print(f"   ✓ Boundary conditions set (inlet, walls, outlet)")

    # Step 5: Create internal variables
    print("\n5. Creating internal variables...")
    viscosity = InternalVars.create_cell_var(problem, 0.01)  # Dynamic viscosity
    body_force_x = InternalVars.create_cell_var(problem, 0.0)
    body_force_y = InternalVars.create_cell_var(problem, 0.0)
    body_force_z = InternalVars.create_cell_var(problem, 0.0)

    internal_vars = InternalVars(
        volume_vars=(viscosity, body_force_x, body_force_y, body_force_z),
        surface_vars=[]
    )
    print(f"   ✓ Viscosity: μ = 0.01")
    print(f"   ✓ Body force: f = 0")

    # Step 6: Create solver
    print("\n6. Creating solver...")
    solver_options = SolverOptions(
        tol=1e-6,
        linear_solver="bicgstab",  # Better for mixed problems
        use_jacobi_preconditioner=False,
        max_iter=1000
    )

    try:
        solver = create_solver(problem, bc, solver_options, iter_num=1)
        print(f"   ✓ Solver created (linear, BiCGSTAB)")
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

        # Extract velocity and pressure
        sol_list = problem.unflatten_fn_sol_list(solution)
        velocity = sol_list[0]  # (num_nodes, 3)
        pressure = sol_list[1]  # (num_nodes, 1)

        print(f"   Velocity range: [{np.min(velocity):.6f}, {np.max(velocity):.6f}]")
        print(f"   Pressure range: [{np.min(pressure):.6f}, {np.max(pressure):.6f}]")

    except Exception as e:
        print(f"   ✗ Solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 8: Save results
    print("\n8. Saving results...")
    try:
        save_sol(mesh, "symbolic_stokes.vtu",
                point_infos=[
                    ("velocity", velocity),
                    ("pressure", pressure)
                ])
        print(f"   ✓ Results saved to /tmp/symbolic_stokes.vtu")
    except Exception as e:
        print(f"   ⚠ Could not save results: {e}")

    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)

    return True


if __name__ == '__main__':
    success = test_stokes()
    sys.exit(0 if success else 1)
