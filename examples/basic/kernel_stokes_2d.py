"""
Legacy kernel-based 2D Stokes flow implementation.
This uses the traditional FEAX kernel approach without symbolic DSL.
"""

import jax
import jax.numpy as np
from feax import Mesh
from feax.problem import Problem
from feax.mesh import rectangle_mesh
from feax import DirichletBC, DirichletBCSpec, SolverOptions, create_solver
from feax.internal_vars import InternalVars
from feax.utils import save_sol, zero_like_initial_guess


class Stokes2D(Problem):
    """2D Stokes flow using universal kernel.

    Variables: [velocity (vec=2), pressure (vec=1)]
    Total DOFs: 3 * num_nodes

    Weak form:
        ∫ 2μ ε(u):ε(v) dΩ - ∫ p div(v) dΩ + ∫ q div(u) dΩ = 0

    where:
        ε(u) = 0.5*(∇u + ∇u^T) - strain tensor
        div(u) = ∂u_x/∂x + ∂u_y/∂y - divergence
    """

    def __init__(self, mesh):
        # Multi-variable problems need list of meshes (same mesh for both variables)
        super().__init__([mesh, mesh], vec=[2, 1], dim=2, ele_type=['QUAD4', 'QUAD4'])

    def get_universal_kernel(self):
        """Universal kernel implementing all terms for velocity-pressure coupling."""

        dim = self.dim
        vec_u = self.vec[0]  # velocity components (2)
        vec_p = self.vec[1]  # pressure components (1)
        num_nodes = self.fes[0].num_nodes

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                  cell_JxW, cell_v_grads_JxW, *internal_vars):
            """
            Args:
                cell_sol_flat: (total_dofs_per_cell,) - flattened solution for ONE cell
                cell_shape_grads: (num_quads, num_nodes, dim) - shape function gradients at quad points
                cell_v_grads_JxW: Integrated gradient terms (not used here, we'll compute directly)
                cell_JxW: (num_quads,) - Jacobian * quadrature weights
                internal_vars: (mu,) - dynamic viscosity (num_quads,)
            """
            mu = internal_vars[0]  # (num_quads,) - viscosity at each quad point

            num_quads = cell_JxW.shape[0]
            shape_vals = self.fes[0].shape_vals  # (num_quads, num_nodes)

            # Extract solution variables (use unflatten_fn_dof)
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            u_sol = cell_sol_list[0]  # (num_nodes, 2) - velocity
            p_sol = cell_sol_list[1]  # (num_nodes, 1) - pressure

            # Extract shape function gradients for each variable
            # cell_shape_grads has shape (num_quads, total_nodes_all_vars, dim)
            # Need to split by variable using num_nodes_cumsum
            start_idx_u = self.num_nodes_cumsum[0]
            end_idx_u = self.num_nodes_cumsum[1]
            start_idx_p = self.num_nodes_cumsum[1]
            end_idx_p = self.num_nodes_cumsum[2]

            shape_grads_u = cell_shape_grads[:, start_idx_u:end_idx_u, :]  # (num_quads, num_nodes, dim)
            shape_grads_p = cell_shape_grads[:, start_idx_p:end_idx_p, :]  # (num_quads, num_nodes, dim)

            # Compute velocity gradient at quadrature points: ∇u[q, i, j] = ∂u_i/∂x_j
            # grad_u[q, comp, dim] = sum_n u_sol[n, comp] * shape_grads_u[q, n, dim]
            grad_u = np.einsum('ni,qnj->qij', u_sol, shape_grads_u)  # (num_quads, 2, 2)

            # Compute strain tensor: ε = 0.5*(∇u + ∇u^T)
            epsilon = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))  # (num_quads, 2, 2)

            # Compute divergence: div(u) = trace(∇u)
            div_u = np.trace(grad_u, axis1=1, axis2=2)  # (num_quads,)

            # Interpolate pressure to quadrature points
            p_quad = np.einsum('n,qn->q', p_sol[:, 0], shape_vals)  # (num_quads,)

            # Total DOFs per cell for output (flattened)
            dofs_u = num_nodes * vec_u
            dofs_p = num_nodes * vec_p
            total_dofs = dofs_u + dofs_p
            val = np.zeros(total_dofs)

            # ========================================
            # Velocity equations (first num_nodes*2 DOFs)
            # ========================================
            # For each velocity DOF (node n, component i):
            # Residual = ∫ [2μ ε(u):∇φ_n^i - p div(φ_n^i)] dΩ

            for n in range(num_nodes):
                for i in range(vec_u):
                    dof_idx = n * vec_u + i

                    # Gradient of test function φ_n^i at quad points
                    # φ_n^i has only component i active, with spatial gradient shape_grads_u[:, n, :]
                    grad_phi = shape_grads_u[:, n, :]  # (num_quads, dim)

                    # Term 1: ∫ 2μ ε(u):∇φ dΩ
                    # ε(u):∇φ = sum_j ε(u)[i,j] * grad_phi[j]
                    stress_phi = np.sum(epsilon[:, i, :] * grad_phi, axis=1)  # (num_quads,)

                    visc_term = np.sum(2.0 * mu * stress_phi * cell_JxW)

                    # Term 2: -∫ p div(φ) dΩ
                    # div(φ_n^i) = ∂φ_n/∂x_i = shape_grads_u[:, n, i]
                    div_phi = shape_grads_u[:, n, i]  # (num_quads,)
                    pressure_term = -np.sum(p_quad * div_phi * cell_JxW)

                    val = val.at[dof_idx].set(visc_term + pressure_term)

            # ========================================
            # Pressure equation (last num_nodes DOFs)
            # ========================================
            # For each pressure DOF (node m):
            # Residual = ∫ ψ_m div(u) dΩ

            for m in range(num_nodes):
                dof_idx = dofs_u + m

                # Pressure test function ψ_m at quad points
                psi = shape_vals[:, m]  # (num_quads,)

                # Integrate: ψ * div(u)
                incomp_term = np.sum(psi * div_u * cell_JxW)

                val = val.at[dof_idx].set(incomp_term)

            return val

        return kernel


def main():
    # Create 2D rectangular mesh
    print("Creating mesh...")
    mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=1.0, domain_y=1.0)
    print(f"Mesh: {mesh.cells.shape[0]} cells, {mesh.points.shape[0]} nodes")

    # Create problem
    problem = Stokes2D(mesh)
    print(f"Problem: vec={problem.vec}, dim={problem.dim}")
    print(f"Total DOFs: {mesh.points.shape[0] * sum(problem.vec)}")

    # Boundary conditions
    def left_wall(point):
        return np.isclose(point[0], 0.0, atol=1e-6)

    def right_wall(point):
        return np.isclose(point[0], 1.0, atol=1e-6)

    def bottom_wall(point):
        return np.isclose(point[1], 0.0, atol=1e-6)

    def top_wall(point):
        return np.isclose(point[1], 1.0, atol=1e-6)

    # Apply BCs
    # Create a special location function that only matches the outlet corner for pressure
    def pressure_ref_point(point):
        return np.isclose(point[0], 1.0, atol=1e-6) & np.isclose(point[1], 0.0, atol=1e-6)

    bc = DirichletBC.from_specs(problem, [
        # Inlet (left): parabolic velocity profile u_x = 4*y*(1-y), u_y = 0
        DirichletBCSpec(left_wall, 0, lambda p: 4.0 * p[1] * (1.0 - p[1])),  # u_x
        DirichletBCSpec(left_wall, 1, 0.0),  # u_y

        # Walls (top/bottom): no-slip
        DirichletBCSpec(bottom_wall, 0, 0.0),
        DirichletBCSpec(bottom_wall, 1, 0.0),
        DirichletBCSpec(top_wall, 0, 0.0),
        DirichletBCSpec(top_wall, 1, 0.0),

        # Outlet (right): only fix u_y, let u_x be free
        DirichletBCSpec(right_wall, 1, 0.0),  # u_y = 0

        # Pressure reference: fix p=0 at outlet corner
        # This will apply component 0 to both velocity and pressure at this point
        # For velocity: u_x=0 at corner (already satisfied by right_wall BC)
        # For pressure: p=0 at corner (this is what we need)
        DirichletBCSpec(pressure_ref_point, 0, 0.0),
    ])

    print(f"Boundary conditions: {len(bc.bc_rows)} DOFs constrained")

    # Internal variables (viscosity)
    mu = InternalVars.create_cell_var(problem, 0.1)
    internal_vars = InternalVars(volume_vars=(mu,))

    # Solver
    solver_options = SolverOptions(tol=1e-8, linear_solver='bicgstab')
    solver = create_solver(problem, bc, solver_options, iter_num=1)

    # Solve
    print("Solving...")
    initial_guess = zero_like_initial_guess(problem, bc)
    solution = solver(internal_vars, initial_guess)

    # Extract velocity and pressure
    num_nodes = mesh.points.shape[0]
    dofs_u = num_nodes * 2

    u_sol = solution[:dofs_u].reshape(-1, 2)
    p_sol = solution[dofs_u:].reshape(-1, 1)

    print(f"Velocity range: [{np.min(u_sol):.3f}, {np.max(u_sol):.3f}]")
    print(f"Pressure range: [{np.min(p_sol):.3f}, {np.max(p_sol):.3f}]")

    # Save
    save_sol(mesh, "kernel_stokes_2d.vtu",
             point_infos=[("velocity", u_sol), ("pressure", p_sol)])
    print("Saved to kernel_stokes_2d.vtu")


if __name__ == "__main__":
    main()
