"""Cahn-Hilliard phase separation using mixed formulation.

Reproduces the FEniCS/DOLFINx Cahn-Hilliard demo:
    https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_cahn-hilliard.html

Operator-split form (two second-order equations):
    ∂c/∂t - ∇·(M ∇μ) = 0
    μ - df/dc + λ ∇²c = 0

Free energy: f(c) = 100 c² (1 - c)²
    => df/dc = 200 c (1 - c)(1 - 2c)

Mixed formulation with two scalar fields (c, μ) avoids C1 elements.
Time discretization: backward Euler (θ = 1).

Note: The FEniCS demo uses a θ-method (Crank-Nicolson, θ = 0.5) on both
the diffusion and reaction terms.  FEAX's get_weak_form does not expose
gradients of internal variables, so a full θ-method on ∇μ is not directly
available.  We use backward Euler, which is unconditionally stable and
gives qualitatively identical spinodal decomposition behaviour.

This example uses ``ImplicitPipeline`` from ``feax.solvers.time_solver``.
"""

import os

import jax
import jax.numpy as np
import jax.random as random

import feax as fe
from feax.solvers.time_solver import ImplicitPipeline, TimeConfig, run

# ── Parameters (matching FEniCS demo) ──────────────────────────────────
lmbda = 1.0e-2   # Surface tension parameter (interface width ~ √λ)
dt = 5.0e-6      # Time step
M = 1.0          # Mobility (implicit, coefficient on ∇μ)
n_steps = 50     # Number of time steps (T = 50 * dt = 2.5e-4)
Nx, Ny = 96, 96  # Mesh resolution


# ── Problem definition ────────────────────────────────────────────────
class CahnHilliard(fe.problem.Problem):
    """Cahn-Hilliard problem with mixed (c, μ) formulation.

    Weak form (at each quadrature point):
      Eq 1 (test fn q for c):
        mass: (c - c_old) / dt
        grad: M ∇μ
      Eq 2 (test fn v for μ):
        mass: μ - df/dc
        grad: -λ ∇c

    where f(c) = 100 c²(1-c)²  =>  df/dc = 200 c(1-c)(1-2c).
    """

    def get_weak_form(self):
        def weak_form(vals, grads, x, c_old):
            c, mu = vals[0], vals[1]                # (1,), (1,)
            grad_c, grad_mu = grads[0], grads[1]    # (1, 2), (1, 2)

            # df/dc = 200 c (1 - c)(1 - 2c)
            dfdc = 200.0 * c * (1.0 - c) * (1.0 - 2.0 * c)

            # Eq 1: concentration evolution  ∂c/∂t = ∇·(M ∇μ)
            R_c_mass = (c - c_old) / dt             # (1,)
            R_c_grad = M * grad_mu                  # (1, 2)

            # Eq 2: chemical potential  μ = df/dc - λ ∇²c
            R_mu_mass = mu - dfdc                   # (1,)
            R_mu_grad = -lmbda * grad_c             # (1, 2)

            return ([R_c_mass, R_mu_mass],
                    [R_c_grad, R_mu_grad])

        return weak_form


# ── Pipeline ──────────────────────────────────────────────────────────
class CahnHilliardPipeline(ImplicitPipeline):

    def build(self, mesh):
        self.mesh = mesh
        num_nodes = mesh.points.shape[0]

        self.problem = CahnHilliard(
            mesh=[mesh, mesh],
            vec=[1, 1],
            dim=2,
            ele_type=['QUAD4', 'QUAD4'],
        )

        # No Dirichlet BCs (natural zero-flux boundaries)
        bc = fe.DirichletBCConfig([]).create_bc(self.problem)

        # Initial c for solver shape detection
        self._c0 = 0.63 + 0.02 * (
            0.5 - random.uniform(random.PRNGKey(42), shape=(num_nodes, 1)))

        solver_opts = fe.IterativeSolverOptions(
            solver='auto',
            tol=1e-10,
            atol=1e-10,
            maxiter=10000,
            use_jacobi_preconditioner=True,
            verbose=True,
        )
        newton_opts = fe.NewtonOptions(
            tol=1e-6,
            rel_tol=1e-8,
            max_iter=25,
            internal_jit=True,
        )
        self.solver = fe.create_solver(
            self.problem, bc,
            solver_options=solver_opts,
            newton_options=newton_opts,
            internal_vars=fe.InternalVars(volume_vars=(self._c0[:, 0],)),
        )

    def initial_state(self):
        mu0 = np.zeros((self.mesh.points.shape[0], 1))
        return jax.flatten_util.ravel_pytree([self._c0, mu0])[0]

    def update_vars(self, state, t, dt_val):
        sol_list = self.problem.unflatten_fn_sol_list(state)
        c_old = sol_list[0][:, 0]   # (num_nodes,)
        return fe.InternalVars(volume_vars=(c_old,))

    def save(self, state, step, t, output_dir):
        sol_list = self.problem.unflatten_fn_sol_list(state)
        vtk_path = os.path.join(output_dir, f'ch_{step:04d}.vtu')
        fe.utils.save_sol(
            mesh=self.mesh,
            sol_file=vtk_path,
            point_infos=[("c", sol_list[0]), ("mu", sol_list[1])],
        )

    def monitor(self, state, step, t):
        c = self.problem.unflatten_fn_sol_list(state)[0]
        return {'c_min': float(c.min()), 'c_max': float(c.max())}


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=1.0, domain_y=1.0)
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'vtk')

    result = run(
        CahnHilliardPipeline(),
        mesh,
        TimeConfig(dt=dt, t_end=n_steps * dt, save_every=10, print_every=1),
        output_dir=data_dir,
    )
