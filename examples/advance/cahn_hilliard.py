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
"""

import os

import jax
import jax.numpy as np
import jax.random as random

import feax as fe

# ── Parameters (matching FEniCS demo) ──────────────────────────────────
lmbda = 1.0e-2   # Surface tension parameter (interface width ~ √λ)
dt = 5.0e-6      # Time step
M = 1.0          # Mobility (implicit, coefficient on ∇μ)
n_steps = 50     # Number of time steps (T = 50 * dt = 2.5e-4)
Nx, Ny = 96, 96  # Mesh resolution

# ── Mesh ───────────────────────────────────────────────────────────────
mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=1.0, domain_y=1.0)


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


# Two scalar fields (c, μ) on the same mesh
problem = CahnHilliard(
    mesh=[mesh, mesh],
    vec=[1, 1],
    dim=2,
    ele_type=['QUAD4', 'QUAD4'],
)

# No Dirichlet BCs (natural zero-flux boundaries)
bc_config = fe.DCboundary.DirichletBCConfig([])
bc = bc_config.create_bc(problem)

# ── Initial condition (matching FEniCS demo) ───────────────────────────
# c₀ = 0.63 + 0.02 * (0.5 - rand)  =>  uniform in [0.62, 0.64]
# μ₀ = 0
num_nodes = mesh.points.shape[0]
key = random.PRNGKey(42)
c0 = 0.63 + 0.02 * (0.5 - random.uniform(key, shape=(num_nodes, 1)))
mu0 = np.zeros((num_nodes, 1))

# Pack initial solution as flat vector
sol = jax.flatten_util.ravel_pytree([c0, mu0])[0]

# ── Solver setup ──────────────────────────────────────────────────────
# The Cahn-Hilliard system is non-symmetric, use iterative solver.
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

solver = fe.create_solver(
    problem, bc,
    solver_options=solver_opts,
    newton_options=newton_opts,
    internal_vars=fe.InternalVars(volume_vars=(c0[:, 0],)),
)

# ── Output directory ──────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)


def save_step(sol, step):
    sol_list = problem.unflatten_fn_sol_list(sol)
    c_field = sol_list[0]
    mu_field = sol_list[1]
    vtk_path = os.path.join(data_dir, f'vtk/ch_{step:04d}.vtu')
    fe.utils.save_sol(
        mesh=mesh,
        sol_file=vtk_path,
        point_infos=[("c", c_field), ("mu", mu_field)],
    )


# ── Time stepping loop ───────────────────────────────────────────────
print(f"Cahn-Hilliard: {Nx}x{Ny} mesh, lmbda={lmbda}, dt={dt}, {n_steps} steps")
save_step(sol, 0)

for step in range(1, n_steps + 1):
    # Extract current c as internal variable for the backward Euler step
    sol_list = problem.unflatten_fn_sol_list(sol)
    c_old = sol_list[0][:, 0]   # (num_nodes,)

    internal_vars = fe.InternalVars(volume_vars=(c_old,))
    sol = solver(internal_vars, sol)

    c_vals = problem.unflatten_fn_sol_list(sol)[0]
    print(
        f"  Step {step:3d}: "
        f"c in [{float(c_vals.min()):.6f}, {float(c_vals.max()):.6f}]"
    )

    if step % 10 == 0 or step == n_steps:
        save_step(sol, step)

print("Done.")
