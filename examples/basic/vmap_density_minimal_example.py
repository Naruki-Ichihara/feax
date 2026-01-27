"""Minimal vmap example with density-based SIMP material interpolation in feax."""
import feax as fe
import jax
import jax.numpy as jnp

# Material parameters
E0 = 70e3  # Base Young's modulus
E_eps = 1e-3  # Minimum stiffness
nu = 0.3  # Poisson's ratio
p = 3  # SIMP penalization parameter
T = 1e2  # Traction magnitude

# Problem definition with SIMP material model
class DensityElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            # SIMP material interpolation: E(rho) = (E0 - E_eps) * rho^p + E_eps
            E = (E0 - E_eps) * rho**p + E_eps
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            strain = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(strain) * jnp.eye(3) + 2.0 * mu * strain
        return stress

    def get_surface_maps(self):
        def traction_map(u_grad, quad_pt, mag):
            return jnp.array([0.0, 0.0, -mag])
        return [traction_map]

# Setup
mesh = fe.mesh.box_mesh((2, 1, 1), mesh_size=0.2)
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(
        location=lambda p: jnp.isclose(p[0], 0, atol=1e-5),
        component='all', value=0.0
    )
])
problem = DensityElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[lambda p: jnp.isclose(p[0], 2, atol=1e-5)]
)
bc = bc_config.create_bc(problem)
solver = fe.solver.create_solver(
    problem, bc,
    fe.solver.SolverOptions(tol=1e-8, linear_solver="cudss"),
    iter_num=1
)

# Single-parameter solve function
def single_solve(density):
    """Solve for a single density value using SIMP material interpolation."""
    rho = fe.internal_vars.InternalVars.create_uniform_volume_var(problem, density)
    traction = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, T)
    internal_vars = fe.internal_vars.InternalVars(
        volume_vars=[rho],
        surface_vars=[(traction,)]
    )
    return solver(internal_vars, fe.utils.zero_like_initial_guess(problem, bc))

# Batch with vmap
solve_vmap = jax.vmap(single_solve)

# Run
density_values = jnp.array([0.1, 0.5, 1.0])
solutions = solve_vmap(density_values)
print(f"Input shape: {density_values.shape}")
print(f"Output shape: {solutions.shape}")
