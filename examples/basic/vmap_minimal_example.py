"""Minimal vmap example in feax."""
import feax as fe
import jax
import jax.numpy as jnp

# Problem definition
class ElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        E, nu = 70e3, 0.3
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        def stress(u_grad):
            strain = 0.5 * (u_grad + u_grad.T)
            return 2.0 * mu * strain + lam * jnp.trace(strain) * jnp.eye(3)
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
problem = ElasticityProblem(
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
def single_solve(traction_mag):
    surface_var = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, traction_mag)
    internal_vars = fe.internal_vars.InternalVars(surface_vars=[(surface_var,)])
    return solver(internal_vars, fe.utils.zero_like_initial_guess(problem, bc))

# Batch with vmap
solve_vmap = jax.vmap(single_solve)

# Run
traction_values = jnp.array([1.0, 5.0, 10.0])
solutions = solve_vmap(traction_values)
print(f"Input shape: {traction_values.shape}")
print(f"Output shape: {solutions.shape}")
