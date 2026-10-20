"""gene.similarity_order / group_spread and BatchedSolver.operator_signature."""
import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe
from feax import gene


def test_similarity_order_recovers_a_path():
    # designs along a 1-D path, shuffled: the order must be monotone in the path parameter
    t = jnp.linspace(0, 1, 9)
    base = jax.random.normal(jax.random.PRNGKey(0), (50,))
    direction = jax.random.normal(jax.random.PRNGKey(1), (50,))
    F = base[None, :] + t[:, None] * direction[None, :]
    perm = jax.random.permutation(jax.random.PRNGKey(2), 9)
    order = gene.similarity_order(F[perm])
    recovered = perm[order]
    assert bool((jnp.diff(recovered) > 0).all()) or bool((jnp.diff(recovered) < 0).all())
    # grouping consecutive similar designs has a smaller spread than the shuffled order
    assert float(gene.group_spread(F[perm], 3, order=order).mean()) < \
        float(gene.group_spread(F[perm], 3).mean())


class SIMPElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0.0, 0.0, t])]


def test_operator_signature_orders_designs_by_their_operators():
    mesh = fe.StructuredGrid((12, 4, 4), spacing=(0.25, 0.25, 0.25)).to_mesh()
    tip = lambda p: jnp.isclose(p[0], 3.0, atol=1e-6)
    problem = SIMPElasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[tip])
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    n = mesh.points.shape[0]
    x = jnp.asarray(mesh.points[:, 0]) / 3.0
    # designs: solid fraction growing from the clamp, in a shuffled order
    levels = jnp.array([0.2, 0.8, 0.4, 1.0, 0.6])
    rho = jnp.clip(1.0 - 4.0 * (x[None, :] - levels[:, None]), 0.0, 1.0)
    E = gene.simp(rho, 1.0)
    t = fe.TracedParams.create_uniform_surface_var(problem, -1e-2)
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    solver = fe.create_solver(problem, bc, direct, linear=True,
                              traced_params=fe.TracedParams(volume_vars=(E[0],), surface_vars=[(t,)]),
                              traced_structure=ts)
    batched = fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(0,), surface_vars=[(None,)]))
    sig = batched.operator_signature(fe.TracedParams(volume_vars=(E,), surface_vars=[(t,)]))
    assert sig.shape == (5, ts.csr_indptr.shape[0] - 1)
    order = gene.similarity_order(sig)
    assert [int(i) for i in order] in ([0, 2, 4, 1, 3], [3, 1, 4, 2, 0])
