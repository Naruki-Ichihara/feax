"""Does the first-run corruption hit user-facing paths? (a) single-cell direct homogenization gradient under jit,
(b) fe.vmap solver path with a compliance-like loss (adjoint rhs independent of U)."""
import sys, os, jax; jax.config.update("jax_enable_x64", True)
import numpy as onp, jax.numpy as jnp, feax as fe, feax.flat as flat
WHICH = sys.argv[1]
E0, NU = 100.0, 0.3
class Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E/(2*(1+nu)); lam = E*nu/((1+nu)*(1-2*nu)); eps = 0.5*(u_grad+u_grad.T)
            return lam*jnp.trace(eps)*jnp.eye(3) + 2*mu*eps
        return stress
    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0.0, 0.0, -t])]
if WHICH == "homog":
    class Box(flat.unitcell.UnitCell):
        def mesh_build(self, mesh_size=0.25): return fe.mesh.box_mesh((1.0, 1.0, 1.0), mesh_size=mesh_size, element_type="HEX8")
    cell = Box(mesh_size=0.25); mesh = cell.mesh
    problem = Elast(mesh, vec=3, dim=3, ele_type="HEX8")
    P = flat.pbc.prolongation_matrix(flat.pbc.periodic_bc_3D(cell, vec=3, dim=3), mesh, vec=3)
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    rho = 0.2 + 0.8*onp.random.default_rng(0).random(len(mesh.cells)); E = jnp.asarray(E0*(1e-3 + rho**3))
    nu = fe.TracedParams.create_cell_var(problem, NU)
    solve = flat.solver.create_homogenization_solver(problem, bc, P, mesh, solver_options=fe.DirectSolverOptions(solver="cudss"), dim=3)
    W = jnp.asarray(onp.random.default_rng(1).standard_normal((6, 6)))
    f = lambda Eb: jnp.sum(W * solve(fe.TracedParams(volume_vars=(Eb, nu))).C_hom)
    g_e = jax.grad(f)(E); comp = jax.jit(jax.grad(f)).lower(E).compile()
    res = [f"{float(jnp.abs(comp(E)-g_e).max()/jnp.abs(g_e).max()):.0e}" for _ in range(3)]
    print("single-cell direct homogenization jit(grad): first/later rel err:", " ".join(res), flush=True)
else:
    mesh = fe.mesh.box_mesh((2.0, 0.5, 0.5), mesh_size=0.125, element_type="HEX8")
    right = lambda p: jnp.isclose(p[0], 2.0, atol=1e-6)
    problem = Elast(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[right])
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False); u0 = fe.zero_like_initial_guess(problem, bc)
    N = 4; n_nodes = mesh.points.shape[0]
    E = jnp.exp(0.3 * jax.random.normal(jax.random.PRNGKey(1), (N, n_nodes)))
    t1 = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)
    nu = fe.TracedParams.create_cell_var(problem, NU)
    tp0 = fe.TracedParams(volume_vars=(E[0], nu), surface_vars=[(t1,)])
    solver = fe.create_solver(problem, bc, fe.KrylovSolverOptions(solver="cg", tol=1e-12, maxiter=3000), linear=True, traced_params=tp0, traced_structure=ts)
    batched = fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(0, None), surface_vars=[(None,)]), preconditioner=fe.shared_factor(),
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-12, atol=1e-14, maxiter=3000))
    direct = fe.create_solver(problem, bc, fe.DirectSolverOptions(solver="cudss"), linear=True, traced_params=tp0, traced_structure=ts)
    fvec = jnp.asarray(onp.random.default_rng(2).standard_normal(problem.num_total_dofs_all_vars))
    f = lambda Ea: jnp.sum(batched(fe.TracedParams(volume_vars=(Ea, nu), surface_vars=[(t1,)]), u0, ts).dofs * fvec[None])   # linear in U
    g_ref = jnp.stack([jax.grad(lambda Eb: jnp.sum(direct(fe.TracedParams(volume_vars=(Eb, nu), surface_vars=[(t1,)]), u0, traced_structure=ts).dofs * fvec))(E[i]) for i in range(N)])
    comp = jax.jit(jax.grad(f)).lower(E).compile()
    res = [f"{float(jnp.abs(comp(E)-g_ref).max()/jnp.abs(g_ref).max()):.0e}" for _ in range(3)]
    print(f"fe.vmap solver path ({problem.num_total_dofs_all_vars} DOF x {N}), linear loss, jit(grad): first/later rel err:", " ".join(res), flush=True)
