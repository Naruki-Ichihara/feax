"""Library path: create_batched_homogenization_solver gradient, many runs, with convergence info."""
import sys, jax; jax.config.update("jax_enable_x64", True)
import numpy as onp, jax.numpy as jnp, feax as fe, feax.flat as flat
E0, NU = 100.0, 0.3
class Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E/(2*(1+nu)); lam = E*nu/((1+nu)*(1-2*nu)); eps = 0.5*(u_grad+u_grad.T)
            return lam*jnp.trace(eps)*jnp.eye(3) + 2*mu*eps
        return stress
class Box(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size=0.25): return fe.mesh.box_mesh((1.0, 1.0, 1.0), mesh_size=mesh_size, element_type="HEX8")
cell = Box(mesh_size=0.25); mesh = cell.mesh
problem = Elast(mesh, vec=3, dim=3, ele_type="HEX8")
P = flat.pbc.prolongation_matrix(flat.pbc.periodic_bc_3D(cell, vec=3, dim=3), mesh, vec=3)
bc = fe.DirichletBCConfig([]).create_bc(problem)
B = 3; rho = 0.2 + 0.8*onp.random.default_rng(0).random((B, len(mesh.cells)))
E = jnp.asarray(E0*(1e-3 + rho**3)); nu = fe.TracedParams.create_cell_var(problem, NU)
bs = flat.solver.create_batched_homogenization_solver(problem, bc, P, mesh, dim=3, in_axes=fe.TracedParams(volume_vars=(0, None)), tol=1e-11)
W = jnp.asarray(onp.random.default_rng(1).standard_normal((B, 6, 6)))
def loss(Ea):
    r = bs.evaluate(fe.TracedParams(volume_vars=(Ea, nu)))
    return jnp.sum(W * r.C_hom), r.iterations.max()
g_e, _ = jax.grad(loss, has_aux=True)(E)
comp = jax.jit(jax.grad(loss, has_aux=True)).lower(E).compile()
res = []
for r in range(int(sys.argv[1])):
    g, its = comp(E); res.append(f"{float(jnp.abs(g-g_e).max()/jnp.abs(g_e).max()):.0e}/{int(its)}")
print("library batched homogenization grad runs:", " ".join(res), flush=True)
