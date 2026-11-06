"""Batched lattice homogenization: C_hom of B BCC lattices with different strut
radii in one fe.vmap batch of the single-cell homogenization solver, CG
preconditioned by one shared cuDSS factorization per group of SHARE designs
(Jacobi without cuDSS).

Run:  python examples/advance/batched_lattice_homogenization.py
"""
import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe
import feax.flat as flat
from feax import gene
from feax.solvers.options import has_cudss

jax.config.update("jax_enable_x64", True)

E_BASE, NU = 9026.0, 0.3
L = 3.0
MESH = 0.15
B = 50
R_MIN, R_MAX = 0.35, 0.75
SHARE = 4
RHO_VOID = 1e-6


class LinearElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2 * mu * eps
        return stress


class Cell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=L, mesh_size=mesh_size, element_type="HEX8")


def bulk_modulus(C):
    return jnp.sum(C[..., :3, :3], axis=(-2, -1)) / 9.0


cell = Cell(mesh_size=MESH)
mesh = cell.mesh
problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[])
P = flat.pbc.prolongation_matrix(flat.pbc.periodic_bc_3D(cell, vec=3, dim=3), mesh, vec=3)
bc = fe.DirichletBCConfig([]).create_bc(problem)
print(f"mesh: {len(mesh.points)} nodes, {len(mesh.cells)} HEX8 cells, reduced DOFs ~{P.shape[1]}; "
      f"cuDSS: {has_cudss()}")

# BCC: 8 corners joined to the centre node
corners = onp.array([[i, j, k] for i in (0, L) for j in (0, L) for k in (0, L)], dtype=float)
nodes = jnp.asarray(onp.vstack([corners, [[L / 2, L / 2, L / 2]]]))
edges = jnp.asarray([[i, 8] for i in range(8)])
radii = onp.linspace(R_MIN, R_MAX, B)
rho = onp.stack([onp.asarray(flat.graph.create_lattice_density_field_nodal(
    problem, flat.graph.create_lattice_function(nodes, edges, radius=r),
    density_solid=1.0, density_void=RHO_VOID)) for r in radii])
E = jnp.asarray(E_BASE * rho)
nu = fe.TracedParams.create_cell_var(problem, NU)
tp = fe.TracedParams(volume_vars=(E, nu))
in_axes = fe.TracedParams(volume_vars=(0, None))

# C_hom of all designs: fe.vmap of the single-cell solver
solve = flat.solver.create_homogenization_solver(problem, bc, P, mesh, dim=3)
pc = fe.shared_factor(scaling=True) if has_cudss() else fe.jacobi()
batched = fe.vmap(solve, in_axes=in_axes, preconditioner=pc, share=SHARE,
                  options=fe.KrylovSolverOptions(solver="cg", tol=1e-8))
r = batched(tp)
C = onp.asarray(r.C_hom)

print(f"\n{B} designs x 6 strain cases ({'shared factorization' if has_cudss() else 'Jacobi'}, share={SHARE})")
print("  radius  rel.density   C11/Es    C12/Es    C44/Es   CG its (max over cases)")
for b in range(B):
    print(f"  {radii[b]:.3f}   {rho[b].mean():.3f}      {C[b, 0, 0] / E_BASE:.4f}   {C[b, 0, 1] / E_BASE:.4f}"
          f"   {C[b, 3, 3] / E_BASE:.4f}    {int(onp.asarray(r.iterations)[b].max())}")

# similarity-ordered groups, factorizations reused on perturbed designs
if has_cudss():
    order = onp.asarray(gene.similarity_order(onp.asarray(batched.operator_signature(tp))))
    factors = batched.factorize(fe.TracedParams(volume_vars=(E[order], nu)))
    E_pert = E[order] * (1.0 + 0.02 * jnp.asarray(onp.random.default_rng(0).standard_normal(E.shape)))
    r_o = batched(fe.TracedParams(volume_vars=(E_pert, nu)), factors=factors)
    print(f"\nsimilarity order {order.tolist()}; reused factorizations on 2%-perturbed designs: "
          f"CG its per design {onp.asarray(r_o.iterations).max(1).tolist()}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rel = rho.mean(axis=1)
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(rel, C[:, 0, 0] / E_BASE, "o-", label="C11 / Es")
ax.loglog(rel, C[:, 3, 3] / E_BASE, "s-", label="C44 / Es")
ax.loglog(rel, onp.asarray(bulk_modulus(C)) / E_BASE, "^-", label="K / Es")
ax.set_xlabel("relative density")
ax.set_ylabel("normalized stiffness")
ax.legend()
ax.set_title(f"BCC lattices, {B} designs in one batch")
fig.tight_layout()
fig.savefig("batched_lattice_homogenization.png", dpi=150)
print("saved batched_lattice_homogenization.png")
