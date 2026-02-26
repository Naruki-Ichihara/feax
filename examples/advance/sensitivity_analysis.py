"""Sensitivity analysis: dC_hom[i,j] / drho

Demonstrates differentiating the homogenized stiffness C_hom[0,0]
with respect to the per-cell density field rho using jax.grad.
This is the key ingredient for gradient-based topology optimization.

The chain rule path:
    rho  ->  E_field = E_base * rho  ->  internal_vars  ->  C_hom[0,0]
             (linear)                    (custom VJP adjoint solve)
"""

import jax
import jax.numpy as jnp
import feax as fe
import feax.flat as flat

# ── Problem setup (coarse mesh for speed) ────────────────────────────────────

E_base = 210e9   # Pa
nu_const = 0.3


class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * jnp.trace(eps) * jnp.eye(self.dim) + 2 * mu * eps
        return stress


class BCCUnitCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')


unitcell = BCCUnitCell(mesh_size=0.1)   # ~125 elements — coarse for demo
mesh = unitcell.mesh
print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} elements")

# BCC lattice geometry
corners = jnp.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=jnp.float32)
center  = jnp.array([[0.5, 0.5, 0.5]], dtype=jnp.float32)
nodes = jnp.vstack([corners, center])
edges = jnp.array([[i, 8] for i in range(8)])

problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])

# Initial density field from BCC strut geometry
lattice_func = flat.graph.create_lattice_function(nodes, edges, radius=0.1)
rho0 = flat.graph.create_lattice_density_field(
    problem, lattice_func, density_solid=1.0, density_void=0.01
)

# Periodic BCs and prolongation matrix
pairings = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)
bc = fe.DCboundary.DirichletBCConfig([]).create_bc(problem)

# Homogenization solver
solver_opts = fe.IterativeSolverOptions(solver="cg", tol=1e-10, atol=1e-10, maxiter=10000)
compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P, mesh, solver_options=solver_opts, dim=3
)

# nu is constant — captured in closure, not differentiated
nu_field = fe.internal_vars.InternalVars.create_cell_var(problem, nu_const)


# ── Differentiable objective ──────────────────────────────────────────────────

def objective(rho):
    """C_hom[0,0] as a differentiable function of the density field."""
    E_field = fe.internal_vars.InternalVars.create_cell_var(problem, E_base * rho)
    iv = fe.internal_vars.InternalVars(volume_vars=(E_field, nu_field), surface_vars=())
    return compute_C_hom(iv)[0, 0]


# ── Forward pass ──────────────────────────────────────────────────────────────

print("\nForward pass: C_hom[0,0] ...")
c11 = objective(rho0)
print(f"  C_hom[0,0] = {c11 / 1e9:.4f} GPa")

# ── Reverse-mode gradient: dC11/drho ─────────────────────────────────────────

print("\nBackward pass: dC_hom[0,0]/drho (jax.grad) ...")
grad_fn = jax.jit(jax.grad(objective))
sensitivity = grad_fn(rho0)   # shape: (num_cells,)

print(f"  Sensitivity shape : {sensitivity.shape}  (= num_cells)")
print(f"  Max sensitivity   : {sensitivity.max():.4e}  Pa")
print(f"  Min sensitivity   : {sensitivity.min():.4e}  Pa")

strut_mask = rho0 > 0.5
void_mask  = rho0 < 0.1
print(f"  Mean |sens| in strut region : {jnp.abs(sensitivity[strut_mask]).mean():.4e} Pa")
print(f"  Mean |sens| in void  region : {jnp.abs(sensitivity[void_mask]).mean():.4e} Pa")

# ── Finite-difference check (single cell) ────────────────────────────────────

print("\nFinite-difference check (cell 0) ...")
eps = 1e-3
e0 = jnp.zeros_like(rho0).at[0].set(eps)
fd_grad = (objective(rho0 + e0) - objective(rho0 - e0)) / (2 * eps)
ad_grad = sensitivity[0]
print(f"  AD  : {ad_grad:.6e}")
print(f"  FD  : {fd_grad:.6e}")
print(f"  Rel error: {abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-30):.2e}")
