"""Batched Dirichlet BC example: solve the same linear elasticity problem
for multiple prescribed displacements in one vectorised call using jax.vmap.

Problem setup:
  - 2D plane-stress cantilever (rectangle mesh)
  - Left face fixed (all components = 0)
  - Right face: prescribed x-displacement varies across the batch

The key idea is that DirichletBC is a JAX pytree.  You can swap its
``bc_vals`` via ``bc.replace_vals(new_vals)`` and vmap over the values.
"""

import jax
import jax.numpy as np

import feax as fe

# ── Material ────────────────────────────────────────────────────────────────
E, nu = 70e3, 0.3
batch_size = 10000

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress

# ── Mesh & problem ──────────────────────────────────────────────────────────
mesh = fe.mesh.rectangle_mesh(Nx=20, Ny=5, domain_x=10.0, domain_y=2.0)

left = lambda p: np.isclose(p[0], 0.0, atol=1e-5)
right = lambda p: np.isclose(p[0], 10.0, atol=1e-5)

problem = LinearElasticity(mesh, vec=2, dim=2, ele_type='QUAD4')

# ── Boundary conditions ────────────────────────────────────────────────────
bc_config = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=left, component="all", value=0.0),
    fe.DirichletBCSpec(location=right, component="x", value=0.0),  # placeholder
])
bc = bc_config.create_bc(problem)

# ── Solver (iterative — pure JAX, vmappable) ───────────────────────────────
iv = fe.InternalVars(volume_vars=())
solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1,
    internal_vars=iv,
)

# ── Batch of prescribed displacements ──────────────────────────────────────
# Identify which entries in bc_vals correspond to the right-face x-DOFs.
# bc_vals is ordered by bc_rows; the right-face x entries are the ones we
# set to 0.0 above.  Here we simply overwrite *all* bc_vals per batch entry
# so that the left-face zeros stay zero and the right-face value changes.

displacements = np.linspace(0.1, 100, batch_size)

# Find position of right-face x-DOFs inside bc_vals
right_nodes = np.argwhere(
    jax.vmap(right)(mesh.points)
).reshape(-1)
right_x_dofs = right_nodes * problem.fes[0].vec  # x-component DOF indices

# Build a batch of bc_vals: shape (n_batch, n_bc_dofs)
def make_bc_vals(disp):
    return bc.bc_vals.at[
        np.searchsorted(bc.bc_rows, right_x_dofs)
    ].set(disp)

bc_vals_batch = jax.vmap(make_bc_vals)(displacements)

# ── Solve (vectorised) ─────────────────────────────────────────────────────
@jax.jit
def solve_batch(vals_batch):
    return jax.vmap(lambda v: solver(iv, bc=bc.replace_vals(v)))(vals_batch)

sols = solve_batch(bc_vals_batch)  # (n_batch, total_dofs)

# ── Verify against sequential solves ───────────────────────────────────────
print("Batched Dirichlet BC — linear elasticity (2D)")
print(f"  Batch size : {len(displacements)}")
print(f"  Mesh       : {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")
print("\nDone.")
