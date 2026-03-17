"""Third Medium Contact 3D — HEX20 version of the third-medium contact example.

3D extension of third_medium_contact.py using quadratic hexahedral (HEX20)
elements.  The 2D geometry is extruded in z with a thin depth (D = 0.2).

Two material regions on a single HEX20 mesh:
  - **Body** (solid): Neo-Hookean compressible with full (G, K)
  - **Medium** (background): Neo-Hookean with scaled-down (γ₀·G, γ₀·K)
    plus biharmonic regularization

Boundary conditions:
  - Fixed at x = 0
  - Prescribed vertical displacement at point (L, H, 0) only (one end)
  - z = 0 face constrained in z (symmetry plane)
  - z = D face free

Incremental loading: 20 steps ramping to −0.62 L.
"""

import os

import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import gmsh
import meshio

import feax as fe
from feax.assembler import Operator

# ── Geometry parameters (matching 2D example) ───────────────────────
t = 0.1
L = 1.0
H = 0.5
D = 0.2       # depth in z-direction

# ── Material parameters ─────────────────────────────────────────────
G = 5.0 / 14.0        # shear modulus (mu)
K = 5.0 / 3.0         # bulk-like Lamé constant (lmbda)
kr = 5e-6             # regularization prefactor
gamma0 = 5e-6         # medium scaling factor

# Regularization coefficient (kr * K * L^2)
kr_coeff = kr * K * L ** 2

# ── Mesh construction (HEX20 via Gmsh) ──────────────────────────────
domain_x = L + t / 1   # 1.1  (body + medium strip)
domain_y = H            # 0.5
domain_z = D            # 0.2

dx = 0.05  # element size (finer than 2D version)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

try:
    gmsh.model.add("box3d")
    box_tag = gmsh.model.occ.addBox(0, 0, 0, domain_x, domain_y, domain_z)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", dx)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", dx)

    # Structured transfinite meshing
    nx = max(2, int(domain_x / dx) + 1)
    ny = max(2, int(domain_y / dx) + 1)
    nz = max(2, int(domain_z / dx) + 1)

    curves = gmsh.model.getEntities(1)
    for _, ctag in curves:
        bounds = gmsh.model.getBoundingBox(1, ctag)
        ddx = abs(bounds[3] - bounds[0])
        ddy = abs(bounds[4] - bounds[1])
        ddz = abs(bounds[5] - bounds[2])
        if ddx > ddy and ddx > ddz:
            gmsh.model.mesh.setTransfiniteCurve(ctag, nx)
        elif ddy > ddx and ddy > ddz:
            gmsh.model.mesh.setTransfiniteCurve(ctag, ny)
        else:
            gmsh.model.mesh.setTransfiniteCurve(ctag, nz)

    for _, stag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(stag)
        gmsh.model.mesh.setRecombine(2, stag)

    gmsh.model.mesh.setTransfiniteVolume(box_tag)
    gmsh.model.mesh.setRecombine(3, box_tag)

    # Generate first-order mesh, then elevate to order 2 (HEX20 serendipity)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.setOrder(2)

    # Write to temporary file and read with meshio
    tmp_msh = "/tmp/_tmc3d.msh"
    gmsh.write(tmp_msh)
finally:
    gmsh.finalize()

gmsh_mesh = meshio.read(tmp_msh)
mesh = fe.Mesh.from_gmsh(gmsh_mesh, element_type='HEX20')
points_np = onp.array(mesh.points)
cells_np = onp.array(mesh.cells)

print(f"Mesh: {len(points_np)} nodes, {len(cells_np)} HEX20 elements")

# ── Classify cells: body vs. medium (by centroid) ───────────────────
# Medium region (same x-y logic as 2D, extruded through all z):
#   Interior medium: x ∈ [t, L], y ∈ [t, H-t]
#   Right strip:     x > L
# Body: everything else

cell_centroids = onp.mean(points_np[cells_np[:, :8]], axis=1)  # use corner nodes

is_medium_np = onp.zeros(len(cells_np), dtype=onp.float64)
for c in range(len(cells_np)):
    cx, cy, _ = cell_centroids[c]
    # Interior medium
    if t < cx < L and t < cy < (H - t):
        is_medium_np[c] = 1.0
    # Right strip
    if cx > L:
        is_medium_np[c] = 1.0

is_medium = np.array(is_medium_np)

n_body = int((is_medium_np < 0.5).sum())
n_med = int((is_medium_np > 0.5).sum())
print(f"Body cells: {n_body}, Medium cells: {n_med}")

# Cell-based material properties
mu_cell = np.where(is_medium, G * gamma0, G)
lmbda_cell = np.where(is_medium, K * gamma0, K)


# ── Problem definition ───────────────────────────────────────────────
class ThirdMediumContact3D(fe.Problem):
    """Neo-Hookean 3D + HuHu-LuLu regularization on the medium."""

    def get_energy_density(self):
        """Neo-Hookean compressible energy in 3D.

        ψ(F) = μ/2 tr(C) − μ ln J + λ/2 (ln J)²
        """
        J_min = 1e-4

        def safe_lnJ(J):
            lnJ_min = np.log(J_min)
            s = (J - J_min) / J_min
            ext = lnJ_min + s - 0.5 * s ** 2
            return np.where(J > J_min, np.log(J), ext)

        def psi(u_grad, mu, lmbda, *_unused):
            F = u_grad + np.eye(3)
            C = F.T @ F
            J = np.linalg.det(F)
            lnJ = safe_lnJ(J)
            return mu / 2.0 * np.trace(C) - mu * lnJ + lmbda / 2.0 * lnJ ** 2

        return psi

    def get_universal_kernel(self):
        """HuHu-LuLu regularization on the medium cells."""
        dim = self.dim

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                   cell_JxW, cell_v_grads_JxW,
                   mu, lmbda, cell_shape_hess, cell_is_medium):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]          # (num_nodes, vec=3)
            cell_JxW_1d = cell_JxW[0]            # (num_quads,)

            # Displacement Hessian at quad points: H_{q,v,K,L}
            u_hess = np.einsum('av,qaKL->qvKL', cell_sol, cell_shape_hess)

            # Laplacian: L_{q,v} = trace_KL(H_{q,v,K,L})
            lapl_u = np.trace(u_hess, axis1=2, axis2=3)  # (num_quads, vec)

            # Shape function Laplacian: shape_lapl_{q,a} = trace(hess_{q,a,:,:})
            shape_lapl = np.trace(cell_shape_hess, axis1=-2, axis2=-1)

            # Term 1: H:::∇²v
            term1 = np.einsum('qvKL,qaKL->qav', u_hess, cell_shape_hess)

            # Term 2: (1/dim) L · ∇²v
            term2 = np.einsum('qv,qa->qav', lapl_u, shape_lapl) / dim

            # Integrate
            integrand = (term1 - term2) * cell_JxW_1d[:, None, None]
            result = kr_coeff * cell_is_medium * np.sum(integrand, axis=0)

            return jax.flatten_util.ravel_pytree(result)[0]

        return kernel


problem = ThirdMediumContact3D(
    mesh, vec=3, dim=3, ele_type='HEX20', hess=True,
)

# ── Internal variables ───────────────────────────────────────────────
shape_hess = problem.fes[0].shape_hessians
iv = fe.InternalVars(volume_vars=(mu_cell, lmbda_cell, shape_hess, is_medium))

# ── Boundary conditions ─────────────────────────────────────────────
# Fixed: all DOFs at x = 0
bc_fixed = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], 0.0, atol=1e-6),
    component='all',
    value=0.0,
)

# Move: y-displacement at point (L, H, 0) only — one end push
bc_move = fe.DirichletBCSpec(
    location=lambda p: (
        np.isclose(p[0], L, atol=1e-6)
        & np.isclose(p[1], H, atol=1e-6)
        & np.isclose(p[2], 0.0, atol=1e-6)
    ),
    component='y',
    value=0.0,
)

# Constrain z-displacement on z = 0 face only (symmetry); z = D face is free
bc_z0 = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[2], 0.0, atol=1e-6),
    component='z',
    value=0.0,
)

bc = fe.DirichletBCConfig([bc_fixed, bc_move, bc_z0]).create_bc(problem)

# ── Solver setup ─────────────────────────────────────────────────────
solver_options = fe.DirectSolverOptions(solver='umfpack', verbose=True)
newton_options = fe.NewtonOptions(tol=1e-6, rel_tol=1e-8, max_iter=10, internal_jit=True)

solver = fe.create_solver(
    problem, bc,
    solver_options=solver_options,
    newton_options=newton_options,
    iter_num=None,       # adaptive Newton
    internal_vars=iv,
    symmetric_bc=False,
)

# ── Incremental loading ─────────────────────────────────────────────
num_steps = 20
max_disp = -0.62 * L

# Find BC row index for prescribed y-DOF at (L, H, 0) — single node
move_dof_node = None
for i, pt in enumerate(points_np):
    if abs(pt[0] - L) < 1e-6 and abs(pt[1] - H) < 1e-6 and abs(pt[2]) < 1e-6:
        move_dof_node = i
        break

assert move_dof_node is not None, f"Could not find node at ({L}, {H}, 0)"
move_dof_index = move_dof_node * 3 + 1   # y-component (vec=3, component 1)
print(f"Prescribed DOF: node {move_dof_node} at ({L}, {H}, 0)")

# Identify position in bc_rows
bc_rows_np = onp.array(bc.bc_rows)
move_bc_pos = onp.where(bc_rows_np == move_dof_index)[0]
assert len(move_bc_pos) == 1, f"Expected 1 match for DOF {move_dof_index}, got {len(move_bc_pos)}"
move_bc_pos = int(move_bc_pos[0])

sol = fe.zero_like_initial_guess(problem, bc)

# ── VTK output setup ────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "data", "vtk_tmc_3d")
os.makedirs(out_dir, exist_ok=True)


def save_vtk(step_idx, u_nodes_np, converged):
    u_3d = u_nodes_np.reshape(-1, 3)
    m = meshio.Mesh(
        points=points_np,   # undeformed (reference) configuration
        cells=[("hexahedron20", cells_np)],
        point_data={"displacement": u_3d},
        cell_data={
            "is_medium": [onp.array(is_medium_np)],
        },
    )
    path = os.path.join(out_dir, f"step_{step_idx:03d}.vtu")
    m.write(path)


print(f"\n{'Step':>4s}  {'Disp':>10s}  {'max|u|':>10s}  {'conv'}")
print("-" * 40)

for step in range(1, num_steps + 1):
    disp = max_disp * step / num_steps

    # Update BC value for the single prescribed node
    new_bc_vals = bc.bc_vals.at[move_bc_pos].set(disp)
    bc_step = bc.replace_vals(new_bc_vals)

    # Solve
    sol = solver(iv, sol, bc=bc_step)

    # Report
    sol_list = problem.unflatten_fn_sol_list(sol)
    u = sol_list[0]   # (num_nodes, 3)
    u_np = onp.array(u)
    max_u = float(np.max(np.abs(u)))
    converged = not onp.any(onp.isnan(u_np))

    # Save VTK
    save_vtk(step, u_np, converged)

    print(f"{step:4d}  {disp:10.4f}  {max_u:10.6f}  {'OK' if converged else 'FAIL'}")

    if not converged:
        print(f"  >> Diverged at step {step}. Stopping.")
        break

print("\nDone. VTK files saved in:", out_dir)
