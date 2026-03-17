"""Third Medium Contact — FEAX reproduction of FElupe ex20.

Frictionless contact via the third-medium method with HuHu-LuLu
Hessian-based regularization [1, 2].

Two material regions on a single QUAD9 mesh:
  - **Body** (solid): Neo-Hookean compressible with full (G, K)
  - **Medium** (background): Neo-Hookean with scaled-down (γ₀·G, γ₀·K)
    plus biharmonic regularization

Boundary conditions:
  - Fixed at x = 0
  - Prescribed vertical displacement at point (L, H)

Incremental loading: 20 steps ramping to −0.62 L.

References
----------
[1] G. L. Bluhm et al., "Internal contact modeling for finite strain
    topology optimization", Comput. Mech. 67, 1099–1114 (2021).
[2] A. H. Frederiksen et al., "Topology optimization of self-contacting
    structures", Comput. Mech. 73, 967–981 (2023).
"""

import os

import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp

import feax as fe
from feax.assembler import Operator

# ── Geometry parameters (matching FElupe ex20) ────────────────────────
t = 0.1
L = 1.0
H = 0.5

nt, nL, nH = 2, 10, 4

# ── Material parameters ──────────────────────────────────────────────
G = 5.0 / 14.0        # shear modulus (mu)
K = 5.0 / 3.0         # bulk-like Lamé constant (lmbda)
kr = 5e-6             # regularization prefactor
gamma0 = 5e-6         # medium scaling factor

# Regularization coefficient (kr * K * L^2)
kr_coeff = kr * K * L ** 2

# ── Mesh construction ─────────────────────────────────────────────────
# The FElupe example builds sub-meshes and merges them.  The combined
# mesh is a structured QUAD9 grid over [0, 1.1] × [0, 0.5] with uniform
# element size Δx = Δy = 0.1 → Nx = 11, Ny = 5.
Nx_total = 11
Ny_total = 5
domain_x = L + t / (nt - 1)   # 1.1
domain_y = H                   # 0.5

mesh = fe.mesh.rectangle_mesh(
    Nx=Nx_total, Ny=Ny_total,
    domain_x=domain_x, domain_y=domain_y,
    ele_type='QUAD9',
)

# ── Classify cells: body vs. medium ──────────────────────────────────
# Cell ordering in rectangle_mesh: cell(i, j) = i * Ny + j
# where i is x-index (0..Nx-1), j is y-index (0..Ny-1).
#
# Medium cells (sub-meshes f and g in FElupe):
#   f: x ∈ [t, L] × y ∈ [t, H−t]  → ix ∈ {1..9}, iy ∈ {1..3}
#   g: x ∈ [L, L+δ] × y ∈ [0, H]  → ix = 10,     iy ∈ {0..4}
# Body cells: everything else (sub-meshes a, b, c, d, e).

is_medium_np = onp.zeros(Nx_total * Ny_total, dtype=onp.float64)
for ix in range(Nx_total):
    for iy in range(Ny_total):
        cell_idx = ix * Ny_total + iy
        # Sub-mesh f: interior medium
        if 1 <= ix <= 9 and 1 <= iy <= 3:
            is_medium_np[cell_idx] = 1.0
        # Sub-mesh g: right strip
        if ix == 10:
            is_medium_np[cell_idx] = 1.0

is_medium = np.array(is_medium_np)

# Cell-based material properties
mu_body = G
mu_medium = G * gamma0
lmbda_body = K
lmbda_medium = K * gamma0

mu_cell = np.where(is_medium, mu_medium, mu_body)
lmbda_cell = np.where(is_medium, lmbda_medium, lmbda_body)


# ── Problem definition ────────────────────────────────────────────────
class ThirdMediumContact(fe.Problem):
    """Neo-Hookean plane strain + HuHu-LuLu regularization on the medium."""

    def get_energy_density(self):
        """Neo-Hookean compressible energy in plane strain.

        ψ(F) = μ/2 (tr C + 1) − μ ln J + λ/2 (ln J)²

        where F = I + ∇u (2×2), C = F^T F, J = det F.
        The "+1" accounts for the plane-strain F₃₃ = 1 contribution to tr(C).

        A smooth extension below J_min prevents NaN from log(J) when the
        Newton solver overshoots into element inversion (common with the
        ~1e-6 stiffness medium).  The extension matches value and first
        derivative at J_min, with a growing penalty for J < J_min.
        """
        J_min = 1e-4

        def safe_lnJ(J):
            """log(J) with smooth quadratic extension for J < J_min."""
            lnJ_min = np.log(J_min)
            # C1-continuous extension: log(J_min) + (J - J_min)/J_min - 0.5*((J-J_min)/J_min)^2
            t = (J - J_min) / J_min
            ext = lnJ_min + t - 0.5 * t ** 2
            return np.where(J > J_min, np.log(J), ext)

        def psi(u_grad, mu, lmbda, *_unused):
            F = u_grad + np.eye(2)
            C = F.T @ F
            J = np.linalg.det(F)
            lnJ = safe_lnJ(J)
            # Plane strain: tr(C_full) = tr(C_2d) + 1
            return mu / 2.0 * (np.trace(C) + 1.0) - mu * lnJ + lmbda / 2.0 * lnJ ** 2

        return psi

    def get_universal_kernel(self):
        """HuHu-LuLu regularization on the medium cells.

        E_reg = kr·K·L² ∫_medium (H:::H − (1/dim) L·L) dΩ

        H_{ijk} = ∂²u_i / ∂x_j ∂x_k   (Hessian of displacement)
        L_i     = Σ_j H_{ijj}           (Laplacian of displacement)
        """
        dim = self.dim

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                   cell_JxW, cell_v_grads_JxW,
                   mu, lmbda, cell_shape_hess, cell_is_medium):
            # Unpack cell solution
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]          # (num_nodes, vec=2)
            cell_JxW_1d = cell_JxW[0]            # (num_quads,)

            # cell_shape_hess: (num_quads, num_nodes, dim, dim)
            # cell_is_medium: scalar (0.0 or 1.0)

            # Displacement Hessian at quad points: H_{q,v,K,L}
            u_hess = np.einsum('av,qaKL->qvKL', cell_sol, cell_shape_hess)

            # Laplacian: L_{q,v} = trace_KL(H_{q,v,K,L})
            lapl_u = np.trace(u_hess, axis1=2, axis2=3)  # (num_quads, vec)

            # Shape function Laplacian: shape_lapl_{q,a} = trace(hess_{q,a,:,:})
            shape_lapl = np.trace(cell_shape_hess, axis1=-2, axis2=-1)  # (num_quads, num_nodes)

            # Residual per node/component:
            # R_{a,i} = kr_coeff * Σ_q [Σ_{KL} H_{q,i,K,L} hess_{q,a,K,L}
            #           − (1/dim) L_{q,i} shape_lapl_{q,a}] JxW_q * is_medium

            # Term 1: H:::∇²v  →  Σ_{KL} H_{q,v,K,L} · hess_{q,a,K,L}
            term1 = np.einsum('qvKL,qaKL->qav', u_hess, cell_shape_hess)

            # Term 2: (1/dim) L · ∇²v
            term2 = np.einsum('qv,qa->qav', lapl_u, shape_lapl) / dim

            # Integrate: (num_quads, num_nodes, vec) → (num_nodes, vec)
            integrand = (term1 - term2) * cell_JxW_1d[:, None, None]
            result = kr_coeff * cell_is_medium * np.sum(integrand, axis=0)

            return jax.flatten_util.ravel_pytree(result)[0]

        return kernel


problem = ThirdMediumContact(
    mesh, vec=2, dim=2, ele_type='QUAD9', hess=True,
)

# ── Internal variables ────────────────────────────────────────────────
# volume_vars ordering: (mu, lmbda, shape_hessians, is_medium)
shape_hess = problem.fes[0].shape_hessians   # (num_cells, num_quads, num_nodes, dim, dim)
iv = fe.InternalVars(volume_vars=(mu_cell, lmbda_cell, shape_hess, is_medium))

# ── Boundary conditions ──────────────────────────────────────────────
# Fixed: all DOFs at x = 0
bc_fixed = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], 0.0, atol=1e-6),
    component='all',
    value=0.0,
)

# Move: y-displacement at point (L, H) — initially zero, will be ramped
bc_move = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], L, atol=1e-6) & np.isclose(p[1], H, atol=1e-6),
    component='y',
    value=0.0,
)

bc = fe.DirichletBCConfig([bc_fixed, bc_move]).create_bc(problem)

# ── Solver setup ─────────────────────────────────────────────────────
solver_options = fe.DirectSolverOptions(solver='umfpack', verbose=True)
newton_options = fe.NewtonOptions(tol=1e-6, rel_tol=1e-8, max_iter=10, internal_jit=True)

solver = fe.create_solver(
    problem, bc,
    solver_options=solver_options,
    newton_options=newton_options,
    iter_num=None,       # adaptive Newton
    internal_vars=iv,
    symmetric_bc=False,  # non-symmetric BC elimination (partitioned solve)
)

# ── Incremental loading ──────────────────────────────────────────────
num_steps = 20
max_disp = -0.62 * L       # total prescribed displacement

# Find the BC row index for the prescribed y-DOF at (L, H)
# bc.bc_vals is ordered by bc.bc_rows; we need to locate the move DOF.
move_dof_node = None
points_np = onp.array(mesh.points)
for i, pt in enumerate(points_np):
    if abs(pt[0] - L) < 1e-6 and abs(pt[1] - H) < 1e-6:
        move_dof_node = i
        break

assert move_dof_node is not None, f"Could not find node at ({L}, {H})"
move_dof_index = move_dof_node * 2 + 1   # y-component (vec=2, component 1)

# Identify position in bc_rows
bc_rows_np = onp.array(bc.bc_rows)
move_bc_pos = onp.where(bc_rows_np == move_dof_index)[0]
assert len(move_bc_pos) == 1, f"Expected 1 match for DOF {move_dof_index}, got {len(move_bc_pos)}"
move_bc_pos = int(move_bc_pos[0])

sol = fe.zero_like_initial_guess(problem, bc)

# ── VTK output setup ─────────────────────────────────────────────────
import meshio

out_dir = os.path.join(os.path.dirname(__file__), "data", "vtk_tmc")
os.makedirs(out_dir, exist_ok=True)
cells_np = onp.array(mesh.cells)
pts_3d = onp.zeros((len(points_np), 3))
pts_3d[:, :2] = points_np


def compute_element_detF(u_nodes_np):
    """Compute min det(F) per element using corner nodes."""
    detF_per_cell = onp.zeros(len(cells_np))
    for c in range(len(cells_np)):
        n0, n1, n2, n3 = cells_np[c, :4]
        dets = []
        for (na, nb, nc) in [(n0, n1, n3), (n1, n2, n0), (n2, n3, n1), (n3, n0, n2)]:
            xa = points_np[na] + u_nodes_np[na]
            xb = points_np[nb] + u_nodes_np[nb]
            xc = points_np[nc] + u_nodes_np[nc]
            e1 = xb - xa
            e2 = xc - xa
            dets.append(e1[0] * e2[1] - e1[1] * e2[0])
        detF_per_cell[c] = min(dets)
    return detF_per_cell


def save_vtk(step_idx, u_nodes_np, converged):
    u_3d = onp.zeros((len(u_nodes_np), 3))
    u_3d[:, :2] = u_nodes_np
    detF = compute_element_detF(u_nodes_np)
    m = meshio.Mesh(
        points=pts_3d,  # undeformed (reference) configuration
        cells=[("quad9", cells_np)],
        point_data={"displacement": u_3d},
        cell_data={
            "is_medium": [onp.array(is_medium_np)],
            "min_detF": [detF],
        },
    )
    tag = "ok" if converged else "FAIL"
    path = os.path.join(out_dir, f"step_{step_idx:03d}_{tag}.vtu")
    m.write(path)
    return detF


print(f"{'Step':>4s}  {'Disp':>10s}  {'max|u|':>10s}  {'minJ_body':>10s}  {'minJ_med':>10s}  {'conv'}")
print("-" * 70)

for step in range(1, num_steps + 1):
    # Prescribed displacement for this step
    disp = max_disp * step / num_steps

    # Update BC values
    new_bc_vals = bc.bc_vals.at[move_bc_pos].set(disp)
    bc_step = bc.replace_vals(new_bc_vals)

    # With non-symmetric BC elimination, do NOT pre-apply BC values.
    # The Newton solver drives BC DOFs to their values via the modified
    # residual, and the K_10 coupling is maintained through the unsymmetric
    # Jacobian.  Pre-applying would be redundant and can hurt convergence.

    # Solve (pass updated BC)
    sol = solver(iv, sol, bc=bc_step)

    # Report
    sol_list = problem.unflatten_fn_sol_list(sol)
    u = sol_list[0]   # (num_nodes, 2)
    u_np = onp.array(u)
    max_u = float(np.max(np.abs(u)))

    # Check element quality
    detF = compute_element_detF(u_np)
    body_mask = is_medium_np < 0.5
    med_mask = is_medium_np > 0.5
    minJ_body = detF[body_mask].min() if body_mask.any() else 0.0
    minJ_med = detF[med_mask].min() if med_mask.any() else 0.0
    converged = not onp.any(onp.isnan(u_np)) and minJ_body > 0

    # Save VTK
    save_vtk(step, u_np, converged)

    print(f"{step:4d}  {disp:10.4f}  {max_u:10.6f}  {minJ_body:10.4e}  {minJ_med:10.4e}  {'OK' if converged else 'FAIL'}")

    if not converged:
        print(f"  >> Diverged at step {step}. Stopping.")
        break

print("\nDone. VTK files saved in:", out_dir)
