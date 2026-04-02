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

Incremental loading: 20 steps ramping to −0.4 L.

References
----------
[1] G. L. Bluhm et al., "Internal contact modeling for finite strain
    topology optimization", Comput. Mech. 67, 1099–1114 (2021).
[2] A. H. Frederiksen et al., "Topology optimization of self-contacting
    structures", Comput. Mech. 73, 967–981 (2023).
"""

import os

import jax.numpy as np
import numpy as onp

import feax as fe
from feax.mechanics.tmc import ThirdMediumContact, classify_medium_cells

# ── Geometry parameters (matching FElupe ex20) ────────────────────────
t = 0.1
L = 1.0
H = 0.5

nt = 2

# ── Material parameters ──────────────────────────────────────────────
G = 5.0 / 14.0        # shear modulus (mu)
K = 5.0 / 3.0         # bulk-like Lamé constant (lmbda)
kr = 5e-7             # regularization prefactor
gamma0 = 5e-7         # medium scaling factor (very soft void)

# ── Mesh construction ─────────────────────────────────────────────────
Nx_total = 33
Ny_total = 15
domain_x = L + t / (nt - 1)   # 1.1
domain_y = H                   # 0.5

mesh = fe.mesh.rectangle_mesh(
    Nx=Nx_total, Ny=Ny_total,
    domain_x=domain_x, domain_y=domain_y,
    ele_type='QUAD9',
)

# ── Classify cells: body vs. medium (centroid-based) ─────────────────
is_medium = classify_medium_cells(
    mesh,
    lambda cx, cy: (t < cx < L and t < cy < (H - t)) or cx > L,
    n_corner_nodes=4,
)

is_medium_np = onp.array(is_medium)
n_body = int((is_medium_np < 0.5).sum())
n_med = int((is_medium_np > 0.5).sum())
print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} QUAD9 elements")
print(f"Body cells: {n_body}, Medium cells: {n_med}")

# ── Problem + internal variables via TMC API ─────────────────────────
problem, iv = ThirdMediumContact.create(
    mesh,
    is_medium=is_medium,
    mu=G,
    lmbda=K,
    gamma0=gamma0,
    kr=kr,
    ele_type='QUAD9',
    ref_length=L,
)

# ── Boundary conditions ──────────────────────────────────────────────
bc_fixed = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], 0.0, atol=1e-6),
    component='all',
    value=0.0,
)

bc_move = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], L, atol=1e-6) & np.isclose(p[1], H, atol=1e-6),
    component='y',
    value=0.0,
)

bc = fe.DirichletBCConfig([bc_fixed, bc_move]).create_bc(problem)

# ── Solver setup ─────────────────────────────────────────────────────
solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(solver='umfpack', verbose=True),
    newton_options=fe.NewtonOptions(tol=1e-6, rel_tol=1e-8, max_iter=100, internal_jit=True),
    iter_num=None,
    internal_vars=iv,
    symmetric_bc=False,
)

# ── Incremental loading ──────────────────────────────────────────────
num_steps = 20
max_disp = -0.4 * L

# Find the BC row index for the prescribed y-DOF at (L, H)
points_np = onp.array(mesh.points)
cells_np = onp.array(mesh.cells)

move_dof_node = None
for i, pt in enumerate(points_np):
    if abs(pt[0] - L) < 1e-6 and abs(pt[1] - H) < 1e-6:
        move_dof_node = i
        break

assert move_dof_node is not None, f"Could not find node at ({L}, {H})"
move_dof_index = move_dof_node * 2 + 1   # y-component

bc_rows_np = onp.array(bc.bc_rows)
move_bc_pos = onp.where(bc_rows_np == move_dof_index)[0]
assert len(move_bc_pos) == 1, f"Expected 1 match for DOF {move_dof_index}, got {len(move_bc_pos)}"
move_bc_pos = int(move_bc_pos[0])

sol = fe.zero_like_initial_guess(problem, bc)

# ── VTK output setup ─────────────────────────────────────────────────
import meshio

out_dir = os.path.join(os.path.dirname(__file__), "data", "vtk_tmc")
os.makedirs(out_dir, exist_ok=True)
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
        points=pts_3d,
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


# ── Solve ─────────────────────────────────────────────────────────────
print(f"\n{'Step':>4s}  {'Disp':>10s}  {'max|u|':>10s}  {'minJ_body':>10s}  {'minJ_med':>10s}  {'conv'}")
print("-" * 70)

for step in range(1, num_steps + 1):
    disp = max_disp * step / num_steps

    new_bc_vals = bc.bc_vals.at[move_bc_pos].set(disp)
    bc_step = bc.replace_vals(new_bc_vals)

    sol = solver(iv, sol, bc=bc_step)

    sol_list = problem.unflatten_fn_sol_list(sol)
    u = sol_list[0]
    u_np = onp.array(u)
    max_u = float(np.max(np.abs(u)))

    detF = compute_element_detF(u_np)
    body_mask = is_medium_np < 0.5
    med_mask = is_medium_np > 0.5
    minJ_body = detF[body_mask].min() if body_mask.any() else 0.0
    minJ_med = detF[med_mask].min() if med_mask.any() else 0.0
    converged = not onp.any(onp.isnan(u_np)) and minJ_body > 0

    save_vtk(step, u_np, converged)

    print(f"{step:4d}  {disp:10.4f}  {max_u:10.6f}  {minJ_body:10.4e}  {minJ_med:10.4e}  {'OK' if converged else 'FAIL'}")

    if not converged:
        print(f"  >> Diverged at step {step}. Stopping.")
        break

print("\nDone. VTK files saved in:", out_dir)
