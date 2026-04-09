"""Third Medium Contact 3D — HEX20 version of the third-medium contact example.

3D extension of third_medium_contact.py using quadratic hexahedral (HEX20)
elements.  The 2D geometry is extruded in z with a thin depth (D = 0.2).

Two material regions on a single HEX20 mesh:
  - **Body** (solid): Neo-Hookean compressible with full (G, K)
  - **Medium** (background): Neo-Hookean with scaled-down (γ₀·G, γ₀·K)
    plus biharmonic regularization

Boundary conditions:
  - Fixed at x = 0
  - Prescribed uniform vertical displacement on top edge (x = L, y = H)
  - z = 0 face constrained in z (symmetry plane)
  - z = D face free

Incremental loading: 20 steps ramping to −0.62 L.
"""

import os

import jax.numpy as np
import numpy as onp
import gmsh
import meshio

import feax as fe
from feax.mechanics.tmc import ThirdMediumContact, classify_medium_cells

# ── Geometry parameters (matching 2D example) ───────────────────────
t = 0.1
L = 1.0
H = 0.5
D = 0.2       # depth in z-direction

# ── Material parameters ─────────────────────────────────────────────
G = 5.0 / 14.0        # shear modulus (mu)
K = 5.0 / 3.0         # bulk-like Lamé constant (lmbda)
kr = 5e-7             # regularization prefactor
gamma0 = 5e-7         # medium scaling factor (very soft void)

# ── Mesh construction (HEX20 via Gmsh) ──────────────────────────────
domain_x = L + t / 1   # 1.1  (body + medium strip)
domain_y = H            # 0.5
domain_z = D            # 0.2

dx = 0.033  # element size (matching 2D: 33×15 grid)

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

    tmp_msh = "/tmp/_tmc3d.msh"
    gmsh.write(tmp_msh)
finally:
    gmsh.finalize()

gmsh_mesh = meshio.read(tmp_msh)
mesh = fe.Mesh.from_gmsh(gmsh_mesh, element_type='HEX20')
points_np = onp.array(mesh.points)
cells_np = onp.array(mesh.cells)

print(f"Mesh: {len(points_np)} nodes, {len(cells_np)} HEX20 elements")

# ── Classify cells: body vs. medium ─────────────────────────────────
is_medium = classify_medium_cells(
    mesh,
    lambda cx, cy, cz: (t < cx < L and t < cy < (H - t)) or cx > L,
    n_corner_nodes=8,
)

is_medium_np = onp.array(is_medium)
n_body = int((is_medium_np < 0.5).sum())
n_med = int((is_medium_np > 0.5).sum())
print(f"Body cells: {n_body}, Medium cells: {n_med}")

# ── Problem + internal variables via TMC API ─────────────────────────
problem, iv = ThirdMediumContact.create(
    mesh,
    is_medium=is_medium,
    mu=G,
    lmbda=K,
    gamma0=gamma0,
    kr=kr,
    ele_type='HEX20',
    ref_length=L,
)

# ── Boundary conditions ─────────────────────────────────────────────
bc_fixed = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], 0.0, atol=1e-6),
    component='all',
    value=0.0,
)

# Uniform y-displacement on the top edge at x = L, y = H
bc_move = fe.DirichletBCSpec(
    location=lambda p: (
        np.isclose(p[0], L, atol=1e-6)
        & np.isclose(p[1], H, atol=1e-6)
    ),
    component='y',
    value=0.0,
)

# Symmetry: z = 0 face constrained in z
bc_z0 = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[2], 0.0, atol=1e-6),
    component='z',
    value=0.0,
)

bc = fe.DirichletBCConfig([bc_fixed, bc_move, bc_z0]).create_bc(problem)

# ── Solver setup ─────────────────────────────────────────────────────
solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(solver='spsolve', verbose=True),
    newton_options=fe.NewtonOptions(tol=1e-6, rel_tol=1e-8, max_iter=30, internal_jit=True),
    iter_num=None,
    internal_vars=iv,
    symmetric_bc=False,
)

# ── Incremental loading ─────────────────────────────────────────────
num_steps = 20
max_disp = -0.62 * L

# Find BC row indices for prescribed y-DOFs on the top edge
move_dof_nodes = [i for i, pt in enumerate(points_np)
                  if abs(pt[0] - L) < 1e-6 and abs(pt[1] - H) < 1e-6]
assert len(move_dof_nodes) > 0, f"No nodes found at x = {L}, y = {H}"
move_dof_indices = [n * 3 + 1 for n in move_dof_nodes]  # y-component
print(f"Prescribed DOFs: {len(move_dof_nodes)} nodes on edge x = {L}, y = {H}")

bc_rows_np = onp.array(bc.bc_rows)
move_bc_positions = []
for dof_idx in move_dof_indices:
    pos = onp.where(bc_rows_np == dof_idx)[0]
    assert len(pos) == 1, f"Expected 1 match for DOF {dof_idx}, got {len(pos)}"
    move_bc_positions.append(int(pos[0]))
move_bc_positions = onp.array(move_bc_positions)

sol = fe.zero_like_initial_guess(problem, bc)

# ── VTK output setup ────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "data", "vtk_tmc_3d")
os.makedirs(out_dir, exist_ok=True)


def save_vtk(step_idx, u_nodes_np, converged):
    u_3d = u_nodes_np.reshape(-1, 3)
    m = meshio.Mesh(
        points=points_np,
        cells=[("hexahedron20", cells_np)],
        point_data={"displacement": u_3d},
        cell_data={"is_medium": [onp.array(is_medium_np)]},
    )
    path = os.path.join(out_dir, f"step_{step_idx:03d}.vtu")
    m.write(path)


# ── Solve ─────────────────────────────────────────────────────────────
print(f"\n{'Step':>4s}  {'Disp':>10s}  {'max|u|':>10s}  {'conv'}")
print("-" * 40)

for step in range(1, num_steps + 1):
    disp = max_disp * step / num_steps

    new_bc_vals = bc.bc_vals.at[move_bc_positions].set(disp)
    bc_step = bc.replace_vals(new_bc_vals)

    sol = solver(iv, sol, bc=bc_step)

    sol_list = problem.unflatten_fn_sol_list(sol)
    u = sol_list[0]
    u_np = onp.array(u)
    max_u = float(np.max(np.abs(u)))
    converged = not onp.any(onp.isnan(u_np))

    save_vtk(step, u_np, converged)

    print(f"{step:4d}  {disp:10.4f}  {max_u:10.6f}  {'OK' if converged else 'FAIL'}")

    if not converged:
        print(f"  >> Diverged at step {step}. Stopping.")
        break

print("\nDone. VTK files saved in:", out_dir)
