"""Cohesive zone fracture in 3D (Mode I) — Fully matrix-free Newton solver.

Reproduces the tatva quasi-static 3D fracture example using feax:
  - Bulk: 3D linear elasticity (HEX8, energy density via feax)
  - Interface: exponential cohesive law (pure JAX)
  - Solver: feax.create_solver with MatrixFreeOptions (JVP-based tangent)

The mesh consists of two half-blocks (top/bottom) separated at y=0.
A pre-crack extends from x=0 to x=crack_length (free surfaces).
Mode I loading is applied via prescribed displacement at top/bottom faces.

Parameters match the tatva example:
  E = 106e3 Pa, nu = 0.35, Gamma = 15 J/m², sigma_c = 20e3 Pa
  Geometry scaled by Griffith length L_G = 2*mu*Gamma / (pi*(1-nu)*sigma_inf^2)
"""

import logging
import os

import jax.numpy as np
import meshio
import numpy as onp

import feax as fe
from feax.mechanics.cohesive import (
    CohesiveInterface,
    compute_lumped_area_weights,
    exponential_potential,
)
from feax.solvers.matrix_free import (
    LinearSolverOptions,
    MatrixFreeOptions,
    create_energy_fn,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Material parameters (matching tatva example)
# ============================================================
prestrain = 0.1
E = 106e3          # Young's modulus [Pa]
nu = 0.35          # Poisson's ratio
Gamma = 15.0       # Fracture energy [J/m²]
sigma_c = 20e3     # Critical cohesive traction [Pa]

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# ============================================================
# Geometry (matching tatva example)
# ============================================================
sigma_inf = prestrain * E
L_G = 2 * mu * Gamma / (onp.pi * (1 - nu) * sigma_inf**2)
crack_length = L_G
Lx = 10 * L_G
Ly = 2 * L_G       # Total height (top half = L_G)
Lz = 1 * L_G       # Thickness

applied_disp = 2.25 * prestrain * Ly  # Total applied displacement
n_steps = 180
h_tip = 1.5e-3       # Mesh size near crack tip
h_far = 3e-3       # Mesh size far field


# ============================================================
# 3D HEX mesh with split interface at y=0 (tatva-identical)
# ============================================================
def generate_unstructured_hex_fracture_3d(
    length, height, thickness, crack_tip_x, mesh_size_tip, mesh_size_far,
    work_dir=None,
):
    """Generate a 3D fracture mesh with unstructured hexahedra.

    Identical to the tatva example mesh generation: builds top half-block
    with explicit geo kernel, subdivides to hex, mirrors for bottom half.
    """
    import gmsh

    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(work_dir, exist_ok=True)
    filename = os.path.join(work_dir, "temp_half_block_hex.msh")

    gmsh.initialize()
    gmsh.model.add("half_block_hex")

    h_half = height / 2.0

    # Points: Bottom Face (y=0)
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size_far)
    p4 = gmsh.model.geo.addPoint(0, 0, thickness, mesh_size_far)
    pt1 = gmsh.model.geo.addPoint(crack_tip_x, 0, 0, mesh_size_tip)
    pt2 = gmsh.model.geo.addPoint(crack_tip_x, 0, thickness, mesh_size_tip)
    p2 = gmsh.model.geo.addPoint(length, 0, 0, mesh_size_tip)
    p3 = gmsh.model.geo.addPoint(length, 0, thickness, mesh_size_tip)

    # Points: Top Face (y=h/2)
    p5 = gmsh.model.geo.addPoint(0, h_half, 0, mesh_size_far)
    p8 = gmsh.model.geo.addPoint(0, h_half, thickness, mesh_size_far)
    pt3 = gmsh.model.geo.addPoint(crack_tip_x, h_half, 0, mesh_size_far)
    pt4 = gmsh.model.geo.addPoint(crack_tip_x, h_half, thickness, mesh_size_far)
    p6 = gmsh.model.geo.addPoint(length, h_half, 0, mesh_size_far)
    p7 = gmsh.model.geo.addPoint(length, h_half, thickness, mesh_size_far)

    # Lines: Bottom
    l1 = gmsh.model.geo.addLine(p1, pt1)
    l2 = gmsh.model.geo.addLine(pt1, p2)
    l_right_b = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, pt2)
    l4 = gmsh.model.geo.addLine(pt2, p4)
    l_left_b = gmsh.model.geo.addLine(p4, p1)
    l_crack_b = gmsh.model.geo.addLine(pt1, pt2)

    # Lines: Top
    l5 = gmsh.model.geo.addLine(p5, pt3)
    l6 = gmsh.model.geo.addLine(pt3, p6)
    l_right_t = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, pt4)
    l8 = gmsh.model.geo.addLine(pt4, p8)
    l_left_t = gmsh.model.geo.addLine(p8, p5)
    l_crack_t = gmsh.model.geo.addLine(pt3, pt4)

    # Lines: Vertical
    v1 = gmsh.model.geo.addLine(p1, p5)
    v_tip1 = gmsh.model.geo.addLine(pt1, pt3)
    v2 = gmsh.model.geo.addLine(p2, p6)
    v3 = gmsh.model.geo.addLine(p3, p7)
    v_tip2 = gmsh.model.geo.addLine(pt2, pt4)
    v4 = gmsh.model.geo.addLine(p4, p8)

    # Surfaces
    loop_if_1 = gmsh.model.geo.addCurveLoop([l1, l_crack_b, l4, l_left_b])
    s_if_1 = gmsh.model.geo.addPlaneSurface([loop_if_1])
    loop_if_2 = gmsh.model.geo.addCurveLoop([l2, l_right_b, l3, -l_crack_b])
    s_if_2 = gmsh.model.geo.addPlaneSurface([loop_if_2])
    loop_top_1 = gmsh.model.geo.addCurveLoop([l5, l_crack_t, l8, l_left_t])
    s_top_1 = gmsh.model.geo.addPlaneSurface([loop_top_1])
    loop_top_2 = gmsh.model.geo.addCurveLoop([l6, l_right_t, l7, -l_crack_t])
    s_top_2 = gmsh.model.geo.addPlaneSurface([loop_top_2])

    s_left = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l_left_b, v1, -l_left_t, -v4])])
    s_right = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l_right_b, v3, -l_right_t, -v2])])
    s_front_1 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l1, v_tip1, -l5, -v1])])
    s_front_2 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l2, v2, -l6, -v_tip1])])
    s_back_1 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l4, v4, -l8, -v_tip2])])
    s_back_2 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l3, v_tip2, -l7, -v3])])
    s_mid = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l_crack_b, v_tip2, -l_crack_t, -v_tip1])])

    # Volumes
    sl1 = gmsh.model.geo.addSurfaceLoop(
        [s_if_1, s_top_1, s_left, s_front_1, s_back_1, s_mid])
    vol1 = gmsh.model.geo.addVolume([sl1])
    sl2 = gmsh.model.geo.addSurfaceLoop(
        [s_if_2, s_top_2, s_right, s_front_2, s_back_2, s_mid])
    vol2 = gmsh.model.geo.addVolume([sl2])

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [s_if_1, s_if_2], 1, name="interface_surface")
    gmsh.model.addPhysicalGroup(3, [vol1, vol2], 2, name="top_domain")
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

    gmsh.model.mesh.generate(3)
    gmsh.write(filename)
    gmsh.finalize()

    # Read via meshio (same as tatva)
    _m = meshio.read(filename)
    if os.path.exists(filename):
        os.remove(filename)

    points_top = _m.points
    hex_top = _m.cells_dict["hexahedron"]

    # Interface quads from physical group
    interface_surf_idx = _m.cell_sets_dict["interface_surface"]["quad"]
    all_interface_quads_top = _m.cells_dict["quad"][interface_surf_idx]

    # Mirror to create bottom half
    points_bottom = points_top.copy()
    points_bottom[:, 1] *= -1
    points_bottom[:, 1] -= 1e-7

    N_half = len(points_top)

    hex_bottom = hex_top.copy() + N_half
    hex_bottom[:, [1, 3]] = hex_bottom[:, [3, 1]]
    hex_bottom[:, [5, 7]] = hex_bottom[:, [7, 5]]

    coords = onp.vstack([points_top, points_bottom])
    elements = onp.vstack([hex_top, hex_bottom])

    # Active (cohesive) interface quads: x >= crack_tip_x
    quad_coords = points_top[all_interface_quads_top]
    centroids_x = onp.mean(quad_coords[:, :, 0], axis=1)
    active_mask = centroids_x >= (crack_tip_x - 1e-9)
    active_quads_top = all_interface_quads_top[active_mask]
    active_quads_bottom = active_quads_top + N_half

    # Interface nodes
    bottom_interface_nodes = onp.unique(active_quads_bottom)
    top_interface_nodes = bottom_interface_nodes - N_half

    return (coords, elements,
            top_interface_nodes, bottom_interface_nodes,
            active_quads_top, active_quads_bottom)


print("Generating 3D mesh...")
coords, elements, top_nodes, bottom_nodes, \
    active_quads_top, active_quads_bottom = \
    generate_unstructured_hex_fracture_3d(
        length=Lx, height=Ly, thickness=Lz,
        crack_tip_x=crack_length,
        mesh_size_tip=h_tip, mesh_size_far=h_far,
    )

mesh = fe.mesh.Mesh(points=np.array(coords), cells=np.array(elements))
num_nodes = mesh.points.shape[0]
n_cohesive = len(top_nodes)

print(f"Mesh: {num_nodes} nodes, {mesh.cells.shape[0]} elements")
print(f"Interface: {len(top_nodes)} cohesive node pairs, "
      f"{len(active_quads_top)} active quads")


# ============================================================
# feax problem (3D elasticity, HEX8)
# ============================================================
class Elasticity3D(fe.problem.Problem):
    """3D linear elasticity."""

    def get_energy_density(self):
        def psi(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 0.5 * lmbda * np.trace(eps)**2 + mu * np.sum(eps * eps)
        return psi


problem = Elasticity3D(mesh, vec=3, dim=3, ele_type='HEX8')
vec = 3


# ============================================================
# Boundary locations
# ============================================================
y_max = float(np.max(mesh.points[:, 1]))
y_min = float(np.min(mesh.points[:, 1]))
x_min = float(np.min(mesh.points[:, 0]))
z_min = float(np.min(mesh.points[:, 2]))
z_max = float(np.max(mesh.points[:, 2]))

top_face = lambda pt: np.isclose(pt[1], y_max, atol=1e-5)
bottom_face = lambda pt: np.isclose(pt[1], y_min, atol=1e-5)
left_face = lambda pt: np.isclose(pt[0], x_min, atol=1e-5)
front_face = lambda pt: np.isclose(pt[2], z_min, atol=1e-5)
back_face = lambda pt: np.isclose(pt[2], z_max, atol=1e-5)


# ============================================================
# Cohesive interface (node weights from quad element areas)
# ============================================================
coh_top = top_nodes
coh_bottom = bottom_nodes

weights = compute_lumped_area_weights(coh_bottom, coords, active_quads_bottom)
total_area = float(weights.sum())
expected_area = (Lx - crack_length) * Lz
print(f"  Cohesive interface area: {total_area:.6f} (expected: {expected_area:.6f})")

interface = CohesiveInterface.from_axis(
    top_nodes=coh_top, bottom_nodes=coh_bottom,
    weights=weights, normal_axis=1, vec=3, beta=0.0,
)


# ============================================================
# Energy functions
# ============================================================
elastic_energy = create_energy_fn(problem)
cohesive_energy = interface.create_energy_fn(
    exponential_potential, Gamma=Gamma, sigma_c=sigma_c,
)


def total_energy(u_flat, delta_max):
    """Total potential energy = elastic + cohesive."""
    return elastic_energy(u_flat) + cohesive_energy(u_flat, delta_max)


# ============================================================
# Boundary conditions (matching tatva example)
# ============================================================
def make_bc(disp):
    """Create BC for given displacement magnitude (half applied to each face)."""
    specs = [
        # Top face: all DOFs fixed, y = +disp/2
        fe.DCboundary.DirichletBCSpec(top_face, 'x', 0.0),
        fe.DCboundary.DirichletBCSpec(top_face, 'y', disp / 2),
        fe.DCboundary.DirichletBCSpec(top_face, 'z', 0.0),
        # Bottom face: all DOFs fixed, y = -disp/2
        fe.DCboundary.DirichletBCSpec(bottom_face, 'x', 0.0),
        fe.DCboundary.DirichletBCSpec(bottom_face, 'y', -disp / 2),
        fe.DCboundary.DirichletBCSpec(bottom_face, 'z', 0.0),
        # Left face: x fixed
        fe.DCboundary.DirichletBCSpec(left_face, 'x', 0.0),
        # Front/back faces: z fixed (plane strain constraint)
        fe.DCboundary.DirichletBCSpec(front_face, 'z', 0.0),
        fe.DCboundary.DirichletBCSpec(back_face, 'z', 0.0),
    ]
    return fe.DCboundary.DirichletBCConfig(specs).create_bc(problem)


# Solver options and solver (created once, reused for all steps)
solver_options = MatrixFreeOptions(
    newton_tol=1e-8,
    newton_max_iter=200,
    linear_solver=LinearSolverOptions(solver='cg', atol=1e-8, maxiter=200),
    verbose=True,
)
bc0 = make_bc(0.0)
solver = fe.create_solver(
    problem, bc0,
    solver_options=solver_options,
    energy_fn=total_energy,
)


# ============================================================
# Output
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk_3d'), exist_ok=True)


def save_step(u_flat, delta_max_field, step):
    u = problem.unflatten_fn_sol_list(u_flat)[0]
    d_max_full = np.zeros(num_nodes)
    d_max_full = d_max_full.at[coh_bottom].set(delta_max_field)
    d_max_full = d_max_full.at[coh_top].set(delta_max_field)

    vtk_path = os.path.join(data_dir, f'vtk_3d/fracture3d_{step:04d}.vtu')
    fe.utils.save_sol(
        mesh=mesh, sol_file=vtk_path,
        point_infos=[("displacement", u), ("delta_max", d_max_full[:, None])]
    )


# ============================================================
# Quasi-static loading
# ============================================================
print(f"Cohesive fracture 3D (matrix-free Newton): "
      f"{num_nodes} nodes, {mesh.cells.shape[0]} elements, {n_steps} load steps")
print(f"  E={E}, nu={nu}, Gamma={Gamma}, sigma_c={sigma_c}")
print(f"  L_G={L_G:.4f}, crack_length={crack_length:.4f}")
print(f"  Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")
print(f"  applied_disp={applied_disp:.6f}")

import jax

# Upper face node DOFs for reaction force
upper_node_ids = onp.where(onp.isclose(coords[:, 1], y_max, atol=1e-5))[0]
upper_y_dofs = 3 * upper_node_ids + 1

gradient_fn = jax.grad(total_energy)
height = y_max - y_min

u_flat = np.zeros(problem.num_total_dofs_all_vars)
delta_max = np.zeros(interface.n_nodes)

# Tracking arrays (matching tatva plots)
displacement_on_top = [0.0]
force_on_top = [0.0]
energies = {"elastic": [0.0], "cohesive": [0.0]}

for step in range(1, n_steps + 1):
    disp = applied_disp * step / n_steps

    # Apply BC values to initial guess, then solve
    bc = make_bc(disp)
    u_flat = u_flat.at[bc.bc_rows].set(bc.bc_vals)
    u_flat = solver(delta_max, u_flat)

    # Update state variables
    delta_current = interface.get_opening(u_flat)
    delta_max = np.maximum(delta_max, delta_current)

    # Track energies and force
    e_el = float(elastic_energy(u_flat))
    e_coh = float(cohesive_energy(u_flat, delta_max))
    energies["elastic"].append(e_el)
    energies["cohesive"].append(e_coh)

    fint = gradient_fn(u_flat, delta_max)
    force_on_top.append(float(np.sum(fint[upper_y_dofs])))
    displacement_on_top.append(float(np.mean(u_flat[upper_y_dofs])))

    if step % 5 == 0 or step <= 5:
        strain = displacement_on_top[-1] / height * 2
        max_opening = float(delta_current.max())
        logger.info(
            f"  step {step:3d}/{n_steps}: ε={strain:.5f}, δ_max={max_opening:.6f}"
        )
        save_step(u_flat, delta_max, step)

print("Done.")
save_step(u_flat, delta_max, n_steps)


# ============================================================
# Post-processing: Energy and force-displacement curves
# ============================================================
import matplotlib.pyplot as plt

Gamma_W = Gamma * (Lx - crack_length) * Lz
displacement_on_top = onp.array(displacement_on_top)
force_on_top = onp.array(force_on_top)
e_elastic = onp.array(energies["elastic"])
e_cohesive = onp.array(energies["cohesive"])
strain_arr = displacement_on_top / height / 2

fig, axs = plt.subplots(
    1, 2, figsize=(7, 3.8), layout="constrained",
    gridspec_kw={"width_ratios": [1, 1]},
)

axs[0].plot(strain_arr, (e_elastic + e_cohesive) / Gamma_W,
            markevery=5, label="Total")
axs[0].plot(strain_arr, e_elastic / Gamma_W,
            markevery=5, label="Elastic")
axs[0].plot(strain_arr, e_cohesive / Gamma_W,
            markevery=5, label="Cohesive")
axs[0].axhline(1, color="gray", zorder=-1, linestyle="--")
axs[0].set_xlabel(r"$\varepsilon$")
axs[0].set_ylabel(r"$\Psi/\Gamma\cdot{}W$")
axs[0].grid(True)
axs[0].set_xlim(0, 0.04)
axs[0].legend(frameon=False, numpoints=1, markerscale=1.25)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)

axs[1].plot(strain_arr, force_on_top / (sigma_c * Lz * (Lx - crack_length)),
            color="#AC8D18")
axs[1].set_xlabel(r"$\varepsilon$")
axs[1].set_ylabel(r"$F/(\sigma_c \cdot t \cdot W)$")
axs[1].grid(True)
axs[1].set_xlim(0, 0.04)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

fig_path = os.path.join(data_dir, "energy_force_curves.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved plot to {fig_path}")
plt.show()
