"""Cohesive zone fracture in 2D (Mode I) — Fully matrix-free Newton solver.

Demonstrates fully matrix-free Newton fracture simulation:
  - Bulk: 2D plane strain elasticity (QUAD4, energy density via feax)
  - Interface: exponential cohesive law (pure JAX)
  - Solver: feax.create_solver with MatrixFreeOptions (JVP-based tangent)

The mesh is split along y=0 to create a cohesive interface.
A pre-crack extends from x=0 to x=a (free surfaces).
Mode I loading is applied via prescribed displacement at top/bottom.
"""

import logging
import os

import jax.numpy as np
import numpy as onp

import feax as fe
from feax.mechanics.cohesive import (
    CohesiveInterface,
    compute_trapezoidal_weights,
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
# Material parameters
# ============================================================
E = 1e6           # Young's modulus [Pa]
nu = 0.35         # Poisson's ratio
Gamma = 15.0      # Fracture energy [J/m²]
sigma_c = 20000.0 # Critical cohesive traction [Pa]
delta_c = Gamma / (np.e * sigma_c)  # Critical opening

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# ============================================================
# Geometry
# ============================================================
Lx = 10.0         # Width
Ly = 4.0          # Height (total, centered at y=0)
a = 3.0           # Pre-crack length (from x=0)
h_coarse = 0.1    # Coarse mesh size (far field)
h_fine = 0.01     # Fine mesh size (near crack tip)
n_steps = 200     # Load steps
max_disp = 0.01   # Maximum prescribed displacement (half)


# ============================================================
# Mesh with split interface at y=0 (graded via gmsh)
# ============================================================
def create_split_mesh(Lx, Ly, crack_tip_x, h_fine, h_coarse):
    """Create a QUAD4 mesh graded near the crack tip, split at y=0.

    Uses gmsh distance field to concentrate elements near (crack_tip_x, 0).
    Returns the mesh and interface node pairs.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("cohesive")

        # Rectangle [0, Lx] x [-Ly/2, Ly/2]
        rect = gmsh.model.occ.addRectangle(0, -Ly / 2, 0, Lx, Ly)
        gmsh.model.occ.synchronize()

        # Embed a line at y=0 to force nodes on the interface
        line_y0 = gmsh.model.occ.addLine(
            gmsh.model.occ.addPoint(0, 0, 0),
            gmsh.model.occ.addPoint(Lx, 0, 0),
        )
        gmsh.model.occ.fragment([(2, rect)], [(1, line_y0)])
        gmsh.model.occ.synchronize()

        # Mesh size field 1: refine near crack tip (point)
        tip_pt = gmsh.model.occ.addPoint(crack_tip_x, 0, 0)
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", [tip_pt])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.2)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 3.0)

        # Mesh size field 2: refine along crack path (y=0, x >= crack_tip_x)
        h_crack_path = h_fine
        interface_curves = []
        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            if abs(ymin) < 1e-10 and abs(ymax) < 1e-10:
                interface_curves.append(tag)

        gmsh.model.mesh.field.add("Distance", 3)
        gmsh.model.mesh.field.setNumbers(3, "CurvesList", interface_curves)

        gmsh.model.mesh.field.add("Threshold", 4)
        gmsh.model.mesh.field.setNumber(4, "InField", 3)
        gmsh.model.mesh.field.setNumber(4, "SizeMin", h_crack_path)
        gmsh.model.mesh.field.setNumber(4, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(4, "DistMin", 0.1)
        gmsh.model.mesh.field.setNumber(4, "DistMax", 2.0)

        # Combine: use minimum of both fields
        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])

        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # Generate QUAD4 mesh via recombination
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
        gmsh.model.mesh.generate(2)

        # Extract mesh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        all_points = node_coords.reshape(-1, 3)[:, :2]  # Drop z

        # Get QUAD4 elements (type 3 in gmsh)
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        quad_idx = None
        for i, etype in enumerate(elem_types):
            if etype == 3:  # 4-node quad
                quad_idx = i
                break
        if quad_idx is None:
            raise RuntimeError("No quad elements found. Check gmsh recombination.")
        cells_raw = elem_node_tags[quad_idx].reshape(-1, 4) - 1  # 0-based

        # Reindex to contiguous nodes
        unique_nodes = onp.unique(cells_raw.flatten())
        node_remap = onp.full(int(all_points.shape[0]) + 1, -1, dtype=onp.int64)
        node_remap[unique_nodes] = onp.arange(len(unique_nodes))
        points = all_points[unique_nodes]
        cells = node_remap[cells_raw]

    finally:
        gmsh.finalize()

    # --- Split at y=0 ---
    tol = 1e-10
    interface_mask = onp.abs(points[:, 1]) < tol
    interface_indices = onp.where(interface_mask)[0]
    interface_indices = interface_indices[onp.argsort(points[interface_indices, 0])]

    n_orig = len(points)
    dup_points = points[interface_indices].copy()
    new_points = onp.vstack([points, dup_points])

    node_map = {}
    for i, orig in enumerate(interface_indices):
        node_map[int(orig)] = n_orig + i

    # Remap upper half elements to duplicated nodes
    cell_centroids_y = onp.mean(points[cells, 1], axis=1)
    for cell_idx in range(len(cells)):
        if cell_centroids_y[cell_idx] > tol:
            for j in range(cells.shape[1]):
                nid = int(cells[cell_idx, j])
                if nid in node_map:
                    cells[cell_idx, j] = node_map[nid]

    mesh = fe.mesh.Mesh(
        points=np.array(new_points), cells=np.array(cells))
    bottom_nodes = np.array(interface_indices)
    top_nodes = np.array([node_map[int(i)] for i in interface_indices])

    return mesh, bottom_nodes, top_nodes


mesh, bottom_nodes, top_nodes = create_split_mesh(Lx, Ly, a, h_fine, h_coarse)
num_nodes = mesh.points.shape[0]
interface_x = mesh.points[bottom_nodes, 0]

# Pre-crack mask: nodes where x < a (free, no cohesive traction)
cohesive_mask = interface_x >= a - 1e-10  # True for cohesive nodes
n_cohesive = int(cohesive_mask.sum())

print(f"Mesh: {num_nodes} nodes, {mesh.cells.shape[0]} elements")
print(f"Interface: {len(bottom_nodes)} node pairs, {n_cohesive} cohesive, "
      f"{len(bottom_nodes) - n_cohesive} pre-crack")


# ============================================================
# feax problem (for FE data: shape_grads, JxW, cells)
# ============================================================
class Elasticity(fe.problem.Problem):
    """2D plane strain linear elasticity."""

    def get_energy_density(self):
        def psi(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 0.5 * lmbda * np.trace(eps)**2 + mu * np.sum(eps * eps)
        return psi


# Boundary locations
bottom_face = lambda pt: np.isclose(pt[1], -Ly / 2, atol=1e-5)
top_face = lambda pt: np.isclose(pt[1], Ly / 2, atol=1e-5)
left_face = lambda pt: np.isclose(pt[0], 0.0, atol=1e-5)

problem = Elasticity(mesh, vec=2, dim=2, ele_type='QUAD4')
vec = 2

# ============================================================
# Cohesive interface (using feax.mechanics)
# ============================================================
coh_idx = np.where(cohesive_mask)[0]
coh_bottom = bottom_nodes[coh_idx]
coh_top = top_nodes[coh_idx]
coh_x = interface_x[coh_idx]
weights = compute_trapezoidal_weights(coh_x)

interface = CohesiveInterface.from_axis(
    top_nodes=coh_top, bottom_nodes=coh_bottom,
    weights=weights, normal_axis=1, vec=2,
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
# Boundary conditions
# ============================================================
def make_bc(disp):
    """Create BC for given displacement magnitude."""
    specs = [
        fe.DCboundary.DirichletBCSpec(bottom_face, 'y', -disp),
        fe.DCboundary.DirichletBCSpec(top_face, 'y', disp),
        fe.DCboundary.DirichletBCSpec(left_face, 'x', 0.0),
    ]
    return fe.DCboundary.DirichletBCConfig(specs).create_bc(problem)


# Solver options and solver (created once, reused for all steps)
bc0 = make_bc(0.0)
solver_options = MatrixFreeOptions(
    newton_tol=1e-6,
    newton_max_iter=1000,
    linear_solver=LinearSolverOptions(solver='cg', atol=1e-8, maxiter=200),
    verbose=True,
)
solver = fe.create_solver(
    problem, bc0,
    solver_options=solver_options,
    energy_fn=total_energy,
)


# ============================================================
# Output
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)


def save_step(u_flat, delta_max_field, step):
    u = problem.unflatten_fn_sol_list(u_flat)[0]
    d_max_full = np.zeros(num_nodes)
    d_max_full = d_max_full.at[coh_bottom].set(delta_max_field)
    d_max_full = d_max_full.at[coh_top].set(delta_max_field)

    vtk_path = os.path.join(data_dir, f'vtk/cohesive_{step:04d}.vtu')
    fe.utils.save_sol(
        mesh=mesh, sol_file=vtk_path,
        point_infos=[("displacement", u), ("delta_max", d_max_full[:, None])]
    )


# ============================================================
# Quasi-static loading
# ============================================================
print(f"Cohesive fracture 2D (matrix-free Newton): "
      f"{num_nodes} nodes, {mesh.cells.shape[0]} elements, {n_steps} load steps")
print(f"  E={E}, nu={nu}, Gamma={Gamma}, sigma_c={sigma_c}")
print(f"  Pre-crack: x=[0, {a}], Cohesive: x=[{a}, {Lx}]")

u_flat = np.zeros(problem.num_total_dofs_all_vars)
delta_max = np.zeros(interface.n_nodes)

for step in range(1, n_steps + 1):
    disp = max_disp * step / n_steps

    # Apply BC values to initial guess, then solve
    bc = make_bc(disp)
    u_flat = u_flat.at[bc.bc_rows].set(bc.bc_vals)
    u_flat = solver(delta_max, u_flat)

    # Update state variables
    delta_current = interface.get_opening(u_flat)
    delta_max = np.maximum(delta_max, delta_current)

    if step % 2 == 0 or step <= 5:
        max_opening = float(delta_current.max())
        logger.info(
            f"  step {step:3d}: disp={disp:.5f}, δ_max={max_opening:.6f}"
        )
        save_step(u_flat, delta_max, step)

print("Done.")
save_step(u_flat, delta_max, n_steps)
