"""Arbitrary lattice -> spgrid -> NarrowBandCMG -> solution on a recovered mesh.

The full spgrid workflow on a geometry that is NOT grid-based:

  1. **Define** a lattice as a plain 3D graph — a twisted tower whose struts
     are oblique to every grid axis — and **generate a real triangle mesh**
     for it (trimesh cylinders; also written to STL).
  2. **Embed** it into an enclosing spgrid: ``StructuredGrid.fit`` places the
     background grid over the geometry's bounding box (``align=8`` keeps it
     multigrid-coarsenable), ``voxelize`` marks the occupied cells.
  3. **Solve** linear elasticity on the active band only with
     ``NarrowBandCMG`` — matrix-free geometric multigrid, O(band) memory;
     the design enters as ``TracedParams`` (via ``SparseDesign``) and the
     result comes back as a ``fe.Solution``.
  4. **Recover a mesh**: ``NarrowBand(grid, active)`` materializes the band
     as an explicit HEX8 ``Mesh`` sharing the cmg node ordering, so the
     displacement field maps onto it directly (saved as VTU).

The solve is verified against a feax-assembled stiffness matrix on the
recovered mesh: ``|| (K u - b)_free || / ||b||`` should be at the MGPCG
tolerance.

Note: cmg operates on the unit-spacing index grid, so the lattice below is
defined directly in cell units (h = 1).
"""

import os

import numpy as onp
import jax.numpy as np
import trimesh

import feax as fe

# ── 1. Lattice graph (non-grid-based) + real mesh ────────────────────────────
# Twisted tower: square rings at several heights, each rotated by 18° w.r.t.
# the previous one — verticals, ring edges and diagonals are all oblique.
# Dimensions are in cell units (h = 1): a fine background grid with slender
# struts (radius 2 cells vs. ring radius 40 -> slenderness ~1/20).
n_levels, ring_r, level_h, twist = 6, 40.0, 30.0, onp.deg2rad(18.0)
strut_r = 2.0                              # strut radius (in cell units)

rings = []
for l in range(n_levels + 1):
    a = l * twist
    ring = [(ring_r * onp.cos(a + q * onp.pi / 2),
             ring_r * onp.sin(a + q * onp.pi / 2),
             l * level_h) for q in range(4)]
    rings.append(onp.array(ring))
nodes = onp.vstack(rings)                                  # (4*(n+1), 3)

nid = lambda l, q: 4 * l + q
edges = []
for l in range(n_levels):
    for q in range(4):
        edges.append((nid(l, q), nid(l + 1, q)))           # twisted verticals
        edges.append((nid(l, q), nid(l + 1, (q + 1) % 4))) # diagonals
for l in range(n_levels + 1):
    for q in range(4):
        edges.append((nid(l, q), nid(l, (q + 1) % 4)))     # ring edges

# A real triangle mesh of the lattice (arbitrary shape — nothing grid-based)
cylinders = [trimesh.creation.cylinder(radius=strut_r, segment=nodes[[i, j]],
                                       sections=24) for i, j in edges]
lattice_mesh = trimesh.util.concatenate(cylinders)
print(f"lattice mesh: {len(lattice_mesh.vertices)} vertices, "
      f"{len(lattice_mesh.faces)} triangles, {len(edges)} struts")

# ── 2. Enclosing spgrid + voxelization ───────────────────────────────────────
grid = fe.StructuredGrid.fit(nodes, h=1.0, pad_cells=3, align=8)
print(f"spgrid: {grid.nx} x {grid.ny} x {grid.nz} cells "
      f"({grid.num_cells:,} total), origin {grid.origin}")

# Occupancy sampling: deterministic point cloud filling each strut (axis +
# concentric rings, denser than the cell size); O(points), never touches the
# full grid.
def strut_points(p0, p1, r, h):
    axis = p1 - p0
    length = onp.linalg.norm(axis)
    d = axis / length
    # local frame
    e1 = onp.cross(d, [0., 0., 1.])
    if onp.linalg.norm(e1) < 1e-8:
        e1 = onp.cross(d, [0., 1., 0.])
    e1 /= onp.linalg.norm(e1)
    e2 = onp.cross(d, e1)
    t = onp.linspace(0.0, 1.0, max(2, int(3 * length / h)))[:, None]
    centers = p0 + t * axis                                   # (nt, 3)
    pts = [centers]
    for rho in (0.5 * r, 0.95 * r):
        for ang in onp.linspace(0, 2 * onp.pi, 8, endpoint=False):
            off = rho * (onp.cos(ang) * e1 + onp.sin(ang) * e2)
            pts.append(centers + off)
    return onp.vstack(pts)

samples = onp.vstack([strut_points(nodes[i], nodes[j], strut_r, 1.0)
                      for i, j in edges])
active = grid.voxelize(samples)
print(f"active band: {active.size:,} cells "
      f"({100.0 * active.size / grid.num_cells:.1f}% of the grid)")

# ── 3. Narrow-band geometric-multigrid solve ─────────────────────────────────
z0, z1 = nodes[:, 2].min(), nodes[:, 2].max()
oz = grid.origin[2]

# Dirichlet: fix every band node in the bottom ring layer (by grid index -> z)
fixed_pred = lambda ni, nj, nk, nx, ny, nz: (oz + nk) < z0 + 1.0

# Physical scale: cell = 1 mm, E = 70 GPa (aluminium, in MPa), total load 500 N
E0, NU, F_TOTAL = 70e3, 0.3, 500.0

# Slender oblique struts are bending-dominated -> harder for the MG smoother:
# use more smoothing sweeps and a higher PCG iteration cap than the defaults.
cmg = fe.NarrowBandCMG(grid, fixed_pred, nu=NU, E0=E0, penal=3.0,
                       pre=3, post=3, cg_tol=1e-9, cg_maxit=2000)
levels = cmg.build(active)
print(f"cmg: {cmg.L} MG levels, band DOFs = {levels[0]['ndof']:,}")

# Load: unit downward force on the band nodes of the top ring layer
top_candidates = grid.nodes_where(lambda I, J, K: (oz + K) > z1 - 1.0)
top_nodes = onp.intersect1d(top_candidates, levels[0]["nodes"])
# Distribute the total force over the loaded nodes
b = cmg.load_vector(levels, top_nodes, comp=2, value=-F_TOTAL / top_nodes.size)
print(f"load: {F_TOTAL:.0f} N over {top_nodes.size} top nodes")

# Design via SparseDesign -> TracedParams (fully solid lattice)
design = fe.SparseDesign.uniform(active, 1.0)
solver = cmg.create_solver(levels, b)      # cuDSS coarsest; returns fe.Solution
sol = solver(design.traced_params(active))

u = sol.field(0)                           # (num_band_nodes, 3)
compliance = float(np.dot(np.asarray(b), sol.flat))
print(f"compliance = {compliance:.6e}, max|u| = {float(np.max(np.abs(u))):.4e}")

# ── 4. Mesh recovery + verification + output ─────────────────────────────────
band = fe.NarrowBand(grid, active)         # explicit HEX8 mesh of the band
assert onp.array_equal(band.node_map, levels[0]["nodes"])   # same node order


class Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            E, nu = E0, NU                 # must match the cmg material
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
        return stress


# Residual check against the feax-assembled operator on the recovered mesh:
# (K u)_free must reproduce the applied load b_free.
problem = Elast(band.mesh, vec=3, dim=3, ele_type='HEX8')
tp0 = fe.TracedParams(volume_vars=())
ts = fe.TracedStructure.from_problem(problem)
K = fe.get_jacobian(problem, problem.unflatten_fn_sol_list(sol), tp0, ts)
free = onp.repeat(onp.asarray(levels[0]["free"])[0::3] > 0, 3)
rel_res = float(onp.linalg.norm((onp.asarray(K @ sol.flat) - b)[free])
                / onp.linalg.norm(b))
print(f"verification vs feax-assembled K on the recovered mesh: "
      f"||(K u - b)_free|| / ||b|| = {rel_res:.2e}")
assert rel_res < 1e-6

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
lattice_mesh.export(os.path.join(data_dir, 'lattice.stl'))
fe.utils.save_sol(
    mesh=band.mesh,
    sol_file=os.path.join(data_dir, 'vtk/lattice_band_u.vtu'),
    point_infos=[('displacement', u)])
print("wrote", os.path.join(data_dir, 'vtk/lattice_band_u.vtu'))
