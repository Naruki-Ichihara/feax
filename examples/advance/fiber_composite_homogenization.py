"""Fiber-reinforced composite homogenization example.

Computes the homogenized transverse stiffness tensor of a unidirectional
carbon-fiber / epoxy composite using periodic boundary conditions.

Unit cell:
  - 100 μm × 100 μm square cross-section (fibers run along z)
  - Carbon fibers: 7 μm diameter, straight cylinders, ~60% volume fraction
  - Hex-packed fiber arrangement with conforming mesh at fiber-matrix interfaces
  - Plane-strain 2D analysis (transverse cross-section)

Material properties (transverse):
  - Carbon fiber: E = 15 GPa, ν = 0.2
  - Epoxy matrix:  E = 3.5 GPa, ν = 0.35
"""

import os

import gmsh
import jax
import jax.numpy as np
import meshio
import numpy as onp

import feax as fe
import feax.flat as flat


# ============================================================================
# Parameters
# ============================================================================
L = 100.0           # Unit cell side length [μm]
r_fiber = 3.5       # Fiber radius [μm]
d_fiber = 7.0       # Fiber diameter [μm]
Vf_target = 0.50    # Target volume fraction

# Material properties (transverse)
E_fiber = 15.0e3    # [MPa] (15 GPa)
nu_fiber = 0.2
E_matrix = 3.5e3    # [MPa] (3.5 GPa)
nu_matrix = 0.35

# Mesh size
mesh_size_fiber = 1.0    # [μm] at fiber interface
mesh_size_matrix = 0.5   # [μm] in matrix


# ============================================================================
# Random fiber placement (periodic)
# ============================================================================
def _periodic_distance(c1, c2, L):
    """Minimum image distance under periodic boundary conditions."""
    d = onp.abs(c1 - c2)
    d = onp.where(d > L / 2, L - d, d)
    return onp.linalg.norm(d)


def generate_random_fibers(L, r, Vf_target, min_gap=0.3, seed=42,
                           perturb_fraction=0.35):
    """Generate randomly perturbed hex-packed fibers with periodic BC.

    Starts from a perfect hex lattice that achieves the target Vf, then
    applies random perturbations to each center while respecting
    minimum distance constraints.  This reliably reaches high Vf.

    Parameters
    ----------
    L : float             Unit cell side length.
    r : float             Fiber radius.
    Vf_target : float     Target volume fraction.
    min_gap : float       Minimum gap between fiber surfaces (default 0.3 μm).
    seed : int            Random seed for reproducibility.
    perturb_fraction : float
        Maximum perturbation as fraction of the hex spacing.

    Returns
    -------
    all_centers : ndarray  All fiber centers (including periodic images).
    Vf_actual : float      Achieved volume fraction.
    n_fibers : int         Number of primary fibers placed.
    """
    rng = onp.random.default_rng(seed)
    min_dist = 2 * r + min_gap

    # Build hex lattice with the target Vf
    s = onp.sqrt(2 * onp.pi * r**2 / (onp.sqrt(3) * Vf_target))
    n_cols = max(1, round(L / s))
    s_x = L / n_cols
    n_rows = max(2, round(L / (s_x * onp.sqrt(3) / 2)))
    if n_rows % 2 != 0:
        n_rows += 1
    s_y = L / n_rows

    base = []
    for j in range(n_rows):
        x_off = s_x / 2 if j % 2 == 1 else 0.0
        for i in range(n_cols):
            base.append([i * s_x + x_off, j * s_y])
    base = onp.array(base)

    # Perturb each center randomly, keep periodic minimum distance
    max_shift = perturb_fraction * s_x
    for _ in range(5):  # multiple passes for better mixing
        for idx in range(len(base)):
            dx, dy = rng.uniform(-max_shift, max_shift, size=2)
            candidate = onp.array([(base[idx, 0] + dx) % L,
                                   (base[idx, 1] + dy) % L])
            ok = True
            for jdx in range(len(base)):
                if jdx == idx:
                    continue
                if _periodic_distance(candidate, base[jdx], L) < min_dist:
                    ok = False
                    break
            if ok:
                base[idx] = candidate

    n_fibers = len(base)

    # Periodic copies for fibers crossing the boundary
    all_centers = []
    for cx, cy in base:
        for dx in [-L, 0, L]:
            for dy in [-L, 0, L]:
                nx, ny = cx + dx, cy + dy
                if (nx + r > 0 and nx - r < L and
                    ny + r > 0 and ny - r < L):
                    all_centers.append((nx, ny))

    # Deduplicate
    unique = [all_centers[0]]
    for c in all_centers[1:]:
        if onp.min(onp.linalg.norm(onp.array(unique) - c, axis=1)) > 0.01:
            unique.append(c)
    all_centers = onp.array(unique)

    Vf_actual = _compute_vf(all_centers, r, L)
    return all_centers, Vf_actual, n_fibers


def _compute_vf(centers, r, L):
    """Compute fiber volume fraction by pixel counting."""
    N = 500
    xs = onp.linspace(0, L, N)
    ys = onp.linspace(0, L, N)
    xx, yy = onp.meshgrid(xs, ys)
    inside = onp.zeros_like(xx, dtype=bool)
    for cx, cy in centers:
        inside |= (xx - cx)**2 + (yy - cy)**2 <= r**2
    return onp.mean(inside)


# ============================================================================
# Gmsh mesh generation
# ============================================================================
def create_mesh_gmsh(centers, r, L, mesh_size_f, mesh_size_m):
    """Create a 2D periodic mesh with conforming fiber-matrix interfaces.

    Uses gmsh OCC fragment to split the rectangle and disks, keeps only
    the rectangle fragments, and enforces periodic meshing via setPeriodic.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("fiber_composite")
    factory = gmsh.model.occ

    # Unit cell rectangle
    rect = factory.addRectangle(0, 0, 0, L, L)

    # Fiber disks
    fiber_disk_tags = []
    for cx, cy in centers:
        fiber_disk_tags.append(factory.addDisk(cx, cy, 0, r, r))

    # Fragment: rectangle × all disks → conforming interfaces
    all_objects = [(2, rect)]
    all_tools = [(2, t) for t in fiber_disk_tags]
    out_dimtags, out_map = factory.fragment(all_objects, all_tools)
    factory.synchronize()

    # ------------------------------------------------------------------
    # out_map[0] = fragments of the original rectangle → inside [0,L]^2
    # Everything else = disk pieces outside the rectangle → remove
    # ------------------------------------------------------------------
    rect_fragment_tags = {tag for dim, tag in out_map[0] if dim == 2}
    to_remove = [(d, t) for d, t in out_dimtags
                 if d == 2 and t not in rect_fragment_tags]
    if to_remove:
        factory.remove(to_remove, recursive=True)
        factory.synchronize()

    # Classify remaining surfaces as fiber or matrix
    fiber_surfaces = []
    matrix_surfaces = []
    for tag in rect_fragment_tags:
        xc, yc, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        is_fiber = any(
            onp.sqrt((xc - cx)**2 + (yc - cy)**2) < r - 0.1
            for cx, cy in centers
        )
        (fiber_surfaces if is_fiber else matrix_surfaces).append(tag)

    # Physical groups
    if fiber_surfaces:
        gmsh.model.addPhysicalGroup(2, fiber_surfaces, tag=1, name="fiber")
    if matrix_surfaces:
        gmsh.model.addPhysicalGroup(2, matrix_surfaces, tag=2, name="matrix")

    # ------------------------------------------------------------------
    # Periodic meshing: match left↔right and bottom↔top boundary curves
    # ------------------------------------------------------------------
    all_surfs = [(2, t) for t in fiber_surfaces + matrix_surfaces]
    boundary = gmsh.model.getBoundary(all_surfs, combined=True, oriented=False)
    bnd_curve_tags = list({abs(t) for _, t in boundary})

    left, right, bottom, top = [], [], [], []
    tol_bnd = 0.5
    for ct in bnd_curve_tags:
        bb = gmsh.model.occ.getBoundingBox(1, ct)  # xmin,ymin,zmin,xmax,ymax,zmax
        xmin, ymin, _, xmax, ymax, _ = bb
        if abs(xmin) < tol_bnd and abs(xmax) < tol_bnd:
            left.append(ct)
        elif abs(xmin - L) < tol_bnd and abs(xmax - L) < tol_bnd:
            right.append(ct)
        elif abs(ymin) < tol_bnd and abs(ymax) < tol_bnd:
            bottom.append(ct)
        elif abs(ymin - L) < tol_bnd and abs(ymax - L) < tol_bnd:
            top.append(ct)

    # Affine transforms (4×4 row-major): translation by (L,0,0) and (0,L,0)
    tx = [1, 0, 0, L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ty = [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1]

    if left and right:
        gmsh.model.mesh.setPeriodic(1, right, left, tx)
    if bottom and top:
        gmsh.model.mesh.setPeriodic(1, bottom, top, ty)

    # ------------------------------------------------------------------
    # Mesh size: fine at fiber interfaces, coarser in matrix
    # ------------------------------------------------------------------
    fiber_curves = set()
    for tag in fiber_surfaces:
        for _, ct in gmsh.model.getBoundary([(2, tag)], oriented=False):
            fiber_curves.add(abs(ct))

    if fiber_curves:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", list(fiber_curves))
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size_f)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size_m)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)
        gmsh.model.mesh.field.setNumber(2, "DistMax", r * 2)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Generate
    gmsh.model.mesh.generate(2)

    # Export → merge all triangle blocks
    msh_path = "/tmp/fiber_composite.msh"
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    all_tri = [cb.data for cb in msh.cells if cb.type == "triangle"]
    if not all_tri:
        raise RuntimeError("No triangle elements found in gmsh output")

    # Strip z-coordinate for 2D problem
    points_2d = msh.points[:, :2]
    return points_2d, onp.vstack(all_tri), fiber_surfaces, matrix_surfaces


# ============================================================================
# Problem definition
# ============================================================================
class PlaneStrainElasticity(fe.problem.Problem):
    """2D plane-strain linear elasticity with per-element material properties."""

    def get_tensor_map(self):
        def stress(u_grad, E, nu_val):
            mu = E / (2.0 * (1.0 + nu_val))
            lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
            epsilon = 0.5 * (u_grad + u_grad.T)
            # Plane strain: ε33=0, tr(ε)=ε11+ε22, σ in-plane 2x2
            dim = u_grad.shape[0]
            return lmbda * np.trace(epsilon) * np.eye(dim) + 2.0 * mu * epsilon
        return stress


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Fiber-Reinforced Composite Homogenization")
    print("=" * 60)

    # 1. Generate fiber positions
    print("\n[1] Generating random fiber arrangement (periodic RSA)...")
    centers, Vf, n_fibers = generate_random_fibers(L, r_fiber, Vf_target)
    print(f"    Primary fibers: {n_fibers}")
    print(f"    Total (incl. periodic copies): {len(centers)}")
    print(f"    Volume fraction: {Vf:.1%}")

    # 2. Create mesh with gmsh
    print("\n[2] Creating conforming mesh with gmsh...")
    points, cells, fiber_surf_tags, matrix_surf_tags = create_mesh_gmsh(
        centers, r_fiber, L, mesh_size_fiber, mesh_size_matrix
    )

    mesh = fe.Mesh(points, cells, ele_type="TRI3")
    print(f"    Nodes: {len(mesh.points)}")
    print(f"    Elements: {len(mesh.cells)}")

    # 3. Assign material properties per element
    print("\n[3] Assigning material properties...")
    cell_centers = onp.mean(
        onp.array(mesh.points)[onp.array(mesh.cells)], axis=1
    )

    is_fiber = onp.zeros(len(mesh.cells), dtype=bool)
    for cx, cy in centers:
        dist = onp.sqrt(
            (cell_centers[:, 0] - cx)**2 + (cell_centers[:, 1] - cy)**2
        )
        is_fiber |= dist < r_fiber

    E_field = onp.where(is_fiber, E_fiber, E_matrix)
    nu_field = onp.where(is_fiber, nu_fiber, nu_matrix)

    n_fiber_cells = onp.sum(is_fiber)
    print(f"    Fiber elements: {n_fiber_cells}")
    print(f"    Matrix elements: {len(mesh.cells) - n_fiber_cells}")
    print(f"    Element Vf: {n_fiber_cells / len(mesh.cells):.1%}")

    # 4. Set up FEAX problem
    print("\n[4] Setting up FEAX problem...")
    problem = PlaneStrainElasticity(
        mesh=mesh, vec=2, dim=2, ele_type="TRI3", location_fns=[]
    )

    E_var = fe.InternalVars.create_cell_var(problem, E_field)
    nu_var = fe.InternalVars.create_cell_var(problem, nu_field)
    internal_vars = fe.InternalVars(
        volume_vars=(E_var, nu_var), surface_vars=()
    )

    # 5. Periodic boundary conditions
    print("\n[5] Setting up periodic boundary conditions...")

    class FiberUnitCell(flat.unitcell.UnitCell):
        """Unit cell wrapping the gmsh-generated mesh."""
        def __init__(self, mesh_obj):
            self.mesh = mesh_obj
            self.cells = mesh_obj.cells
            self.ele_type = mesh_obj.ele_type
            self.points = mesh_obj.points
            self.atol = 1e-2
            self.lb = np.min(self.points, axis=0)
            self.ub = np.max(self.points, axis=0)
            self.num_dims = self.points.shape[1]

        def mesh_build(self, **kwargs):
            return self.mesh

    unitcell = FiberUnitCell(mesh)
    print(f"    Unit cell bounds: {unitcell.lb} -> {unitcell.ub}")

    # Build 2D periodic pairings manually
    # (periodic_bc_3D doesn't support dim=2 with 2D points)
    from feax.flat.pbc import PeriodicPairing
    Lx, Ly = float(unitcell.ub[0] - unitcell.lb[0]), float(unitcell.ub[1] - unitcell.lb[1])
    origin = unitcell.lb

    pairings = []

    # Corner pairs: origin → 3 other corners
    for corner in [
        (float(origin[0] + Lx), float(origin[1])),
        (float(origin[0]),       float(origin[1] + Ly)),
        (float(origin[0] + Lx), float(origin[1] + Ly)),
    ]:
        mf = unitcell.corner_function(origin)
        sf = unitcell.corner_function(corner)
        mp = unitcell.mapping(mf, sf)
        for i in range(2):
            pairings.append(PeriodicPairing(mf, sf, mp, i))

    # Edge pairs (excluding corners): left↔right, bottom↔top
    for axis, length in [(0, Lx), (1, Ly)]:
        mf = unitcell.face_function(axis, float(origin[axis]), excluding_corner=True)
        sf = unitcell.face_function(axis, float(origin[axis] + length), excluding_corner=True)
        mp = unitcell.mapping(mf, sf)
        for i in range(2):
            pairings.append(PeriodicPairing(mf, sf, mp, i))
    P = flat.pbc.prolongation_matrix(pairings, mesh, vec=2)
    print(f"    Prolongation matrix: {P.shape}")
    print(f"    DoF reduction: {P.shape[0]} -> {P.shape[1]}")

    bc_config = fe.DirichletBCConfig([])
    bc = bc_config.create_bc(problem)

    # 6. Homogenization
    print("\n[6] Computing homogenized stiffness...")
    solver_options = fe.IterativeSolverOptions(
        solver="cg", tol=1e-10, atol=1e-10, maxiter=10000, verbose=True
    )

    solve = flat.solver.create_homogenization_solver(
        problem, bc, P, mesh, solver_options=solver_options, dim=2
    )
    result = solve(internal_vars)
    C_hom = result.C_hom
    u_totals = result.u_totals
    labels = result.labels
    print(f"\n    Homogenized stiffness matrix (3x3, Voigt):")
    print(f"    C11 = {C_hom[0, 0]:.1f} MPa")
    print(f"    C22 = {C_hom[1, 1]:.1f} MPa")
    print(f"    C12 = {C_hom[0, 1]:.1f} MPa")
    print(f"    C66 = {C_hom[2, 2]:.1f} MPa")

    S = np.linalg.inv(C_hom)
    E_11 = 1.0 / S[0, 0]
    E_22 = 1.0 / S[1, 1]
    nu_12 = -S[0, 1] / S[0, 0]
    G_12 = 1.0 / S[2, 2]

    print(f"\n    Effective transverse properties:")
    print(f"    E_11  = {E_11:.1f} MPa ({E_11/1e3:.2f} GPa)")
    print(f"    E_22  = {E_22:.1f} MPa ({E_22/1e3:.2f} GPa)")
    print(f"    nu_12 = {nu_12:.4f}")
    print(f"    G_12  = {G_12:.1f} MPa ({G_12/1e3:.2f} GPa)")

    E_rom = Vf * E_fiber + (1 - Vf) * E_matrix
    E_reuss = 1.0 / (Vf / E_fiber + (1 - Vf) / E_matrix)
    print(f"\n    Rule of mixtures E (upper bound): {E_rom:.1f} MPa ({E_rom/1e3:.2f} GPa)")
    print(f"    Inverse rule of mixtures E (lower bound): {E_reuss:.1f} MPa ({E_reuss/1e3:.2f} GPa)")

    # 7. Compute per-element stress fields and save
    print("\n[7] Computing stress distributions and saving...")
    output_dir = os.path.join(os.path.dirname(__file__), "data", "vtk")
    os.makedirs(output_dir, exist_ok=True)

    # Helper: compute element-centroid stress from displacement field
    tensor_map = problem.get_tensor_map()
    shape_grads = problem.shape_grads  # (num_cells, num_quads, nodes_per_cell, dim)
    cells_arr = problem.cells_list[0]

    num_cells = len(cells_arr)
    num_quads = shape_grads.shape[1]

    # Prepare per-quad internal vars
    shape_vals = problem.fes[0].shape_vals
    vol_vars_quad = []
    for var in internal_vars.volume_vars:
        if var.ndim == 1 and var.shape[0] == num_cells:
            vol_vars_quad.append(np.tile(var[:, None], (1, num_quads)))
        elif var.ndim == 1:
            var_cell = var[cells_arr]
            vol_vars_quad.append(np.einsum('qn,cn->cq', shape_vals, var_cell))
        else:
            vol_vars_quad.append(var)

    def compute_cell_stress(u_total):
        """Compute average stress per cell: (num_cells, vec, dim)."""
        sol_list = problem.unflatten_fn_sol_list(u_total)
        cell_sol = sol_list[0][cells_arr]  # (num_cells, nodes_per_cell, vec)
        u_grads = np.einsum('cqnd,cnv->cqvd', shape_grads, cell_sol)

        def cell_avg_stress(u_grads_c, *vars_c):
            stresses = jax.vmap(tensor_map)(u_grads_c, *vars_c)  # (num_quads, vec, dim)
            return np.mean(stresses, axis=0)  # (vec, dim)

        return jax.vmap(cell_avg_stress)(u_grads, *vol_vars_quad)  # (num_cells, vec, dim)

    material_id = onp.where(is_fiber, 1.0, 0.0).astype(onp.float64)

    for k in range(n_cases):
        print(f"    Processing {labels[k]}...")
        sigma_cells = compute_cell_stress(u_totals[k])  # (num_cells, 2, 2)

        # Cell-level Voigt components
        s11_c = onp.array(sigma_cells[:, 0, 0])
        s22_c = onp.array(sigma_cells[:, 1, 1])
        s12_c = onp.array(sigma_cells[:, 0, 1])
        vm_c = onp.sqrt(s11_c**2 + s22_c**2 - s11_c * s22_c + 3 * s12_c**2)

        # Displacement as point data
        u_arr = onp.array(u_totals[k]).reshape(-1, 2)

        vtk_file = os.path.join(output_dir, f"fiber_composite_{labels[k]}.vtu")
        fe.utils.save_sol(
            mesh=mesh,
            sol_file=vtk_file,
            point_infos=[("displacement", u_arr)],
            cell_infos=[
                ("material", material_id),
                ("sigma_11", s11_c),
                ("sigma_22", s22_c),
                ("sigma_12", s12_c),
                ("von_mises", vm_c),
            ],
        )
        print(f"    Saved: {vtk_file}")

    print("\n" + "=" * 60)
    print("Homogenization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
