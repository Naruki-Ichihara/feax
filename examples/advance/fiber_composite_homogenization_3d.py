"""3D fiber-reinforced composite homogenization example.

Computes the full 6x6 homogenized stiffness tensor of a unidirectional
carbon-fiber / epoxy composite using periodic boundary conditions.

Unit cell:
  - 10 um (x, fiber dir) x 50 um (y) x 50 um (z)
  - Carbon fibers: 7 um diameter, ~50% volume fraction
  - 2D cross-section mesh (yz plane, gmsh, periodic) extruded in x -> TET4
  - Periodic boundary conditions on all 6 faces

Material properties (isotropic for simplicity):
  - Carbon fiber: E = 230 GPa, nu = 0.2
  - Epoxy matrix:  E = 3.5 GPa, nu = 0.35
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
L = 50.0            # Unit cell side length [um]
H = 5.0            # Extrusion thickness [um]
r_fiber = 3.5       # Fiber radius [um]
Vf_target = 0.50    # Target volume fraction
n_layers = 10        # Number of extrusion layers in z

# Material properties (isotropic)
E_fiber = 230.0e3   # [MPa] (230 GPa)
nu_fiber = 0.2
E_matrix = 3.5e3    # [MPa] (3.5 GPa)
nu_matrix = 0.35

# Mesh size
mesh_size_fiber = 1.0    # [um] at fiber interface
mesh_size_matrix = 0.5   # [um] in matrix


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
    """Generate randomly perturbed hex-packed fibers with periodic BC."""
    rng = onp.random.default_rng(seed)
    min_dist = 2 * r + min_gap

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

    max_shift = perturb_fraction * s_x
    for _ in range(5):
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

    all_centers = []
    for cx, cy in base:
        for dx in [-L, 0, L]:
            for dy in [-L, 0, L]:
                nx, ny = cx + dx, cy + dy
                if (nx + r > 0 and nx - r < L and
                    ny + r > 0 and ny - r < L):
                    all_centers.append((nx, ny))

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
# Mesh generation: 2D gmsh (periodic) + manual z-extrusion
# ============================================================================
def create_mesh_2d(centers, r, L, mesh_size_f, mesh_size_m):
    """Create a 2D periodic mesh with conforming fiber-matrix interfaces.

    Same approach as the 2D example: gmsh OCC fragment + setPeriodic on curves.
    Returns 2D points and triangle connectivity.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("fiber_composite_2d")
    factory = gmsh.model.occ

    rect = factory.addRectangle(0, 0, 0, L, L)
    fiber_disk_tags = []
    for cx, cy in centers:
        fiber_disk_tags.append(factory.addDisk(cx, cy, 0, r, r))

    all_objects = [(2, rect)]
    all_tools = [(2, t) for t in fiber_disk_tags]
    out_dimtags, out_map = factory.fragment(all_objects, all_tools)
    factory.synchronize()

    rect_fragment_tags = {tag for dim, tag in out_map[0] if dim == 2}
    to_remove = [(d, t) for d, t in out_dimtags
                 if d == 2 and t not in rect_fragment_tags]
    if to_remove:
        factory.remove(to_remove, recursive=True)
        factory.synchronize()

    fiber_surfaces = []
    matrix_surfaces = []
    for tag in rect_fragment_tags:
        xc, yc, _ = factory.getCenterOfMass(2, tag)
        is_fiber = any(
            onp.sqrt((xc - cx)**2 + (yc - cy)**2) < r - 0.1
            for cx, cy in centers
        )
        (fiber_surfaces if is_fiber else matrix_surfaces).append(tag)

    if fiber_surfaces:
        gmsh.model.addPhysicalGroup(2, fiber_surfaces, tag=1, name="fiber")
    if matrix_surfaces:
        gmsh.model.addPhysicalGroup(2, matrix_surfaces, tag=2, name="matrix")

    # Periodic curve matching (left<->right, bottom<->top)
    all_surfs = [(2, t) for t in fiber_surfaces + matrix_surfaces]
    boundary = gmsh.model.getBoundary(all_surfs, combined=True, oriented=False)
    bnd_curve_tags = list({abs(t) for _, t in boundary})

    left, right, bottom, top = [], [], [], []
    tol_bnd = 0.5
    for ct in bnd_curve_tags:
        bb = factory.getBoundingBox(1, ct)
        xmin, ymin, _, xmax, ymax, _ = bb
        if abs(xmin) < tol_bnd and abs(xmax) < tol_bnd:
            left.append(ct)
        elif abs(xmin - L) < tol_bnd and abs(xmax - L) < tol_bnd:
            right.append(ct)
        elif abs(ymin) < tol_bnd and abs(ymax) < tol_bnd:
            bottom.append(ct)
        elif abs(ymin - L) < tol_bnd and abs(ymax - L) < tol_bnd:
            top.append(ct)

    print(f"    Boundary curves: left={len(left)}, right={len(right)}, "
          f"bottom={len(bottom)}, top={len(top)}")

    def match_curves(masters, slaves, translation):
        """Match master/slave curve pairs by translated centroid."""
        matched_m, matched_s = [], []
        used = set()
        for mt in masters:
            mc = onp.array(factory.getCenterOfMass(1, mt))
            expected = mc + translation
            best_d, best_ct = float('inf'), None
            for ct in slaves:
                if ct in used:
                    continue
                sc = onp.array(factory.getCenterOfMass(1, ct))
                d = onp.linalg.norm(sc - expected)
                if d < best_d:
                    best_d, best_ct = d, ct
            if best_ct is not None and best_d < 1.0:
                matched_m.append(mt)
                matched_s.append(best_ct)
                used.add(best_ct)
        return matched_m, matched_s

    tx = [1, 0, 0, L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ty = [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1]
    if left and right:
        m, s = match_curves(left, right, onp.array([L, 0, 0]))
        print(f"    x-periodic: {len(m)} matched curve pairs")
        if m:
            gmsh.model.mesh.setPeriodic(1, s, m, tx)
    if bottom and top:
        m, s = match_curves(bottom, top, onp.array([0, L, 0]))
        print(f"    y-periodic: {len(m)} matched curve pairs")
        if m:
            gmsh.model.mesh.setPeriodic(1, s, m, ty)

    # Mesh size field
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

    gmsh.model.mesh.generate(2)

    msh_path = "/tmp/fiber_composite_2d_for_extrude.msh"
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    all_tri = [cb.data for cb in msh.cells if cb.type == "triangle"]
    if not all_tri:
        raise RuntimeError("No triangle elements found in gmsh output")

    points_2d = msh.points[:, :2]
    return points_2d, onp.vstack(all_tri)


def extrude_to_3d(points_2d, cells_2d, H, n_layers):
    """Extrude 2D cross-section mesh along x (fiber direction) into TET4.

    2D points (p0, p1) become (y, z) coordinates in 3D.
    Extrusion adds x in [0, H] as the first coordinate.
    Each triangular prism is split into 3 tetrahedra.
    """
    n_pts = len(points_2d)
    dx = H / n_layers

    # Create 3D points: (x, y, z) = (extrusion, 2d_col0, 2d_col1)
    pts_layers = []
    for k in range(n_layers + 1):
        x = k * dx
        pts_layers.append(onp.column_stack([
            onp.full(n_pts, x), points_2d
        ]))
    points_3d = onp.vstack(pts_layers)

    # Split each prism into 3 tets
    # Bottom: (n0, n1, n2), Top: (n3, n4, n5) directly above
    # Decomposition: (n0,n1,n2,n3), (n1,n2,n3,n4), (n2,n3,n4,n5)
    tets = []
    for layer in range(n_layers):
        bot = layer * n_pts
        top = (layer + 1) * n_pts
        for a, b, c in cells_2d:
            n0, n1, n2 = a + bot, b + bot, c + bot
            n3, n4, n5 = a + top, b + top, c + top
            tets.append([n0, n1, n2, n3])
            tets.append([n1, n2, n3, n4])
            tets.append([n2, n3, n4, n5])
    tets = onp.array(tets)

    # Verify orientation: flip tets with negative volume
    v01 = points_3d[tets[:, 1]] - points_3d[tets[:, 0]]
    v02 = points_3d[tets[:, 2]] - points_3d[tets[:, 0]]
    v03 = points_3d[tets[:, 3]] - points_3d[tets[:, 0]]
    vols = onp.einsum('ij,ij->i', v01, onp.cross(v02, v03))
    neg = vols < 0
    if onp.any(neg):
        tets[neg] = tets[neg][:, [0, 2, 1, 3]]

    return points_3d, tets


# ============================================================================
# Problem definition
# ============================================================================
class LinearElasticity3D(fe.problem.Problem):
    """3D isotropic linear elasticity with per-element material properties."""

    def get_tensor_map(self):
        def stress(u_grad, E, nu_val):
            mu = E / (2.0 * (1.0 + nu_val))
            lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(3) + 2.0 * mu * epsilon
        return stress


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("3D Fiber-Reinforced Composite Homogenization")
    print("=" * 60)

    # 1. Generate fiber positions
    print("\n[1] Generating random fiber arrangement (periodic)...")
    centers, Vf, n_fibers = generate_random_fibers(L, r_fiber, Vf_target)
    print(f"    Primary fibers: {n_fibers}")
    print(f"    Total (incl. periodic copies): {len(centers)}")
    print(f"    Volume fraction: {Vf:.1%}")

    # 2. Create 2D mesh, then extrude to 3D
    print("\n[2] Creating mesh...")
    print("    Generating 2D periodic mesh with gmsh...")
    points_2d, cells_2d = create_mesh_2d(
        centers, r_fiber, L, mesh_size_fiber, mesh_size_matrix
    )
    print(f"    2D mesh: {len(points_2d)} nodes, {len(cells_2d)} triangles")

    print(f"    Extruding along x ({n_layers} layers, dx={H/n_layers:.2f} um)...")
    points_3d, cells_3d = extrude_to_3d(points_2d, cells_2d, H, n_layers)

    mesh = fe.Mesh(points_3d, cells_3d, ele_type="TET4")
    print(f"    3D mesh: {len(mesh.points)} nodes, {len(mesh.cells)} elements")

    # 3. Assign material properties per element
    print("\n[3] Assigning material properties...")
    cell_centers = onp.mean(
        onp.array(mesh.points)[onp.array(mesh.cells)], axis=1
    )

    is_fiber = onp.zeros(len(mesh.cells), dtype=bool)
    for cx, cy in centers:
        # Cross-section is in yz plane (columns 1, 2)
        dist_yz = onp.sqrt(
            (cell_centers[:, 1] - cx)**2 + (cell_centers[:, 2] - cy)**2
        )
        is_fiber |= dist_yz < r_fiber

    E_field = onp.where(is_fiber, E_fiber, E_matrix)
    nu_field = onp.where(is_fiber, nu_fiber, nu_matrix)

    n_fiber_cells = onp.sum(is_fiber)
    print(f"    Fiber elements: {n_fiber_cells}")
    print(f"    Matrix elements: {len(mesh.cells) - n_fiber_cells}")
    print(f"    Element Vf: {n_fiber_cells / len(mesh.cells):.1%}")

    # 4. Set up FEAX problem
    print("\n[4] Setting up FEAX problem...")
    problem = LinearElasticity3D(
        mesh=mesh, vec=3, dim=3, ele_type="TET4", location_fns=[]
    )

    E_var = fe.InternalVars.create_cell_var(problem, E_field)
    nu_var = fe.InternalVars.create_cell_var(problem, nu_field)
    internal_vars = fe.InternalVars(
        volume_vars=(E_var, nu_var), surface_vars=()
    )

    # 5. Periodic boundary conditions
    print("\n[5] Setting up periodic boundary conditions...")

    class FiberUnitCell3D(flat.unitcell.UnitCell):
        """Unit cell wrapping the extruded 3D mesh."""
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

    unitcell = FiberUnitCell3D(mesh)
    print(f"    Unit cell bounds: {unitcell.lb} -> {unitcell.ub}")

    pairings = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
    P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)
    print(f"    Prolongation matrix: {P.shape}")
    print(f"    DoF reduction: {P.shape[0]} -> {P.shape[1]}")

    bc_config = fe.DirichletBCConfig([])
    bc = bc_config.create_bc(problem)

    # 6. Homogenization
    print("\n[6] Computing homogenized stiffness (6 strain cases)...")
    solver_options = fe.IterativeSolverOptions(
        solver="cg", tol=1e-10, atol=1e-10, maxiter=10000, verbose=True
    )

    solve = flat.solver.create_homogenization_solver(
        problem, bc, P, mesh, solver_options=solver_options, dim=3
    )
    result = solve(internal_vars)
    C_hom = result.C_hom
    u_totals = result.u_totals
    labels = solve.labels
    n_cases = len(labels)

    print(f"\n    Homogenized stiffness matrix C_hom (6x6, Voigt):")
    for i in range(6):
        row = "    " + "  ".join(f"{C_hom[i, j]:10.1f}" for j in range(6))
        print(row)

    # Engineering constants
    S = np.linalg.inv(C_hom)
    E_1 = 1.0 / S[0, 0]
    E_2 = 1.0 / S[1, 1]
    E_3 = 1.0 / S[2, 2]
    nu_12 = -S[0, 1] / S[0, 0]
    nu_13 = -S[0, 2] / S[0, 0]
    nu_23 = -S[1, 2] / S[1, 1]
    G_12 = 1.0 / S[5, 5]
    G_13 = 1.0 / S[4, 4]
    G_23 = 1.0 / S[3, 3]

    print(f"\n    Effective engineering constants:")
    print(f"    E_1  (fiber dir)  = {E_1/1e3:.2f} GPa")
    print(f"    E_2  (transverse) = {E_2/1e3:.2f} GPa")
    print(f"    E_3  (transverse) = {E_3/1e3:.2f} GPa")
    print(f"    nu_12 = {nu_12:.4f}")
    print(f"    nu_13 = {nu_13:.4f}")
    print(f"    nu_23 = {nu_23:.4f}")
    print(f"    G_12 = {G_12/1e3:.2f} GPa")
    print(f"    G_13 = {G_13/1e3:.2f} GPa")
    print(f"    G_23 = {G_23/1e3:.2f} GPa")

    E_rom = Vf * E_fiber + (1 - Vf) * E_matrix
    print(f"\n    Rule of mixtures E_1 (Voigt): {E_rom/1e3:.2f} GPa")

    # 7. Compute per-element stress fields and save
    print("\n[7] Computing stress distributions and saving...")
    output_dir = os.path.join(os.path.dirname(__file__), "data", "vtk")
    os.makedirs(output_dir, exist_ok=True)

    tensor_map = problem.get_tensor_map()
    shape_grads = problem.shape_grads
    cells_arr = problem.cells_list[0]

    num_cells = len(cells_arr)
    num_quads = shape_grads.shape[1]

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
        """Compute average stress per cell: (num_cells, 3, 3)."""
        sol_list = problem.unflatten_fn_sol_list(u_total)
        cell_sol = sol_list[0][cells_arr]
        u_grads = np.einsum('cqnd,cnv->cqvd', shape_grads, cell_sol)

        def cell_avg_stress(u_grads_c, *vars_c):
            stresses = jax.vmap(tensor_map)(u_grads_c, *vars_c)
            return np.mean(stresses, axis=0)

        return jax.vmap(cell_avg_stress)(u_grads, *vol_vars_quad)

    material_id = onp.where(is_fiber, 1.0, 0.0).astype(onp.float64)

    for k in range(n_cases):
        print(f"    Processing {labels[k]}...")
        sigma_cells = compute_cell_stress(u_totals[k])  # (num_cells, 3, 3)

        s11_c = onp.array(sigma_cells[:, 0, 0])
        s22_c = onp.array(sigma_cells[:, 1, 1])
        s33_c = onp.array(sigma_cells[:, 2, 2])
        s23_c = onp.array(sigma_cells[:, 1, 2])
        s13_c = onp.array(sigma_cells[:, 0, 2])
        s12_c = onp.array(sigma_cells[:, 0, 1])
        vm_c = onp.sqrt(0.5 * ((s11_c - s22_c)**2 + (s22_c - s33_c)**2 +
                                (s33_c - s11_c)**2) +
                        3.0 * (s12_c**2 + s23_c**2 + s13_c**2))

        u_arr = onp.array(u_totals[k]).reshape(-1, 3)

        vtk_file = os.path.join(output_dir, f"fiber_composite_3d_{labels[k]}.vtu")
        fe.utils.save_sol(
            mesh=mesh,
            sol_file=vtk_file,
            point_infos=[("displacement", u_arr)],
            cell_infos=[
                ("material", material_id),
                ("sigma_11", s11_c),
                ("sigma_22", s22_c),
                ("sigma_33", s33_c),
                ("sigma_23", s23_c),
                ("sigma_13", s13_c),
                ("sigma_12", s12_c),
                ("von_mises", vm_c),
            ],
        )
        print(f"    Saved: {vtk_file}")

    print("\n" + "=" * 60)
    print("3D Homogenization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
