"""Composite laminate comparison: LayeredSolid vs. Classical Laminate Theory.

Five symmetric, balanced laminate configurations are solved under uniaxial
tension and compared against CLT predictions for the effective axial modulus
E_x and Poisson ratio ν_xy.

Configurations
--------------
  [0]_8          — unidirectional 0°    (upper bound of E_x)
  [90]_8         — unidirectional 90°   (lower bound of E_x)
  [0/90]_2s      — cross-ply            (orthotropic)
  [±45]_2s       — angle-ply            (shear-dominated)
  [0/45/-45/90]_s — quasi-isotropic     (in-plane isotropic)

Material (CFRP UD, MPa)
-----------------------
  E1 = 140 000,  E2 = 10 000,  G12 = 5 000,  ν12 = 0.30,  ν23 = 0.40

Method
------
LayeredSolid uses per-ply Gauss sub-integration through the thickness so
that an arbitrary number of plies live inside a single HEX8 element without
approximation (K = Σ_k ∫_{z_k}^{z_{k+1}} Bᵀ C_k B dz, each interval
integrated exactly).

CLT gives the effective axial modulus from the in-plane compliance:

    E_x = 1 / (H · a_11),   a = A⁻¹  (Voigt 3×3)

References
----------
Reddy, J.N., *Mechanics of Laminated Composite Plates*, 2nd ed., CRC, 2004.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp

import feax as fe
from feax.assembler import get_res
from feax.mechanics.layered_solid_element import (
    transverse_isotropic_stiffness_3d,
    create_layered_solid,
)
from feax.mechanics.shell import (
    orthotropic_in_plane_stiffness,
    laminate_stiffness,
)

# ---------------------------------------------------------------------------
# Material (CFRP UD, MPa)
# ---------------------------------------------------------------------------
E1, E2, G12, nu12, nu23 = 140e3, 10e3, 5e3, 0.30, 0.40
ply_C = transverse_isotropic_stiffness_3d(E1, E2, G12, nu12, nu23)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
L, W, H = 200.0, 50.0, 1.0     # mm  (H = total laminate thickness)
delta_u = 1.0                   # prescribed x-displacement at right face
tol = 1e-6

# ---------------------------------------------------------------------------
# Laminate configurations  (ply angles in degrees, bottom → top)
# ---------------------------------------------------------------------------
deg = onp.pi / 180.0

LAMINATES = [
    ("[0]_8",           [0.0] * 8),
    ("[90]_8",          [90.0] * 8),
    ("[0/90]_2s",       [0.0, 90.0, 90.0, 0.0, 0.0, 90.0, 90.0, 0.0]),
    ("[±45]_2s",        [45.0, -45.0, -45.0, 45.0, 45.0, -45.0, -45.0, 45.0]),
    ("[0/45/-45/90]_s", [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0])
]

# ---------------------------------------------------------------------------
# Mesh (shared across all configurations)
# ---------------------------------------------------------------------------
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=10.0, element_type="HEX8")
pts  = onp.asarray(mesh.points)
left_mask = onp.isclose(pts[:, 0], 0.0, atol=tol)
far_mask  = onp.isclose(pts[:, 1], W,   atol=tol)

# ---------------------------------------------------------------------------
# CLT helper
# ---------------------------------------------------------------------------
C_in_2d  = orthotropic_in_plane_stiffness(E1, E2, G12, nu12)
G_shear  = onp.diag([G12, G12]).astype(onp.float64)

def clt_moduli(angles_rad, n_ply):
    t_ply = H / n_ply
    A, _, _, _ = laminate_stiffness(C_in_2d, G_shear, angles_rad, [t_ply] * n_ply)
    A_v = onp.array([
        [A[0,0,0,0], A[0,0,1,1], A[0,0,0,1]],
        [A[1,1,0,0], A[1,1,1,1], A[1,1,0,1]],
        [A[0,1,0,0], A[0,1,1,1], A[0,1,0,1]],
    ])
    a = onp.linalg.inv(A_v)
    return 1.0 / (H * a[0, 0]), -a[0, 1] / a[0, 0]    # E_x, ν_xy

# ---------------------------------------------------------------------------
# Boundary conditions (same for all laminates)
# ---------------------------------------------------------------------------
left  = lambda p: jnp.isclose(p[0], 0.0, atol=tol)
right = lambda p: jnp.isclose(p[0], L,   atol=tol)
front = lambda p: jnp.isclose(p[1], 0.0, atol=tol)
bot   = lambda p: jnp.isclose(p[2], 0.0, atol=tol)

bc_config = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=left,  component="x", value=0.0),
    fe.DirichletBCSpec(location=right, component="x", value=delta_u),
    fe.DirichletBCSpec(location=front, component="y", value=0.0),
    fe.DirichletBCSpec(location=bot,   component="z", value=0.0),
])

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = []

for name, angles_deg in LAMINATES:
    angles_rad = [a * deg for a in angles_deg]
    n_ply = len(angles_rad)
    t_ply = H / n_ply

    # CLT
    Ex_clt, nu_xy_clt = clt_moduli(angles_rad, n_ply)

    # FEM
    problem, tp = create_layered_solid(
        mesh,
        ply_C=ply_C,
        ply_angles=angles_rad,
        ply_thicknesses=[t_ply] * n_ply,
        n_inplane=2,
        n_thick_per_ply=2,
    )
    bc = bc_config.create_bc(problem)
    # get_res below is a no-TracedStructure assembly path, so keep host scratch.
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    solver = fe.create_solver(
        problem, bc,
        solver_options=fe.DirectSolverOptions(),
        linear=True,
        traced_params=tp,
        traced_structure=ts,
    )
    sol = solver(tp, traced_structure=ts)

    # Reaction force → E_x
    sol_list  = problem.unflatten_fn_sol_list(sol)
    res_list  = get_res(problem, sol_list, tp)
    res_array = onp.asarray(res_list[0])
    F_x       = float(res_array[left_mask, 0].sum())
    Ex_fem    = abs(F_x) / (W * H) / (delta_u / L)

    # Transverse contraction → ν_xy
    u_array   = onp.asarray(sol_list[0])
    uy_far    = float(u_array[far_mask, 1].mean())
    nu_xy_fem = -(uy_far / W) / (delta_u / L)

    err = 100.0 * abs(Ex_fem - Ex_clt) / Ex_clt
    results.append((name, Ex_clt, Ex_fem, err, nu_xy_clt, nu_xy_fem))

    # VTK per configuration
    data_dir = os.path.join(os.path.dirname(__file__), "data", "vtk")
    os.makedirs(data_dir, exist_ok=True)
    safe_name = name.replace("/", "_").replace("[", "").replace("]", "").replace(" ","")
    fe.utils.save_sol(
        mesh=mesh,
        sol_file=os.path.join(data_dir, f"laminate_{safe_name}.vtu"),
        point_infos=[("displacement", u_array)],
    )

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("  CFRP laminate comparison: LayeredSolid vs. CLT")
print("  E1={:.0f}, E2={:.0f}, G12={:.0f} MPa  |  L×W×H = {:.0f}×{:.0f}×{:.0f} mm".format(
    E1, E2, G12, L, W, H))
print("=" * 74)
print("  {:<20s}  {:>10s}  {:>10s}  {:>8s}  {:>7s}  {:>7s}".format(
    "Laminate", "E_x CLT", "E_x FEM", "error", "ν_xy CLT", "ν_xy FEM"))
print("  {:<20s}  {:>10s}  {:>10s}  {:>8s}  {:>7s}  {:>7s}".format(
    "", "[MPa]", "[MPa]", "", "", ""))
print("-" * 74)
for name, Ex_clt, Ex_fem, err, nu_clt, nu_fem in results:
    print("  {:<20s}  {:>10.1f}  {:>10.1f}  {:>7.2f}%  {:>7.4f}  {:>7.4f}".format(
        name, Ex_clt, Ex_fem, err, nu_clt, nu_fem))
print("=" * 74)
print()
print("VTK files saved to", os.path.join(os.path.dirname(__file__), "data", "vtk"))
