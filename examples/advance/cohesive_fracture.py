"""Cohesive zone fracture in 3D (Mode I) — hybrid matrix-free Newton-Krylov.

Reproduces the tatva quasi-static 3D fracture example using feax:
  - Bulk: 3D linear elasticity (HEX8); feax assembles the residual / bulk
    Jacobian from ``get_energy_density`` (tensor_map = ∂ψ/∂∇u).
  - Interface: exponential cohesive law (pure JAX), supplied as an
    ``extra_residual_fn`` (tangent applied matrix-free via ``jax.jvp``).
  - Solver: ``feax.create_solver(..., extra_residual_fn=...)`` with
    ``KrylovSolverOptions`` (sparse bulk Jacobian + JVP cohesive tangent).

The mesh consists of two half-blocks (top/bottom) separated at y=0.
A pre-crack extends from x=0 to x=crack_length (free surfaces).
Mode I loading is applied via prescribed displacement at top/bottom faces.

Parameters match the tatva example:
  E = 106e3 Pa, nu = 0.35, Gamma = 15 J/m², sigma_c = 20e3 Pa
  Geometry scaled by Griffith length L_G = 2*mu*Gamma / (pi*(1-nu)*sigma_inf^2)
"""

import logging
import os

import jax
import jax.numpy as np
import numpy as onp

import feax as fe
from feax.mechanics.cohesive import (
    CohesiveInterface,
    compute_lumped_area_weights,
    exponential_potential,
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
def generate_structured_hex_fracture_3d(
    length, height, thickness, crack_tip_x, mesh_size_tip, mesh_size_far,
):
    """Generate a 3D fracture mesh with structured hexahedra (pure numpy).

    Builds the top half-block ``0 <= y <= height/2`` as a structured HEX8 grid,
    graded finer along x in the cracked region ``[crack_tip_x, length]``, then
    mirrors it about ``y=0`` to create the bottom half. The ``y=0`` face quads
    form the cohesive interface. No external mesh generator is required.

    Returns the same tuple as the former Gmsh-based generator:
    ``(coords, elements, top_interface_nodes, bottom_interface_nodes,
    active_quads_top, active_quads_bottom)``.
    """
    h_half = height / 2.0

    def graded_axis(segments):
        """1D node positions from (start, end, target_h) segments (joints deduped)."""
        nodes = [segments[0][0]]
        for (a, b, h) in segments:
            n_el = max(1, int(round((b - a) / h)))
            nodes.extend(onp.linspace(a, b, n_el + 1)[1:].tolist())
        return onp.asarray(nodes, dtype=onp.float64)

    # Fine spacing in the cracked span [crack_tip_x, length], coarse ahead of it.
    x_nodes = graded_axis([
        (0.0, crack_tip_x, mesh_size_far),
        (crack_tip_x, length, mesh_size_tip),
    ])
    y_nodes = graded_axis([(0.0, h_half, mesh_size_far)])
    z_nodes = graded_axis([(0.0, thickness, mesh_size_far)])
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)

    X, Y, Z = onp.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    points_top = onp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Structured HEX8 connectivity in meshio/VTK hexahedron order.
    # Node id (i, j, k) = (i * ny + j) * nz + k  (i->x, j->y, k->z).
    ei, ej, ek = onp.meshgrid(
        onp.arange(nx - 1), onp.arange(ny - 1), onp.arange(nz - 1), indexing='ij',
    )
    ei, ej, ek = ei.ravel(), ej.ravel(), ek.ravel()

    def nod(di, dj, dk):
        return ((ei + di) * ny + (ej + dj)) * nz + (ek + dk)

    hex_top = onp.stack([
        nod(0, 0, 0), nod(1, 0, 0), nod(1, 1, 0), nod(0, 1, 0),
        nod(0, 0, 1), nod(1, 0, 1), nod(1, 1, 1), nod(0, 1, 1),
    ], axis=1).astype(onp.int64)

    # Interface (y=0, j=0) face quads, in CCW meshio quad order.
    qi, qk = onp.meshgrid(onp.arange(nx - 1), onp.arange(nz - 1), indexing='ij')
    qi, qk = qi.ravel(), qk.ravel()
    all_interface_quads_top = onp.stack([
        (qi * ny + 0) * nz + qk,
        ((qi + 1) * ny + 0) * nz + qk,
        ((qi + 1) * ny + 0) * nz + (qk + 1),
        (qi * ny + 0) * nz + (qk + 1),
    ], axis=1).astype(onp.int64)

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
    generate_structured_hex_fracture_3d(
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
class Elasticity3D(fe.Problem):
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
# Energy functions and cohesive residual
# ============================================================
# The bulk elastic residual is assembled by feax from get_energy_density.
# The cohesive interface adds r_coh = ∂Φ_coh/∂u, supplied to the solver as
# `extra_residual_fn` (its tangent is applied matrix-free via jax.jvp).
# The scalar energies are retained for force / energy post-processing.
elastic_energy = fe.create_energy_fn(problem)
cohesive_energy = interface.create_energy_fn(
    exponential_potential, Gamma=Gamma, sigma_c=sigma_c,
)


def total_energy(u_flat, delta_max):
    """Total potential energy = elastic + cohesive (post-processing only)."""
    return elastic_energy(u_flat) + cohesive_energy(u_flat, delta_max)


def cohesive_residual(u_flat, delta_max):
    """Cohesive residual = ∂Φ_coh/∂u at fixed history δ_max."""
    return jax.grad(lambda u: cohesive_energy(u, delta_max))(u_flat)


# ============================================================
# Boundary conditions (matching tatva example)
# ============================================================
def make_bc(disp):
    """Create BC for given displacement magnitude (half applied to each face)."""
    specs = [
        # Top face: all DOFs fixed, y = +disp/2
        fe.DirichletBCSpec(top_face, 'x', 0.0),
        fe.DirichletBCSpec(top_face, 'y', disp / 2),
        fe.DirichletBCSpec(top_face, 'z', 0.0),
        # Bottom face: all DOFs fixed, y = -disp/2
        fe.DirichletBCSpec(bottom_face, 'x', 0.0),
        fe.DirichletBCSpec(bottom_face, 'y', -disp / 2),
        fe.DirichletBCSpec(bottom_face, 'z', 0.0),
        # Left face: x fixed
        fe.DirichletBCSpec(left_face, 'x', 0.0),
        # Front/back faces: z fixed (plane strain constraint)
        fe.DirichletBCSpec(front_face, 'z', 0.0),
        fe.DirichletBCSpec(back_face, 'z', 0.0),
    ]
    return fe.DirichletBCConfig(specs).create_bc(problem)


# Solver (created once, reused for all steps). The cohesive history δ_max is a
# quasi-static state variable updated between load steps; it flows into the
# extra residual through a mutable holder so the solver need not be rebuilt.
bc0 = make_bc(0.0)
history = {'delta_max': np.zeros(interface.n_nodes)}

ts = fe.TracedStructure.from_problem(problem)

solver = fe.create_solver(
    problem, bc0,
    solver_options=fe.KrylovSolverOptions(
        solver='cg', atol=1e-8, maxiter=200, verbose=True,
    ),
    preconditioner=fe.jacobi(),
    newton_options=fe.NewtonOptions(tol=1e-8, max_iter=200),
    extra_residual_fn=lambda u: cohesive_residual(u, history['delta_max']),
    linear=False,
    traced_structure=ts,
)
EMPTY_IV = fe.TracedParams()  # bulk elasticity carries no internal variables


# ============================================================
# Output
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk_3d'), exist_ok=True)


def save_step(u_flat, delta_max_field, step):
    u = u_flat.field(0)          # u_flat is a fe.Solution
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
    history['delta_max'] = delta_max
    # .dofs: keep the carried guess a flat array (the solver returns a
    # Solution, which has no .at for the BC re-seed at the next step)
    u_flat = solver(EMPTY_IV, u_flat, bc=bc, traced_structure=ts).dofs

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
