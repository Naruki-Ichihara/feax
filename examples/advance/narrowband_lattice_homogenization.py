"""Narrow-band with PERIODIC boundary conditions: BCC lattice homogenization.

Computes the homogenized stiffness C_hom of a BCC unit cell (density-field lattice)
under full periodicity, TWICE:

  * full : solve the whole unit cell (lattice = E_base, void = Emin)
  * band : solve only the active band (solid struts) via feax.NarrowBand, with the
           prolongation matrix rebuilt on the band mesh — the void is excluded

and checks that C_hom matches. Key points for narrow-band + PBC:

  1. Periodic pairings are coordinate predicates, so the SAME pairings build P on
     the band via ``prolongation_matrix(pairings, band.mesh, vec)`` — no new code.
  2. ``average_stress`` divides by the volume of the mesh it is given, so the band
     C_hom (divided by the solid volume) is rescaled by band_vol/full_vol to get
     the macroscopic C_hom (divided by the full cell volume). Void stress ~ 0, so
     the two agree.

Run:  python examples/advance/narrowband_lattice_homogenization.py
"""
import time

import numpy as onp
import jax
import jax.numpy as np

import feax as fe
import feax.flat as flat
from feax.flat import graph

jax.config.update("jax_enable_x64", True)

E_base, Emin, nu = 9026.0, 9026.0 * 1e-6, 0.3      # void floor keeps the system SPD
A = 3.0                                             # unit-cell length
MESH_SIZE = 0.2
RADIUS = 0.4


class LinearElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu_val):
            mu = E / (2.0 * (1.0 + nu_val))
            lmbda = E * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress


class BCCUnitCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=A, mesh_size=mesh_size, element_type='HEX8')


def cell_volume(problem):
    JxW = problem.JxW
    return float(JxW.reshape(JxW.shape[0], -1).sum())


def homogenize(problem, mesh, pairings, E_cell):
    """C_hom on the given (sub)problem + its own prolongation. Returns (C, volume, t)."""
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)
    nu_cell = fe.TracedParams.create_cell_var(problem, nu)
    tp = fe.TracedParams(volume_vars=(np.asarray(E_cell), nu_cell), surface_vars=())
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-9, atol=1e-12, maxiter=5000)
    solve = flat.solver.create_homogenization_solver(problem, bc, P, mesh, opts, dim=3)
    t0 = time.perf_counter()
    res = solve(tp)
    C = onp.asarray(res.C_hom)
    C.flatten()  # force
    jax.block_until_ready(res.C_hom)
    return C, cell_volume(problem), time.perf_counter() - t0


# --- unit cell + BCC density field -------------------------------------------
unitcell = BCCUnitCell(mesh_size=MESH_SIZE)
mesh = unitcell.mesh
full = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])
nc = full.num_cells

corners = onp.array([[i, j, k] for i in (0, A) for j in (0, A) for k in (0, A)], float)
center = onp.array([[A / 2, A / 2, A / 2]], float)
nodes = onp.vstack([corners, center])
edges = onp.array([[i, 8] for i in range(8)])
lattice_func = graph.create_lattice_function(np.asarray(nodes), np.asarray(edges), radius=RADIUS)
rho = onp.array(graph.create_lattice_density_field(full, lattice_func,
                                                   density_solid=1.0, density_void=0.0))
E_full = Emin + rho * (E_base - Emin)
solid_frac = float((rho > 0.5).mean())
print(f"BCC unit cell {A}^3, mesh_size={MESH_SIZE}: cells={nc}, "
      f"nodes={mesh.points.shape[0]}, solid fraction={solid_frac*100:.1f}%")

pairings = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)

# --- full homogenization -----------------------------------------------------
C_full, vol_full, t_full = homogenize(full, mesh, pairings, E_full)

# --- band homogenization -----------------------------------------------------
active = onp.nonzero(rho > 0.5)[0]
band = fe.NarrowBand(mesh, active)
band_pb = LinearElasticity(mesh=band.mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])
E_band = band.gather_cells(E_full)
try:
    C_band_raw, vol_band, t_band = homogenize(band_pb, band.mesh, pairings, E_band)
    C_band = C_band_raw * (vol_band / vol_full)        # rescale solid-vol -> cell-vol
    ok = True
except AssertionError as e:
    print(f"\n[band PBC] pairing mismatch: {e}")
    print("-> the band boundary is not periodic-consistent; retry with keep_cells "
          "= periodic boundary faces (see comments).")
    ok = False

# --- compare -----------------------------------------------------------------
if ok:
    def eng(C):
        C11, C12, C44 = C[0, 0], C[0, 1], C[3, 3]
        E_eff = (C11 - C12) * (C11 + 2 * C12) / (C11 + C12)
        return C11, C12, C44, E_eff
    relC = onp.linalg.norm(C_band - C_full) / onp.linalg.norm(C_full)
    print("\n=== homogenized stiffness: full vs narrow band (PBC) ===")
    print(f"{'':10}{'C11':>12}{'C12':>12}{'C44':>12}{'E_eff':>12}")
    print(f"{'full':10}" + "".join(f"{v:12.2f}" for v in eng(C_full)))
    print(f"{'band':10}" + "".join(f"{v:12.2f}" for v in eng(C_band)))
    print(f"\nC_hom relative Frobenius error (band vs full) = {relC:.2e}")
    print(f"band cells {band.num_active_cells}/{nc} ({band.num_active_cells/nc*100:.0f}%), "
          f"DOF {3*band.mesh.points.shape[0]}/{3*mesh.points.shape[0]}")
    print(f"solve time  full={t_full:.2f}s  band={t_band:.2f}s  speedup={t_full/t_band:.1f}x")
