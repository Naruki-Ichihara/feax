"""Time-evolving lattice homogenization: an exponax Swift-Hohenberg pattern,
stirred by a steady ABC flow, evolves inside a periodic cube; N_SNAP snapshots
of the developed pattern (consecutive times) are thresholded into solid/void
nodal densities and their C_hom are computed in fe.vmap batches of B designs of
the single-cell homogenization solver, CG preconditioned by one shared cuDSS factorization per
group of SHARE consecutive snapshots (Jacobi without cuDSS) -- the
batched_lattice_homogenization.py recipe with the strut graph replaced by a PDE.

    u_t = r u - (k_c + Δ)² u + c0 + c2 u² - u³ - w·∇u   (reaction-diffusion + advection)
    w   = A (sin z + cos y, sin x + cos z, sin y + cos x)   (ABC flow, periodic)

The linear operator selects the wavelength 2π/√k_c, so the PDE runs on a cube
of NWAVE wavelengths that is mapped onto the L x L x L unit cell. The constant
c0 biases the mean of u and thereby the fill fraction of the pattern. Solid is
{ u > U_THR }: a connected labyrinth whose walls the flow keeps re-arranging,
so consecutive snapshots are similar designs (share one factorization) while
the batch as a whole samples clearly different lattices.

Run:  python examples/advance/exponax_rda_lattice_homogenization.py
Needs: pip install exponax
"""
import time

import jax
import jax.numpy as jnp
import numpy as onp
import exponax as ex
from exponax.nonlin_fun import BaseNonlinearFun
from exponax.stepper.reaction import SwiftHohenberg

import feax as fe
import feax.flat as flat
from feax import gene
from feax.solvers.options import has_cudss

E_BASE, NU = 9026.0, 0.3
L = 3.0                 # unit cell edge
MESH = 0.15             # FE element size -> 20^3 HEX8
N_PDE = 40              # pseudo-spectral grid per edge (2 per element)
N_SNAP = 20             # structures (snapshots) to generate and homogenize
B = 20                  # designs per batched solve; N_SNAP is processed in chunks of B
SHARE = 4
RHO_VOID = 1e-6

NWAVE = 3.0             # pattern wavelengths per cell edge
REACTIVITY = 0.3        # r: near onset (0.3) the quadratic term gives spots/rods, larger r
                        # (0.7) a lamellar labyrinth
QUADRATIC = 1.0         # c2 of g(u) = c0 + c2 u^2 - u^3: 0 -> lamellae/labyrinth (u -> -u
                        # symmetric); |c2| ~ 0.5-1 with small r (0.3) -> spots/rods (BCC-like
                        # at onset). c2 > 0 rounds the solid phase, c2 < 0 rounds the voids.
FILL_BIAS = 0.0         # constant term c0 of g(u): shifts the mean of u (fill fraction of the
                        # phase u > 0). Labyrinth (r=0.7, c2=0): -0.2 -> 0.33, -0.1 -> 0.41,
                        # 0 -> 0.5, +0.2 -> 0.68. Rods (r=0.3, c2=1): keep 0 -- a negative
                        # c0 breaks the network into isolated spheres, thin it with U_THR.
ADVECTION = 0.3         # ABC flow amplitude (0 -> stationary pattern after warm-up)
DT = 0.2
T_WARM = 400.0          # pattern formation from noise, no advection (slower at small r)
T_GAP = 2.0             # time between snapshots under advection (~0.1 wavelength of drift)
U_THR = 0.4             # solid where u > U_THR. Rods (r=0.3, c2=1, u in [-1, 1.5]): 0 -> 0.51,
                        # 0.4 -> 0.34, 0.6 -> 0.26, 0.8 -> 0.19, network stays percolating
L_PDE = NWAVE * 2 * onp.pi   # PDE domain (k_c = 1 -> wavelength 2π)


# --- exponax: any stepper + advection by a steady velocity ------------------
class AdvectedNonlinearFun(BaseNonlinearFun):
    """inner(u_hat) - w·∇u, every channel advected by the velocity w (D, ...)."""
    inner: BaseNonlinearFun
    derivative_operator: jax.Array
    velocity: jax.Array

    def __init__(self, inner, derivative_operator, velocity):
        super().__init__(inner.num_spatial_dims, inner.num_points, dealiasing_fraction=None)
        self.dealiasing_mask = inner.dealiasing_mask
        self.inner = inner
        self.derivative_operator = derivative_operator
        self.velocity = velocity

    def __call__(self, u_hat):
        grad = jax.vmap(lambda c: self.ifft(self.derivative_operator * c))(u_hat)  # (C, D, ...)
        advection = jnp.einsum("d...,cd...->c...", self.velocity, grad)
        return self.inner(u_hat) - self.fft(advection)


def abc_flow(domain_extent, num_points, amplitude):
    k = 2 * jnp.pi / domain_extent
    x, y, z = (k * g for g in ex.make_grid(3, domain_extent, num_points))
    return amplitude * jnp.stack([jnp.sin(z) + jnp.cos(y),
                                  jnp.sin(x) + jnp.cos(z),
                                  jnp.sin(y) + jnp.cos(x)])


class SwiftHohenbergABC(SwiftHohenberg):
    amplitude: float

    def __init__(self, num_spatial_dims, domain_extent, num_points, dt, *, amplitude, **kw):
        self.amplitude = amplitude
        super().__init__(num_spatial_dims, domain_extent, num_points, dt, **kw)

    def _build_nonlinear_fun(self, derivative_operator):
        inner = super()._build_nonlinear_fun(derivative_operator)
        w = abc_flow(self.domain_extent, self.num_points, self.amplitude)
        return AdvectedNonlinearFun(inner, derivative_operator, w)


kw = dict(reactivity=REACTIVITY, critical_number=1.0,
          polynomial_coefficients=(FILL_BIAS, 0.0, QUADRATIC, -1.0), order=2)
grow = SwiftHohenbergABC(3, L_PDE, N_PDE, DT, amplitude=0.0, **kw)
stir = SwiftHohenbergABC(3, L_PDE, N_PDE, DT, amplitude=ADVECTION, **kw)

t0 = time.time()
# noise around the biased uniform state (mean ~ c0 / (k_c^2 - r) for small c0)
u0 = FILL_BIAS / (1.0 - REACTIVITY) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (1, N_PDE, N_PDE, N_PDE))
developed = ex.repeat(grow, int(T_WARM / DT))(u0)
snaps = jax.block_until_ready(ex.rollout(ex.repeat(stir, int(T_GAP / DT)), N_SNAP)(developed))[:, 0]
times = T_GAP * onp.arange(N_SNAP)
print(f"exponax {N_PDE}^3 Swift-Hohenberg: {T_WARM:.0f} warm-up + {N_SNAP} snapshots every {T_GAP:.0f} "
      f"under ABC flow ({int((T_WARM + N_SNAP * T_GAP) / DT)} ETDRK2 steps) in {time.time() - t0:.1f}s; "
      f"u in [{float(snaps.min()):.2f}, {float(snaps.max()):.2f}]")

# --- FE unit cell, nodal densities sampled from the PDE field --------------
class LinearElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2 * mu * eps
        return stress


class Cell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=L, mesh_size=mesh_size, element_type="HEX8")


cell = Cell(mesh_size=MESH)
mesh = cell.mesh
problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[])
P = flat.pbc.prolongation_matrix(flat.pbc.periodic_bc_3D(cell, vec=3, dim=3), mesh, vec=3)
bc = fe.DirichletBCConfig([]).create_bc(problem)
print(f"mesh: {len(mesh.points)} nodes, {len(mesh.cells)} HEX8 cells, reduced DOFs ~{P.shape[1]}; "
      f"cuDSS: {has_cudss()}")

# periodic trilinear sampling of u at the FE nodes (cell [0,L]^3 -> PDE grid;
# exact where the grids coincide)
coords = jnp.asarray(problem.fes[0].points).T * (N_PDE / L)
sample = jax.vmap(lambda f: jax.scipy.ndimage.map_coordinates(f, coords, order=1, mode="wrap"))
u_nodes = onp.asarray(sample(snaps))                                     # (N_SNAP, num_nodes)
rho = onp.where(u_nodes > U_THR, 1.0, RHO_VOID)

# time series of the structures on the FE mesh (ParaView: threshold density > 0.5)
with fe.XDMFWriter(mesh, "exponax_rda_lattice_homogenization.xdmf") as w:
    for b in range(N_SNAP):
        w.write_iteration(float(times[b]), point_infos=[("u", u_nodes[b]), ("density", rho[b])])
print("saved exponax_rda_lattice_homogenization.xdmf (+ .h5): u and density per snapshot")
E = jnp.asarray(E_BASE * rho)
nu = fe.TracedParams.create_cell_var(problem, NU)
in_axes = fe.TracedParams(volume_vars=(0, None))

# --- C_hom of all snapshots: fe.vmap of the single-cell solver, N_SNAP in
# chunks of B consecutive times (one compiled batched solver; the last chunk is
# padded by repeating the final snapshot) ---------------------------------
solve = flat.solver.create_homogenization_solver(problem, bc, P, mesh, dim=3)
pc = fe.shared_factor(scaling=True) if has_cudss() else fe.jacobi()
batched = fe.vmap(solve, in_axes=in_axes, preconditioner=pc, share=SHARE,
                  options=fe.KrylovSolverOptions(solver="cg", tol=1e-8, maxiter=2000))
chunks = [onp.minimum(onp.arange(c, c + B), N_SNAP - 1) for c in range(0, N_SNAP, B)]
C = onp.zeros((N_SNAP, 6, 6))
its = onp.zeros(N_SNAP, dtype=int)
t0 = time.time()
for idx in chunks:
    r = batched(fe.TracedParams(volume_vars=(E[idx], nu)))
    keep = onp.unique(idx)                                   # drop the padding of the last chunk
    C[keep] = onp.asarray(r.C_hom)[:len(keep)]
    its[keep] = onp.asarray(r.iterations).max(1)[:len(keep)]
print(f"homogenization of {N_SNAP} snapshots x 6 strain cases in {len(chunks)} batch(es) of {B} "
      f"in {time.time() - t0:.1f}s")


# --- anisotropy index ------------------------------------------------------
# Universal anisotropy index (Ranganathan & Ostoja-Starzewski 2008), valid for
# any symmetry: A_U = 5 G_V/G_R + K_V/K_R - 6, zero for an isotropic C_hom.
def universal_anisotropy_index(C):
    S = onp.linalg.inv(C)
    d, o = onp.einsum("...ii->...i", C[..., :3, :3]).sum(-1), C[..., 0, 1] + C[..., 0, 2] + C[..., 1, 2]
    sh = onp.einsum("...ii->...i", C[..., 3:, 3:]).sum(-1)
    K_V, G_V = (d + 2 * o) / 9, (d - o + 3 * sh) / 15
    sd, so = onp.einsum("...ii->...i", S[..., :3, :3]).sum(-1), S[..., 0, 1] + S[..., 0, 2] + S[..., 1, 2]
    ssh = onp.einsum("...ii->...i", S[..., 3:, 3:]).sum(-1)
    K_R, G_R = 1.0 / (sd + 2 * so), 15.0 / (4 * sd - 4 * so + 3 * ssh)
    return 5 * G_V / G_R + K_V / K_R - 6


A_U = universal_anisotropy_index(C)


def percolates(solid, axis):
    """True if the solid phase spans the periodic cell along ``axis``: a
    connected component of the box tiled twice along the axis touches both ends."""
    from scipy import ndimage
    lab, _ = ndimage.label(onp.concatenate([solid, solid], axis=axis))
    first, last = onp.take(lab, 0, axis=axis).ravel(), onp.take(lab, -1, axis=axis).ravel()
    return bool(set(first[first > 0]) & set(last[last > 0]))


solid = onp.asarray(snaps) > U_THR
perc = onp.array([[percolates(solid[b], a) for a in range(3)] for b in range(N_SNAP)])
rel_density = rho.mean(1)
print(f"\nuniversal anisotropy index A_U: min {A_U.min():.3f}, max {A_U.max():.3f}; "
      f"relative density {rel_density.min():.3f}..{rel_density.max():.3f}")

def bulk_modulus(C):
    return C[..., :3, :3].sum(axis=(-2, -1)) / 9.0


print(f"\n{N_SNAP} snapshots ({'shared factorization' if has_cudss() else 'Jacobi'}, share={SHARE} "
      f"consecutive times)")
print("  time   rel.density   C11/Es    C22/Es    C33/Es    C12/Es    C44/Es    K/Es     A_U    CG its (max)   percolates xyz")
for b in range(N_SNAP):
    print(f"  {times[b]:5.0f}   {rho[b].mean():.3f}      {C[b, 0, 0] / E_BASE:.4f}   {C[b, 1, 1] / E_BASE:.4f}"
          f"   {C[b, 2, 2] / E_BASE:.4f}   {C[b, 0, 1] / E_BASE:.4f}   {C[b, 3, 3] / E_BASE:.4f}"
          f"   {bulk_modulus(C[b]) / E_BASE:.4f}   {A_U[b]:.3f}   {int(its[b]):4d}           {''.join('xyz'[a] if perc[b, a] else '-' for a in range(3))}")

# stiffness spheres E(n) of every snapshot as a second XDMF time series: a unit
# sphere (TRI3 surface) with the directional Young's modulus E_dir = E(n)/Es and
# the vector warp = (E_dir - 1) n, so ParaView's "Warp By Vector" (scale 1)
# turns the sphere into the stiffness surface of radius E(n)/Es
N_THETA, N_PHI = 31, 61
th, ph = onp.meshgrid(onp.linspace(0, onp.pi, N_THETA), onp.linspace(0, 2 * onp.pi, N_PHI), indexing="ij")
dirs = onp.stack([onp.sin(th) * onp.cos(ph), onp.sin(th) * onp.sin(ph), onp.cos(th)], -1).reshape(-1, 3)
i, j = onp.meshgrid(onp.arange(N_THETA - 1), onp.arange(N_PHI - 1), indexing="ij")
q = onp.stack([i * N_PHI + j, (i + 1) * N_PHI + j, (i + 1) * N_PHI + j + 1, i * N_PHI + j + 1], -1).reshape(-1, 4)
tris = onp.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
sphere = fe.mesh.Mesh(dirs, tris, ele_type="TRI3")
E_dir = onp.asarray(jax.vmap(lambda Cb: jax.vmap(
    lambda n: flat.utils.compute_directional_youngs_modulus(Cb, n))(jnp.asarray(dirs)))(jnp.asarray(C))) / E_BASE
with fe.XDMFWriter(sphere, "exponax_rda_stiffness_sphere.xdmf") as w:
    for b in range(N_SNAP):
        w.write_iteration(float(times[b]), point_infos=[("E_dir", E_dir[b]),
                                                        ("warp", (E_dir[b, :, None] - 1.0) * dirs)])
print(f"saved exponax_rda_stiffness_sphere.xdmf (+ .h5): E(n)/Es per snapshot, "
      f"E_max/E_min = {E_dir.max(1) / E_dir.min(1)}")

# time order is the similarity order: consecutive snapshots are neighbours
if has_cudss():
    sig = onp.asarray(batched.operator_signature(fe.TracedParams(volume_vars=(E[chunks[0]], nu))))
    order = onp.asarray(gene.similarity_order(sig))
    print(f"\nfirst batch: similarity order {order.tolist()} (time order {list(range(B))}); mean group spread "
          f"time-order {gene.group_spread(sig, SHARE).mean():.3g}, "
          f"similarity-order {gene.group_spread(sig[order], SHARE).mean():.3g}")

# --- plot: relative density vs anisotropy, coloured by time ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from skimage.measure import marching_cubes

cmap, norm = plt.get_cmap("viridis"), Normalize(times[0], times[-1])
b_max, b_min = int(onp.argmax(A_U)), int(onp.argmin(A_U))
fig = plt.figure(figsize=(15, 5.2), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=(1.6, 1, 1))

ax = fig.add_subplot(gs[0])
sc = ax.scatter(rel_density, A_U, c=times, cmap=cmap, norm=norm, s=34, edgecolors="k", linewidths=0.4, zorder=2)
for b, lab in ((b_max, "max $A^U$"), (b_min, "min $A^U$")):
    ax.scatter(rel_density[b], A_U[b], s=160, facecolors="none", edgecolors="r", linewidths=1.5, zorder=3)
    ax.annotate(lab, (rel_density[b], A_U[b]), textcoords="offset points", xytext=(10, -4), fontsize=9, color="r")
ax.set_xlim(0.0, 0.5)
ax.set_ylim(bottom=0.0)
ax.set_xlabel("relative density")
ax.set_ylabel("universal anisotropy index $A^U$  (0 = isotropic)")
ax.set_title("density vs anisotropy")
ax.grid(alpha=0.3)
fig.colorbar(sc, ax=ax, label="time under ABC flow")

# the most anisotropic and the most isotropic lattice: isosurface u = U_THR of
# the periodic PDE field, mapped onto the unit cell [0, L]^3
u = onp.asarray(snaps)
for g, (b, lab) in zip((gs[1], gs[2]), ((b_max, "most anisotropic"), (b_min, "most isotropic"))):
    a = fig.add_subplot(g, projection="3d")
    field = onp.pad(u[b], ((0, 1),) * 3, mode="wrap")                 # close the periodic cell
    field = onp.pad(field, 1, constant_values=field.min() - 1.0)       # void outside -> capped cuts
    verts, faces, _, _ = marching_cubes(field, level=U_THR, spacing=(L / N_PDE,) * 3)
    verts = onp.clip(verts - L / N_PDE, 0.0, L)                        # caps flat on the cell faces
    a.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color="0.6", edgecolor="none",
                   shade=True, alpha=1.0)
    a.set_xlim(0, L); a.set_ylim(0, L); a.set_zlim(0, L)
    a.set_box_aspect((1, 1, 1))
    a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
    a.set_title(f"{lab}: t = {times[b]:.0f}, $A^U$ = {A_U[b]:.2f}, rel. density {rel_density[b]:.2f}", fontsize=9)
fig.savefig("exponax_rda_lattice_homogenization.png", dpi=150)
print("saved exponax_rda_lattice_homogenization.png")
