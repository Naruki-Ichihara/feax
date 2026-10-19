"""Shape optimization of a CAD feature on a fixed voxel grid — SDF + cmg + MMA.

An L-bracket with a gusset rib. The design is FIVE NUMBERS — where the rib
starts, where it ends, how thick it is — not a density field:

    theta = (ax, az, bx, bz, r)

Minimise compliance under a volume budget. A thin flange is bolted to a wall,
a second flange sticks out, and the gusset braces the corner; the tip load is
out of plane, which is the case a gusset exists for.

Why the grid never moves
------------------------
The geometry is implicit. A fixed rectangular voxel grid carries the whole
problem and the design enters only through the per-cell material weight

    w(x) = H_eps(-phi(x; theta)),    H_eps(s) = (1 + tanh(s/eps)) / 2

so ``theta -> phi -> w -> K(w) u = b -> J`` is one JAX graph and ``jax.grad``
is the discrete shape derivative at frozen structure. Nothing differentiates
through node positions: feax's StructuredGrid wants them fixed, and cmg's
O(band) matvec is index arithmetic on exactly that. Verified against central
differences to 8 significant digits on this problem class.

The smoothed Heaviside costs an O(eps) interface bias, the same diffuse-boundary
error a density-plus-projection topology optimization already carries. A sharp
cut-cell quadrature removes it but needs per-cell element matrices, which the
``KE * scalar`` form of cmg's ScalarMaterial cannot express.

The band is the full grid here: the rib sweeps most of the corner during
optimization, so no frozen subset stays valid. A band pays off when the design
moves little, and then by a factor set by the structure's fill fraction.

Requires cadjoint for the SDF primitives: pip install cadjoint
Outputs: bracket_rib.npy / bracket_rib.vtu
"""
import os

import jax
import jax.numpy as jnp
import numpy as onp

jax.config.update("jax_enable_x64", True)

import feax as fe
from feax.experimental.cmg import NarrowBandCMG, ScalarMaterial, simp
from feax.experimental.mma import MMA
from feax.utils import save_sol
from cadjoint.sdf import Box

H = 1.0                                   # voxel edge, mm
LX, LY, LZ = 50.0, 32.0, 50.0             # domain
T, W = 4.0, 2.0                           # flange thickness, gusset half-width
EPS, VOID = 1.5 * H, 1e-4                 # interface width, ersatz void weight
E0, NU = 1.0, 0.3
NX, NY, NZ = int(LX / H), int(LY / H), int(LZ / H)
NN1 = (NY + 1) * (NZ + 1)
OUT = os.path.dirname(os.path.abspath(__file__))

grid = fe.StructuredGrid((NX, NY, NZ), spacing=(H, H, H))
_i, _j, _k = onp.meshgrid(onp.arange(NX), onp.arange(NY), onp.arange(NZ),
                          indexing="ij")
P = jnp.asarray(onp.stack([(_i + 0.5) * H, (_j + 0.5) * H, (_k + 0.5) * H],
                          axis=-1).reshape(-1, 3))
WALL = jnp.array([T / 2, LY / 2, LZ / 2])
FLOOR = jnp.array([LX / 2, LY / 2, T / 2])


def weights(theta):
    """Per-cell material weight in [VOID, 1] for a design."""
    ax, az, bx, bz, r = theta
    a = jnp.stack([ax, az])
    ab = jnp.stack([bx, bz]) - a
    pa = jnp.stack([P[:, 0], P[:, 2]], axis=1) - a
    t = jnp.clip(jnp.dot(pa, ab) / jnp.maximum(jnp.dot(ab, ab), 1e-9), 0.0, 1.0)
    rib = jnp.maximum(jnp.linalg.norm(pa - t[:, None] * ab[None, :], axis=1) - r,
                      jnp.abs(P[:, 1] - LY / 2) - W)       # bar, clipped to mid-y
    phi = jnp.minimum(jnp.minimum(Box.sdf(P - WALL, WALL),
                                  Box.sdf(P - FLOOR, FLOOR)), rib)
    return VOID + (1.0 - VOID) * 0.5 * (1.0 - jnp.tanh(phi / EPS))


cmg = NarrowBandCMG(grid, lambda ni, nj, nk, *_: ni == 0,
                    material=ScalarMaterial.isotropic_elastic(
                        H, NU, simp(1e-9, E0, 1.0)),
                    cg_tol=1e-9, cg_maxit=800)
levels = cmg.build(onp.arange(NX * NY * NZ))
tip = onp.asarray([NX * NN1 + j * (NZ + 1) + k
                   for j in range(NY + 1) for k in range(int(T / H) + 1)])
b = cmg.load_vector(levels, tip, comp=2, value=-1.0)
solve = cmg.create_solver(levels, b)


def responses(theta):
    w = weights(theta)
    return jnp.dot(b, solve(w)), jnp.sum(w) * H ** 3


#                 ax      az      bx      bz     r
X0 = onp.array([T, 18.0, 18.0, T, 2.5])
LOW = onp.array([T, T + 2.0, T + 2.0, T, 1.0])
UPP = onp.array([T + 2.0, LZ - 6.0, LX - 6.0, T + 2.0, 6.0])

V_TARGET = float(responses(jnp.array([0., 0., 0., 0., 1e-6]))[1]) * 1.15
J0 = float(responses(jnp.asarray(X0))[0])
print(f"{NX}x{NY}x{NZ} = {NX*NY*NZ:,} cells   budget {V_TARGET:,.0f} mm^3")


def value_and_grad(theta):
    c, gc = jax.value_and_grad(lambda t: responses(t)[0])(theta)
    v, gv = jax.value_and_grad(lambda t: responses(t)[1])(theta)
    return (c / J0, gc / J0,
            jnp.array([v / V_TARGET - 1.0]), jnp.stack([gv / V_TARGET]))


opt = MMA(value_and_grad, LOW, UPP, n_constraints=1, c=1e4, move=0.2)
state = opt.init(jnp.asarray(X0))
for it in range(1, 26):
    state = opt.step(state)
    x = onp.asarray(state.x)
    c, v = responses(jnp.asarray(x))
    print(f"  it {it:3d}  A=({x[0]:5.2f},{x[1]:5.2f}) B=({x[2]:5.2f},{x[3]:5.2f}) "
          f"r={x[4]:4.2f}   C/C0={float(c)/J0:.4f}   V/Vmax={float(v)/V_TARGET:.3f}",
          flush=True)

theta = onp.asarray(state.x)
c, v = responses(jnp.asarray(theta))
print(f"\nA=({theta[0]:.2f},{theta[1]:.2f})  B=({theta[2]:.2f},{theta[3]:.2f})  "
      f"r={theta[4]:.2f}")
print(f"compliance {J0:.4e} -> {float(c):.4e}  ({(1 - float(c)/J0)*100:.1f}% stiffer)")
onp.save(os.path.join(OUT, "bracket_rib.npy"), theta)

# ── Results ────────────────────────────────────────────────────────────────
w = onp.asarray(weights(jnp.asarray(theta)))
u = onp.asarray(solve(jnp.asarray(w))).reshape(-1, 3)   # node n = i*NN1 + j*(NZ+1) + k

# VTK HEX8 corner order; the stress uses trilinear gradients at the element
# centre, so the only convention relied on is cmg's node numbering above.
VTK = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
ci, cj, ck = (a.ravel() for a in (_i, _j, _k))
cells = onp.stack([(ci + dx) * NN1 + (cj + dy) * (NZ + 1) + (ck + dz)
                   for dx, dy, dz in VTK], axis=1)
ni, nj, nk = onp.meshgrid(onp.arange(NX + 1), onp.arange(NY + 1),
                          onp.arange(NZ + 1), indexing="ij")
points = onp.stack([ni.ravel() * H, nj.ravel() * H, nk.ravel() * H], axis=1)

g = onp.array([[2 * dx - 1, 2 * dy - 1, 2 * dz - 1] for dx, dy, dz in VTK]) / (4 * H)
grad = onp.einsum("cna,ni->cai", u[cells], g)
eps = 0.5 * (grad + grad.transpose(0, 2, 1))
lam, mu = w * E0 * NU / ((1 + NU) * (1 - 2 * NU)), w * E0 / (2 * (1 + NU))
sig = (2 * mu[:, None, None] * eps
       + lam[:, None, None] * onp.trace(eps, axis1=1, axis2=2)[:, None, None] * onp.eye(3))
dev = sig - onp.eye(3) * (onp.trace(sig, axis1=1, axis2=2)[:, None, None] / 3)
vm = onp.sqrt(onp.maximum(1.5 * (dev * dev).sum(axis=(1, 2)), 0.0))

# Masked by density: the void is weak material, not absence, so its raw fields
# are real solutions and uninformative. Raw copies keep a _raw suffix.
mask = onp.clip(w[onp.clip(ni.ravel(), 0, NX - 1) * NY * NZ
                  + onp.clip(nj.ravel(), 0, NY - 1) * NZ
                  + onp.clip(nk.ravel(), 0, NZ - 1)], 0.0, 1.0)
save_sol(fe.Mesh(points=points, cells=cells, ele_type="HEX8"),
         os.path.join(OUT, "bracket_rib.vtu"),
         point_infos=[("u", u * mask[:, None]), ("u_raw", u),
                      ("u_mag", onp.linalg.norm(u, axis=1) * mask)],
         cell_infos=[("weight", w), ("von_mises", vm * (w > 0.5)),
                     ("von_mises_raw", vm)])
print(f"wrote bracket_rib.npy / bracket_rib.vtu to {OUT}")
