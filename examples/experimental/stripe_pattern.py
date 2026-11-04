"""Stripe patterns (print paths) aligned with a direction field on a 2D mesh.

``feax.experimental.stripe`` is a planar port of Knöppel, Crane, Pinkall &
Schröder, *Stripe Patterns on Surfaces* (2015): given a nodal line field and
a spacing, it computes a phase α whose iso-lines are evenly spaced stripes
running along the field.  The stripes are the filament paths of a 3D print;
``cos α`` is a phase-field-like scalar (1 on the path centre lines) and
``sin α`` (for an oriented field) is zero on the boundaries between
neighbouring filaments — the level set an XFEM interface model needs.

Three fields on a unit square, same spacing:

1. uniform field at 30 deg — exact parallel stripes;
2. vortex (tangential) field — circular stripes around the centre, with the
   field singularity in the middle;
3. a smoothly bending field ``angle = 0.9 x`` (fibres curving across the
   plate) — the spacing cannot be kept everywhere, so the pattern inserts
   branch points (``zero_index != 0``: a stripe starts or ends there).

A fourth panel shows the *differentiable* version: the direction and the
spacing of a uniform field are fitted by gradient descent (gradients through
the eigenproblem, :class:`DifferentiableStripes`) so that the pattern
matches a target of stripes at 60 deg — a toy version of print-path
optimisation.  (Optimising every nodal angle freely works the same way but
needs a smoothness regularisation of the field to stay well posed.)

Output: ``stripe_pattern.png`` (cos α with the direction field and the branch
points) and ``output/stripe_pattern/stripes_<case>.vtu`` for ParaView.

Run:  python examples/experimental/stripe_pattern.py
"""
import os

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe
from feax.experimental.stripe import DifferentiableStripes, StripePattern, smooth_line_field

jax.config.update("jax_enable_x64", True)

N = 60
SPACING = 0.08
OUT_DIR = os.path.join("output", "stripe_pattern")


def tri_mesh(n):
    quad = fe.mesh.rectangle_mesh(n, n, 1.0, 1.0, ele_type="QUAD4")
    c = onp.asarray(quad.cells)
    return fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]), ele_type="TRI3")


def main():
    mesh = tri_mesh(N)
    pts = onp.asarray(mesh.points)
    lam = 2 * onp.pi / SPACING
    cx, cy = 0.5 + 1 / (2 * N), 0.5 + 1 / (2 * N)
    cases = {
        "uniform_30deg": onp.full(len(pts), onp.deg2rad(30.0)),
        "vortex": (onp.arctan2(pts[:, 1] - cy, pts[:, 0] - cx) + onp.pi / 2) % onp.pi,
        "bending": smooth_line_field(mesh, 0.9 * pts[:, 0], t=0.002),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    print(f"mesh {len(pts)} nodes, {len(mesh.cells)} TRI3 cells, stripe spacing {SPACING}")
    print(" case            orientable  sheet-crossings  field-singular faces  stripe branch points")
    for name, ang in cases.items():
        spn = StripePattern.compute(mesh, ang, lam)
        results[name] = spn
        spn.save_vtu(os.path.join(OUT_DIR, f"stripes_{name}.vtu"))
        print(f" {name:14s}  {str(spn.orientable):9s}  {int(spn.crosses_sheets.sum()):15d}  "
              f"{int((spn.field_index != 0).sum()):20d}  {int((spn.zero_index != 0).sum()):20d}")
    print(f"VTU files in {OUT_DIR}/ (point data cos_alpha, alpha, direction; cell data zero_index, field_index)")

    # --- differentiable: fit a (direction, spacing) pair to a target stripe image ---
    ds = DifferentiableStripes(mesh)
    # a coarse target (about one stripe across the plate) keeps the basin of the
    # pattern-matching loss wide; fine patterns decorrelate after a few degrees
    theta_t, spacing_t = onp.deg2rad(60.0), 0.8
    lam_t = 2 * onp.pi / spacing_t
    z_target = onp.exp(1j * lam_t * (-onp.sin(theta_t) * pts[:, 0] + onp.cos(theta_t) * pts[:, 1]))
    area = onp.full(len(pts), 1.0 / len(pts))

    def loss(params):
        theta, log_lam = params
        z = ds.psi(jnp.full(len(pts), theta), jnp.exp(log_lam))
        # the global phase (and sign) of the pattern is free: use the modulus of the correlation
        return 1.0 - jnp.abs(jnp.sum(area * z * onp.conj(z_target)))

    grad = jax.grad(loss)
    params = jnp.array([onp.deg2rad(35.0), onp.log(2 * onp.pi / (0.75 * spacing_t))])
    print("\ngradient descent on (direction, spacing) through the eigenproblem; target 60 deg, spacing "
          f"{spacing_t}:")
    print("  iter   direction[deg]  spacing    loss")
    step = jnp.array([0.05, 0.05])
    for it in range(61):
        g = grad(params)
        if it % 10 == 0:
            print(f"  {it:4d}   {float(jnp.rad2deg(params[0])):8.2f}      {float(2 * jnp.pi / jnp.exp(params[1])):.4f}   "
                  f"{float(loss(params)):.4e}")
        params = params - step * g / (jnp.abs(g).max() + 1e-30)      # normalised gradient step
        step = step * 0.97
    results["optimised"] = StripePattern.compute(mesh, onp.full(len(pts), float(params[0])), float(jnp.exp(params[1])))
    results["optimised"].save_vtu(os.path.join(OUT_DIR, "stripes_optimised.vtu"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    panel_titles = {"optimised": "fitted direction & spacing (target 60 deg, 0.8)"}
    for ax, (name, spn) in zip(axes, results.items()):
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1], spn.mesh.cells)
        ax.tripcolor(tri, spn.scalar(), shading="gouraud", cmap="Greys_r", vmin=-1, vmax=1)
        step = max(1, len(pts) // 400)
        ang = spn.angle
        ax.quiver(pts[::step, 0], pts[::step, 1], onp.cos(ang[::step]), onp.sin(ang[::step]),
                  color="tab:red", scale=35, width=0.003, headwidth=1, headlength=0, pivot="middle")
        zi = spn.zero_index != 0
        if zi.any():
            cen = pts[spn.mesh.cells[zi]].mean(1)
            ax.plot(cen[:, 0], cen[:, 1], "o", mfc="none", mec="tab:blue", ms=7, mew=1.5, label="stripe branch point")
            ax.legend(loc="lower right", fontsize=8)
        fi = spn.field_index != 0
        if fi.any():
            cen = pts[spn.mesh.cells[fi]].mean(1)
            ax.plot(cen[:, 0], cen[:, 1], "x", color="tab:green", ms=9, mew=2)
        ax.set_aspect("equal"); ax.set_title(panel_titles.get(name, f"{name}: cos α, field (red)"))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("stripe_pattern.png", dpi=150)
    print("saved stripe_pattern.png")


if __name__ == "__main__":
    main()
