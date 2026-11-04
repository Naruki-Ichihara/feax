"""XFEM: a partially debonded interface between two materials (NLXP3D style).

Unit square plate, two isotropic materials stacked along y:

.. code-block:: text

        u_y = +d  (top edge, u_x = 0)
    +-------------------------+
    |      material 1 (top)   |  phi > 0
    |=============------------|  y = 0.5: interface on the left half only
    |      material 2 (bottom)|  phi < 0
    +-------------------------+
        u_y = -d  (bottom edge, u_x = 0)

The interface is described exactly as in NLXP3D by two nodal level sets:

* ``phi = y - 0.5``  normal level set (the interface is on ``phi = 0``),
* ``psi = x - 0.5``  tangential level set (the interface exists where ``psi < 0``).

The mesh does not conform to the interface (21 x 21 cells, so y = 0.5 runs
through cell centres).  The enrichment is NLXP3D's: the shifted sign function
``psi_j(x) = sign(phi(x)) - sign(phi_j)`` on the nodes of the interface cells,
i.e. a displacement jump.  The left half therefore opens like a crack between
the two materials under the tension, while the right half stays bonded.

What to look at in the output:

* interface cells / enriched nodes / DOFs: only the nodes of the cells the
  interface crosses carry the extra 2 DOFs (plus one phantom node).
* the opening ``u_y⁺ − u_y⁻`` along the interface: largest at the free edge
  ``x = 0`` and tapering to zero at the tip.  Without tip enrichment the
  jump closes linearly over the cell beyond the last interface cell, so
  the effective tip sits at the far edge of that cell (here x ≈ 0.57: the
  cell whose chord midpoint is exactly at ``psi = 0`` still counts as an
  interface cell, and the next one is the tip cell).
* ``xfem_partial_interface.png``: the deformed quadrature points, coloured by
  material, and the opening profile.

Run:  python examples/experimental/xfem_partial_interface.py
"""
import jax
import jax.numpy as np
import numpy as onp

import feax as fe
from feax.experimental.xfem import ShiftedHeaviside, XFEMSpace, xfem_problem

jax.config.update("jax_enable_x64", True)

N_CELLS = 21                 # odd: the interface line y = 0.5 crosses cell centres
ELE_TYPE = "QUAD4"
E_TOP, E_BOT, NU = 100.0, 50.0, 0.3
DELTA = 0.01                 # prescribed |u_y| on the top and bottom edges
TIP_X = 0.5                  # interface on 0 <= x < TIP_X


def material(u_grad, phi, x):
    """Two isotropic materials (plane strain), selected by the sign of phi."""
    E = np.where(phi > 0, E_TOP, E_BOT)
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    eps = 0.5 * (u_grad + u_grad.T)
    return lam * np.trace(eps) * np.eye(2) + 2 * mu * eps


def main():
    mesh = fe.mesh.rectangle_mesh(N_CELLS, N_CELLS, 1.0, 1.0, ele_type=ELE_TYPE)
    pts = onp.asarray(mesh.points)

    phi = pts[:, 1] - 0.5                     # normal level set
    psi = pts[:, 0] - TIP_X                   # tangential level set: interface where psi < 0
    space = XFEMSpace(mesh, phi, ShiftedHeaviside(), vec=2, ele_type=ELE_TYPE, psi=psi)
    problem = xfem_problem(space, material)

    print(f"mesh: {len(pts)} nodes, {space.num_cells} {ELE_TYPE} cells")
    print(f"interface cells: {space.interface_cells.sum()}  (cut for quadrature: {space.cut_cells.sum()}, "
          f"incl. the tip cell)")
    print(f"enriched nodes: {space.num_enriched}  ->  DOFs = 2*{len(pts)} + 2*({space.num_enriched}+1) "
          f"= {problem.num_total_dofs_all_vars}")

    tol = 1e-8
    bottom = lambda p: np.isclose(p[1], 0.0, tol)
    top = lambda p: np.isclose(p[1], 1.0, tol)
    bc = space.create_bc(problem, fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=bottom, component="all", value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=bottom, component=1, value=-DELTA, variable_index=0),
        fe.DirichletBCSpec(location=top, component=0, value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=top, component=1, value=DELTA, variable_index=0),
    ]))

    ts = fe.TracedStructure.from_problem(problem)
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    solver = fe.create_solver(problem, bc, solver_options=direct, linear=True, traced_structure=ts)
    sol = solver(space.traced_params(), fe.zero_like_initial_guess(problem, bc), traced_structure=ts)

    # --- opening along the interface ---------------------------------------
    x_mid, jump = (onp.asarray(a) for a in space.interface_jump(problem, sol))
    order = onp.argsort(x_mid[:, 0])
    print("\nopening u_y+ - u_y- at the chord midpoints of the cut cells:")
    print("     x       opening     sliding")
    for i in order:
        tag = "  <- tip cell (jump closes at its far edge)" if not space.interface_cells[
            onp.flatnonzero(space.cut_cells)[i]] else ""
        print(f"  {x_mid[i, 0]:.3f}   {jump[i, 1]: .3e}   {jump[i, 0]: .3e}{tag}")

    # --- figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x_q, u_q, phi_q = (onp.asarray(a) for a in space.quad_fields(problem, sol))
    real = space.qw > 0
    scale = 0.1 / DELTA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    xd = x_q[real] + scale * u_q[real]
    top_side = phi_q[real] > 0
    ax1.scatter(xd[top_side, 0], xd[top_side, 1], s=3, c="tab:blue", label=f"material 1 (E={E_TOP:g})")
    ax1.scatter(xd[~top_side, 0], xd[~top_side, 1], s=3, c="tab:orange", label=f"material 2 (E={E_BOT:g})")
    ax1.plot([0, TIP_X], [0.5, 0.5], "k--", lw=1, label="interface (undeformed)")
    ax1.set_aspect("equal"); ax1.set_title(f"deformed quadrature points (x{scale:g})")
    ax1.legend(fontsize=8, loc="upper right")
    ax2.plot(x_mid[order, 0], jump[order, 1], "o-", label="opening  u_y+ - u_y-")
    ax2.plot(x_mid[order, 0], jump[order, 0], "s-", label="sliding  u_x+ - u_x-")
    ax2.axvline(TIP_X, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("x along the interface"); ax2.set_ylabel("jump"); ax2.legend(fontsize=8)
    ax2.set_title("displacement jump across the interface")
    fig.tight_layout()
    fig.savefig("xfem_partial_interface.png", dpi=150)
    print("\nsaved xfem_partial_interface.png")

    # --- von Mises equivalent strain / stress at the quadrature points ----------
    x_q, u_q, phi_q, grad_u = (onp.asarray(a) for a in space.quad_fields(problem, sol, grad=True))
    g = grad_u[real]
    eps = 0.5 * (g + onp.swapaxes(g, 1, 2))                         # (n, 2, 2), plane strain
    sig = onp.asarray(jax.vmap(material)(np.asarray(g), np.asarray(phi_q[real]), np.asarray(x_q[real])))
    nu = NU
    s_zz = nu * (sig[:, 0, 0] + sig[:, 1, 1])                      # plane strain
    mises = onp.sqrt(0.5 * ((sig[:, 0, 0] - sig[:, 1, 1]) ** 2 + (sig[:, 1, 1] - s_zz) ** 2
                            + (s_zz - sig[:, 0, 0]) ** 2) + 3 * sig[:, 0, 1] ** 2)
    e_dev = eps - onp.trace(eps, axis1=1, axis2=2)[:, None, None] / 3 * onp.eye(2)
    e_zz = -onp.trace(eps, axis1=1, axis2=2) / 3                   # eps_zz = 0 -> deviatoric zz part
    eq_strain = onp.sqrt(2 / 3 * (onp.einsum("nij,nij->n", e_dev, e_dev) + e_zz ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, val, title in [(axes[0], eq_strain, "von Mises equivalent strain"),
                           (axes[1], mises, "von Mises stress")]:
        sc = ax.scatter(x_q[real][:, 0], x_q[real][:, 1], c=val, s=10, cmap="viridis")
        ax.plot([0, TIP_X], [0.5, 0.5], "w--", lw=1)
        ax.set_aspect("equal"); ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig("xfem_partial_interface_mises.png", dpi=150)
    print("saved xfem_partial_interface_mises.png")


if __name__ == "__main__":
    main()
