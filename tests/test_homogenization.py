"""Tests for feax.flat.solver.create_homogenization_solver.

The intended configuration is a fully periodic cell with an EMPTY Dirichlet
BC: the reduced system is singular (rigid translations) but consistent, and
each strain case must be solved independently (lax.map — a vmapped Krylov
while_loop would keep iterating converged cases until all six converge and
corrupt them on the singular operator). A homogeneous cell must recover the
analytic isotropic stiffness with zero fluctuation; a heterogeneous cell is
checked against a dense pseudo-inverse solve of the same reduced operator.
A Dirichlet pin on a periodically paired dof must be rejected at build time.
"""

import numpy as onp
import pytest

import jax.numpy as jnp

import feax as fe
import feax.flat as flat

E0, NU = 100.0, 0.3


class _Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2 * mu * eps
        return stress


class _BoxCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size=0.5):
        return fe.mesh.box_mesh((1.0, 1.0, 1.0), mesh_size=mesh_size,
                                element_type="HEX8")


def _setup(mesh_size=0.5):
    cell = _BoxCell(mesh_size=mesh_size)
    mesh = cell.mesh
    problem = _Elast(mesh, vec=3, dim=3, ele_type="HEX8")
    pairings = flat.pbc.periodic_bc_3D(cell, vec=3, dim=3)
    P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    return cell, mesh, problem, P, bc


def test_homogeneous_cell_recovers_isotropic_stiffness():
    cell, mesh, problem, P, bc = _setup()
    E_cells = fe.TracedParams.create_cell_var(problem, E0)
    nu_cells = fe.TracedParams.create_cell_var(problem, NU)
    tp = fe.TracedParams(volume_vars=(E_cells, nu_cells))

    solve = flat.solver.create_homogenization_solver(
        problem, bc, P, mesh,
        solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12), dim=3)

    assert solve.labels == ('eps11', 'eps22', 'eps33', 'gam23', 'gam13', 'gam12')
    assert solve.unit_strains.shape == (6, 3, 3)

    result = solve(tp)
    C = onp.asarray(result.C_hom)
    assert C.shape == (6, 6)

    mu = E0 / (2 * (1 + NU))
    lam = E0 * NU / ((1 + NU) * (1 - 2 * NU))
    C_ref = onp.zeros((6, 6))
    C_ref[:3, :3] = lam
    C_ref[onp.diag_indices(3)] = lam + 2 * mu
    C_ref[3:, 3:] = mu * onp.eye(3)
    # homogeneous cell: zero fluctuation, C_hom is analytic up to solver tol
    assert onp.abs(C - C_ref).max() < 1e-6 * onp.abs(C_ref).max()

    # zero fluctuation: the total displacement IS the affine field
    ndofs = problem.num_total_dofs_all_vars
    assert result.u_totals.shape == (6, ndofs)
    assert result.u_macros.shape == (6, ndofs)
    scale = float(onp.abs(onp.asarray(result.u_macros)).max())
    assert onp.abs(onp.asarray(result.u_totals - result.u_macros)).max() < 1e-8 * scale


def test_heterogeneous_cell_matches_dense_reference():
    from feax.assembler import create_matfree_res_J_parametric
    from feax.flat.solver import UNIT_STRAINS_3D, average_stress, macro_displacement

    cell, mesh, problem, P, bc = _setup()
    n_cells = onp.asarray(mesh.cells).shape[0]
    rng = onp.random.default_rng(0)
    E_cells = jnp.asarray(E0 * (1.0 + 0.5 * rng.random(n_cells)))
    nu_cells = fe.TracedParams.create_cell_var(problem, NU)
    tp = fe.TracedParams(volume_vars=(E_cells, nu_cells))

    solve = flat.solver.create_homogenization_solver(
        problem, bc, P, mesh,
        solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12), dim=3)
    C = onp.asarray(solve(tp).C_hom)
    assert onp.abs(C - C.T).max() < 1e-8 * onp.abs(C).max()

    # gold standard: materialize the reduced operator column by column and
    # solve each case with the minimum-norm pseudo-inverse (the singular
    # translations do not affect the averaged stress)
    mf = create_matfree_res_J_parametric(problem, symmetric=True)
    n_red = int(P.shape[1])
    eye = jnp.eye(n_red)
    C_ref = onp.zeros((6, 6))
    for k in range(6):
        um = macro_displacement(mesh, UNIT_STRAINS_3D[k])
        res, J_matvec = mf(um, tp, bc, None)
        b = -onp.asarray(P.T @ res)
        K = onp.stack([onp.asarray(P.T @ J_matvec(P @ eye[:, j]))
                       for j in range(n_red)], axis=1)
        x = onp.linalg.pinv(K, rcond=1e-8) @ b
        u_tot = onp.asarray(um) + onp.asarray(P @ jnp.asarray(x))
        C_ref[:, k] = onp.asarray(average_stress(problem, jnp.asarray(u_tot), tp, 3))

    assert onp.abs(C - C_ref).max() < 1e-8 * onp.abs(C_ref).max()


def test_pin_on_paired_dof_raises():
    # the origin corner is the master of the periodic corner class: a pin
    # there would be silently diluted by the PᵀJP fold, so the reduced path
    # must reject it at build time
    cell, mesh, problem, P, _ = _setup()
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0) & jnp.isclose(p[1], 0.0)
                           & jnp.isclose(p[2], 0.0), "all", 0.0),
    ]).create_bc(problem)
    with pytest.raises(ValueError, match="periodically paired"):
        flat.solver.create_homogenization_solver(
            problem, bc, P, mesh,
            solver_options=fe.KrylovSolverOptions(solver="cg"), dim=3)


def test_homogenization_rejects_bad_dim():
    cell = _BoxCell(mesh_size=1.0)
    problem = _Elast(cell.mesh, vec=3, dim=3, ele_type="HEX8")
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    with pytest.raises(ValueError):
        flat.solver.create_homogenization_solver(problem, bc, None, cell.mesh, dim=4)
