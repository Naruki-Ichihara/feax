"""DOF-only FE spaces (``with_jxw=False``).

A variable whose space is built with ``with_jxw=False`` contributes DOFs,
connectivity and sparsity but no geometry: its ``shape_grads`` / ``JxW`` are
zero arrays and ``mesh.points`` is never used. This is the building block for
XFEM-style enrichment spaces that live on a *subset* of the nodes (plus one
phantom node for the unenriched slots) without doubling the DOF count.

The functional test builds a 2-variable plane-strain problem where variable 1
is an "enrichment" living on a subset S of the nodes with a unit partition
(fi = 1 on S, 0 elsewhere), so the total displacement is ``u0 + m * a``.
Pinning ``a`` to a known constant on S and solving must reproduce the
single-variable reference solution through ``u0 + m * a``.
"""
import jax
import jax.numpy as np
import numpy as onp
import pytest

import feax as fe

jax.config.update("jax_enable_x64", True)

E, NU = 70e3, 0.3
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
U_RIGHT = 0.1
A_PIN = 0.03


def _direct():
    return fe.DirectSolverOptions(
        solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")


def _stress(eps):
    return LAM * np.trace(eps) * np.eye(2) + 2 * MU * eps


def _build_enrichment_mesh(mesh, subset_mask):
    """Second mesh: nodes of the subset + 1 phantom node (last index).

    Returns the mesh and ``local``: global node -> local node of the second
    mesh (the phantom index where the node is not in the subset).
    """
    subset = onp.flatnonzero(subset_mask)
    n_sub = len(subset)
    phantom = n_sub
    local = onp.full(len(mesh.points), phantom, dtype=onp.int32)
    local[subset] = onp.arange(n_sub, dtype=onp.int32)
    cells1 = local[onp.asarray(mesh.cells)]            # (num_cells, 4)
    points1 = onp.zeros((n_sub + 1, 2))                 # placeholders, never read
    points1[:n_sub] = onp.asarray(mesh.points)[subset]
    return fe.Mesh(points1, cells1, ele_type="QUAD4"), local


@pytest.fixture
def mesh():
    return fe.mesh.rectangle_mesh(8, 4, 2.0, 1.0, ele_type="QUAD4")


def test_dof_only_space_has_zero_geometry(mesh):
    space = fe.fe.FiniteElement(mesh=mesh, vec=2, dim=2, ele_type="QUAD4", with_jxw=False)
    ref = fe.fe.FiniteElement(mesh=mesh, vec=2, dim=2, ele_type="QUAD4")

    assert space.num_total_dofs == ref.num_total_dofs
    assert space.num_quads == ref.num_quads and space.num_nodes == ref.num_nodes
    assert space.shape_grads.shape == ref.shape_grads.shape
    assert space.JxW.shape == ref.JxW.shape
    assert space.v_grads_JxW.shape == ref.v_grads_JxW.shape
    assert not np.any(space.shape_grads) and not np.any(space.JxW)
    assert np.all(np.isfinite(space.JxW))
    # Reference shape functions are untouched (DOF interpolation still works)
    assert np.allclose(space.shape_vals, ref.shape_vals)

    faces = ref.get_boundary_conditions_inds([lambda p: np.isclose(p[0], 0.0)])[0]
    g, n = space.get_face_shape_grads(faces)
    assert g.shape == (len(faces), space.num_face_quads, space.num_nodes, 2)
    assert n.shape == (len(faces), space.num_face_quads)
    assert not np.any(g) and not np.any(n)

    with pytest.raises(ValueError, match="with_jxw=False"):
        space.get_physical_quad_points()
    with pytest.raises(ValueError, match="with_jxw=False"):
        space.get_boundary_conditions_inds([lambda p: np.isclose(p[0], 0.0)])


def test_degenerate_cells_do_not_produce_nan(mesh):
    """Phantom-node cells (all four nodes identical) are fine in a DOF-only space."""
    subset_mask = onp.asarray(mesh.points)[:, 0] > 0.75
    mesh1, _ = _build_enrichment_mesh(mesh, subset_mask)
    space = fe.fe.FiniteElement(mesh=mesh1, vec=2, dim=2, ele_type="QUAD4", with_jxw=False)
    assert np.all(np.isfinite(space.shape_grads)) and np.all(np.isfinite(space.JxW))
    # The same mesh WITH geometry would be degenerate (singular Jacobians).
    geo = fe.fe.FiniteElement(mesh=mesh1, vec=2, dim=2, ele_type="QUAD4")
    assert not np.all(np.isfinite(geo.shape_grads))


def test_problem_rejects_dof_only_variable_zero(mesh):
    with pytest.raises(ValueError, match="Variable 0"):
        fe.Problem(mesh=mesh, vec=2, dim=2, ele_type="QUAD4", with_jxw=False)
    with pytest.raises(ValueError, match="entries"):
        fe.Problem(mesh=[mesh, mesh], vec=[2, 2], dim=2,
                   ele_type=["QUAD4", "QUAD4"], with_jxw=[True])


def test_enrichment_space_on_node_subset(mesh):
    pts = onp.asarray(mesh.points)
    n_nodes = len(pts)
    tol = 1e-8

    # ---- reference: single-variable plane strain ------------------------
    class Reference(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad):
                eps = 0.5 * (u_grad + u_grad.T)
                return _stress(eps)
            return stress

    ref_problem = Reference(mesh=mesh, vec=2, dim=2, ele_type="QUAD4")
    left = lambda p: np.isclose(p[0], 0.0, tol)
    right = lambda p: np.isclose(p[0], 2.0, tol)
    ref_bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=left, component="all", value=0.0),
        fe.DirichletBCSpec(location=right, component=0, value=U_RIGHT),
    ]).create_bc(ref_problem)
    ref_ts = fe.TracedStructure.from_problem(ref_problem)
    ref_solver = fe.create_solver(ref_problem, ref_bc, solver_options=_direct(),
                                  linear=True, traced_structure=ref_ts)
    u_ref = ref_solver(fe.TracedParams(), fe.zero_like_initial_guess(ref_problem, ref_bc),
                       traced_structure=ref_ts)
    u_ref = onp.asarray(u_ref.dofs).reshape(n_nodes, 2)

    # ---- enriched: variable 1 lives on an interior node subset S ----------
    subset_mask = (pts[:, 0] > 0.5 + tol) & (pts[:, 0] < 1.5 - tol)
    mesh1, local = _build_enrichment_mesh(mesh, subset_mask)
    n_sub = int(subset_mask.sum())

    class Enriched(fe.Problem):
        def get_universal_kernel(self):
            unflatten = self.unflatten_fn_dof

            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                       cell_JxW, cell_v_grads_JxW, cell_mask):
                # cell_mask: (4,) gathered from the node-based mask of variable 0
                def energy(flat):
                    u0, a = unflatten(flat)                     # (4,2), (4,2)
                    u_tot = u0 + cell_mask[:, None] * a
                    grads = cell_shape_grads[:, :4, :]          # (nq, 4, 2)
                    u_grad = np.einsum("ai,qaj->qij", u_tot, grads)
                    eps = 0.5 * (u_grad + np.transpose(u_grad, (0, 2, 1)))
                    sig = jax.vmap(_stress)(eps)
                    return 0.5 * np.sum(np.einsum("qij,qij->q", sig, eps) * cell_JxW[0])
                return jax.grad(energy)(cell_sol_flat)
            return kernel

    problem = Enriched(mesh=[mesh, mesh1], vec=[2, 2], dim=2,
                       ele_type=["QUAD4", "QUAD4"], with_jxw=[True, False])

    assert problem.num_total_dofs_all_vars == 2 * n_nodes + 2 * (n_sub + 1)
    assert np.all(np.isfinite(problem.JxW)) and np.all(np.isfinite(problem.shape_grads))

    everywhere = lambda p: p[0] < np.inf
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=left, component="all", value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=right, component=0, value=U_RIGHT, variable_index=0),
        # pin the enrichment to a known constant (and the phantom slot to it too)
        fe.DirichletBCSpec(location=everywhere, component=0, value=A_PIN, variable_index=1),
        fe.DirichletBCSpec(location=everywhere, component=1, value=0.0, variable_index=1),
    ]).create_bc(problem)
    assert len(bc.bc_rows) >= 2 * (n_sub + 1)

    mask = np.asarray(subset_mask, dtype=np.float64)       # node-based on variable 0
    ts = fe.TracedStructure.from_problem(problem)
    assert np.all(np.isfinite(ts.JxW)) and np.all(np.isfinite(ts.shape_grads))
    solver = fe.create_solver(problem, bc, solver_options=_direct(), linear=True,
                              traced_structure=ts)
    sol = solver(fe.TracedParams(volume_vars=(mask,)),
                 fe.zero_like_initial_guess(problem, bc), traced_structure=ts)
    u0, a = problem.unflatten_fn_sol_list(sol)
    u0 = onp.asarray(u0)
    a = onp.asarray(a)

    assert onp.allclose(a[:n_sub, 0], A_PIN) and onp.allclose(a[:, 1], 0.0)
    u_tot = u0 + subset_mask[:, None] * a[local]           # m * a, phantom slots masked out
    # u0 + m*a must equal the reference field on every node
    assert onp.allclose(u_tot, u_ref, atol=1e-9, rtol=1e-9)
    # and the enrichment carries exactly A_PIN of the x-displacement on S
    assert onp.allclose(u0[subset_mask, 0], u_ref[subset_mask, 0] - A_PIN, atol=1e-9)
    assert onp.allclose(u0[~subset_mask], u_ref[~subset_mask], atol=1e-9)
