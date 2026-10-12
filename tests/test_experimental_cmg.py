"""Tests for feax.experimental.cmg: element matrices (make_KE_3d,
make_KE_conduction, KernelMaterial vs feax's assembled Jacobian), auto_levels,
band DOF addressing, the solver for vec=3 / vec=1 / anisotropic materials
against a dense reference solve of the feax-assembled operator, per-component
Dirichlet masks, shared band geometry, and the design gradient.
TracedParams acceptance and Solution wrapping are covered in
test_params_bridge.py and test_solution.py."""

import numpy as onp
import jax
import jax.numpy as jnp
import pytest

import feax as fe
from feax.assembler import get_jacobian
from feax.experimental import (KernelMaterial, NarrowBandCMG, ScalarMaterial,
                               auto_levels, build_geometry, make_KE_3d,
                               make_KE_conduction, simp)
from feax.experimental.cmg import (FEAX_TO_CMG_CORNER, _factorize,
                                   unit_cell_mesh)

_KRYLOV = fe.KrylovSolverOptions(solver="cg", tol=1e-12, maxiter=2000)


# ---------------------------------------------------------------------------
# Problems (feax kernels) used both as cmg materials and as the reference
# ---------------------------------------------------------------------------

class SimpElasticity(fe.Problem):
    def custom_init(self, nu):
        self.nu = nu

    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = 1e-9 + rho ** 3 * (1 - 1e-9)
            nu = self.nu
            lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress


class OrthotropicElasticity(fe.Problem):
    """Per-cell density AND per-cell axis-aligned stiffness ratios: the
    element matrix is no longer a scalar multiple of one KE."""

    def get_tensor_map(self):
        def stress(u_grad, rho, kx, ky, kz):
            E = 1e-6 + rho ** 3
            eps = 0.5 * (u_grad + u_grad.T)
            k = jnp.array([kx, ky, kz])
            # anisotropic "Lamé-like" law: sigma = 2 mu_i eps_ij mu_j^0 ... keep it
            # simple but genuinely anisotropic and symmetric
            A = jnp.diag(k)
            return E * (A @ eps @ A + 0.3 * jnp.trace(eps) * jnp.eye(3))
        return stress


class SimpConduction(fe.Problem):
    def get_tensor_map(self):
        def flux(T_grad, rho):
            return (1e-3 + rho ** 3 * (1 - 1e-3)) * T_grad
        return flux


def _permutation(vec):
    """DOF permutation P with K_feax[P][:, P] == K_cmg (corner order)."""
    P = onp.zeros(8 * vec, int)
    for a in range(8):
        for c in range(vec):
            P[vec * a + c] = vec * FEAX_TO_CMG_CORNER[a] + c
    return P


def _reference_solve(problem_cls, grid, active, vec, params, free, b, additional_info=()):
    """Dense solve of the feax-assembled operator on the band mesh with the
    fixed DOFs eliminated (independent of cmg's kernels)."""
    band = fe.NarrowBand(grid, active)
    problem = problem_cls(mesh=band.mesh, vec=vec, dim=3, ele_type='HEX8',
                          additional_info=tuple(additional_info))
    tp = fe.TracedParams(volume_vars=tuple(jnp.asarray(p) for p in params))
    K = onp.asarray(get_jacobian(problem, [jnp.zeros((band.num_band_nodes, vec))], tp).todense())
    u = onp.zeros(b.size)
    u[free] = onp.linalg.solve(K[onp.ix_(free, free)], b[free])
    return u, band


# ---------------------------------------------------------------------------
# element matrices
# ---------------------------------------------------------------------------

def test_make_ke_3d_symmetric_psd_with_rigid_modes():
    KE = make_KE_3d(nu=0.3, E=1.0)
    assert KE.shape == (24, 24)
    assert onp.allclose(KE, KE.T, atol=1e-12)
    w = onp.linalg.eigvalsh(KE)
    assert w.min() > -1e-10                       # positive semi-definite
    assert (onp.abs(w) < 1e-10).sum() == 6        # exactly the 6 rigid-body modes
    for comp in range(3):
        u = onp.zeros(24)
        u[comp::3] = 1.0
        assert onp.abs(KE @ u).max() < 1e-12


def test_make_ke_3d_scales_with_e_and_spacing():
    K1 = make_KE_3d(nu=0.3, E=1.0)
    assert onp.allclose(make_KE_3d(nu=0.3, E=7.5), 7.5 * K1)
    # 3D elasticity stiffness scales linearly with a uniform cell size
    assert onp.allclose(make_KE_3d(nu=0.3, E=1.0, spacing=2.0), 2.0 * K1)


def test_make_ke_conduction():
    KE = make_KE_conduction(spacing=1.0, k=2.0)
    assert KE.shape == (8, 8)
    assert onp.allclose(KE, KE.T)
    assert onp.abs(KE @ onp.ones(8)).max() < 1e-12       # constant field: zero flux
    w = onp.linalg.eigvalsh(KE)
    assert (onp.abs(w) < 1e-10).sum() == 1
    assert onp.allclose(KE, 2.0 * make_KE_conduction(1.0, 1.0))


@pytest.mark.parametrize("problem_cls,vec,params,info", [
    (SimpElasticity, 3, (0.7,), (0.3,)),
    (SimpConduction, 1, (0.4,), ()),
    (OrthotropicElasticity, 3, (0.8, 1.0, 2.0, 0.5), ()),
])
def test_kernel_material_matches_feax_element_jacobian(problem_cls, vec, params, info):
    grid = fe.StructuredGrid((3, 2, 2), spacing=(0.5, 1.0, 2.0))     # anisotropic spacing
    mat = KernelMaterial.from_problem(problem_cls, grid, vec, additional_info=info,
                                      sample_params=params)
    assert mat.vec == vec
    Kc = onp.asarray(mat.element_matrices(mat.state0(tuple(jnp.array([p]) for p in params))))[0]

    problem = problem_cls(mesh=unit_cell_mesh(grid), vec=vec, dim=3, ele_type='HEX8',
                          additional_info=info)
    tp = fe.TracedParams(volume_vars=tuple(jnp.array([p]) for p in params))
    Kf = onp.asarray(get_jacobian(problem, [jnp.zeros((8, vec))], tp).todense())
    P = _permutation(vec)
    assert onp.allclose(Kf[onp.ix_(P, P)], Kc, rtol=1e-10, atol=1e-12 * onp.abs(Kc).max())


def test_kernel_material_rejects_nonlinear_and_nonsymmetric():
    with pytest.raises(ValueError, match="not linear|homogeneous|affine"):
        KernelMaterial(lambda g, r: r * g + 0.1 * g ** 2, 1, 1.0, sample_params=(0.5,))
    with pytest.raises(ValueError, match="affine"):
        KernelMaterial(lambda g, r: r * g + 1.0, 1, 1.0, sample_params=(0.5,))
    with pytest.raises(ValueError, match="symmetric"):
        A = jnp.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        KernelMaterial(lambda g, r: r * (g @ A), 1, 1.0, sample_params=(0.5,))


def test_scalar_material_shape_check():
    with pytest.raises(ValueError):
        ScalarMaterial(onp.zeros((10, 10)))


# ---------------------------------------------------------------------------
# auto_levels / band DOF addressing
# ---------------------------------------------------------------------------

def test_auto_levels_halves_until_floor():
    assert auto_levels((32, 32, 32), floor=4) == 4     # 32 -> 16 -> 8 -> 4
    assert auto_levels((8, 4, 4), floor=4) == 1        # 4 -> 2 would drop below floor
    assert auto_levels((9, 8, 8), floor=4) == 2        # ceil-halving: 9 -> 5, 8 -> 4
    assert auto_levels((9, 5, 5), floor=4) == 1        # 5 -> 3 would drop below floor
    assert auto_levels((64, 32, 16), floor=8) == 2
    # default floor 16: shallow hierarchies, the coarse solver does the rest
    assert auto_levels((33, 33, 33)) == 2              # 33 -> 17 (>= 16) -> 9 no
    assert auto_levels((1200, 300, 300)) == 5          # 300 -> 150 -> 75 -> 38 -> 19
    assert auto_levels((8, 4, 4)) == 1


def test_compact_nodes_and_load_vector():
    grid = fe.StructuredGrid((8, 4, 4))
    cmg = NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: ni == 0,
                        nu=0.3, penal=3.0, bucket=64)
    assert cmg.vec == 3
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    levels = cmg.build(active)

    tip = grid.nodes_where(lambda I, J, K: (I == 8) & (K == 0))
    cn = cmg.compact_nodes(levels, tip)
    assert onp.array_equal(onp.asarray(levels[0]["nodes"])[cn], onp.asarray(tip))

    b = cmg.load_vector(levels, tip, comp=2, value=-1.0)
    assert b.shape == (levels[0]["ndof"],)
    assert onp.allclose(b[3 * cn + 2], -1.0)
    assert onp.count_nonzero(b) == tip.size
    bx = cmg.load_vector(levels, tip, comp=0, value=2.0)
    assert onp.allclose(bx[3 * cn], 2.0)
    assert onp.dot(b, bx) == 0.0
    # default component is the last one
    assert onp.array_equal(cmg.load_vector(levels, tip, value=-1.0), b)
    with pytest.raises(ValueError):
        cmg.load_vector(levels, tip, comp=3)


def test_load_vector_scalar_field():
    grid = fe.StructuredGrid((4, 4, 4))
    cmg = NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: ni == 0,
                        material=ScalarMaterial.isotropic_conduction(), bucket=64)
    assert cmg.vec == 1
    levels = cmg.build(onp.arange(grid.num_cells))
    assert levels[0]["ndof"] == levels[0]["nnode"]
    nodes = grid.nodes_where(lambda I, J, K: I == 4)
    b = cmg.load_vector(levels, nodes, value=1.0)
    assert onp.count_nonzero(b) == nodes.size
    with pytest.raises(ValueError):
        cmg.load_vector(levels, nodes, comp=2)


# ---------------------------------------------------------------------------
# solves against the feax-assembled reference
# ---------------------------------------------------------------------------

def _grid_and_band():
    grid = fe.StructuredGrid((6, 4, 3), spacing=(0.5, 1.0, 2.0))
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    return grid, active


def test_scalar_and_kernel_elasticity_match_reference():
    grid, active = _grid_and_band()
    fixed = lambda ni, nj, nk, nx, ny, nz: ni == 0
    rng = onp.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.3, 1.0, active.size))

    kernel = NarrowBandCMG(grid, fixed, bucket=64, material=KernelMaterial.from_problem(
        SimpElasticity, grid, 3, additional_info=(0.3,), sample_params=(0.5,)))
    scalar = NarrowBandCMG(grid, fixed, bucket=64, material=ScalarMaterial.isotropic_elastic(
        grid.spacing, nu=0.3, scale_fn=simp(1e-9, 1.0, 3.0)))
    lv = kernel.build(active)
    tip = grid.nodes_where(lambda I, J, K: (I == 6) & (K == 0))
    b = kernel.load_vector(lv, tip, comp=2, value=-1.0)
    u_ref, band = _reference_solve(SimpElasticity, grid, active, 3, (rho,),
                                   lv[0]["free"] > 0, b, additional_info=(0.3,))
    assert onp.array_equal(band.node_map, lv[0]["nodes"])       # same node layout

    for cmg in (kernel, scalar):
        levels = cmg.build(active)
        u = onp.asarray(cmg.create_solver(levels, b, solver_options=_KRYLOV,
                                          return_solution=False)(rho))
        assert onp.allclose(u, u_ref, rtol=1e-8, atol=1e-10 * onp.abs(u_ref).max())


def test_scalar_conduction_matches_reference():
    grid, active = _grid_and_band()
    fixed = lambda ni, nj, nk, nx, ny, nz: ni == 0                  # T = 0 at x = 0
    rng = onp.random.default_rng(1)
    rho = jnp.asarray(rng.uniform(0.2, 1.0, active.size))
    cmg = NarrowBandCMG(grid, fixed, bucket=64, material=ScalarMaterial.isotropic_conduction(
        grid.spacing, scale_fn=simp(1e-3, 1.0, 3.0)))
    lv = cmg.build(active)
    hot = grid.nodes_where(lambda I, J, K: I == 6)
    b = cmg.load_vector(lv, hot, value=1.0)
    u_ref, _ = _reference_solve(SimpConduction, grid, active, 1, (rho,), lv[0]["free"] > 0, b)
    u = onp.asarray(cmg.create_solver(lv, b, solver_options=_KRYLOV, return_solution=False)(rho))
    assert onp.allclose(u, u_ref, rtol=1e-8, atol=1e-10 * onp.abs(u_ref).max())
    assert u.max() > 0 and onp.allclose(u[lv[0]["free"] == 0], 0.0)


def test_kernel_orthotropic_matches_reference_and_grad():
    grid, active = _grid_and_band()
    fixed = lambda ni, nj, nk, nx, ny, nz: ni == 0
    rng = onp.random.default_rng(2)
    n = active.size
    params = (jnp.asarray(rng.uniform(0.3, 1.0, n)), jnp.asarray(rng.uniform(0.5, 2.0, n)),
              jnp.asarray(rng.uniform(0.5, 2.0, n)), jnp.asarray(rng.uniform(0.5, 2.0, n)))
    mat = KernelMaterial.from_problem(OrthotropicElasticity, grid, 3,
                                      sample_params=(0.5, 1.0, 1.5, 0.7))
    cmg = NarrowBandCMG(grid, fixed, material=mat, bucket=64)
    lv = cmg.build(active)
    tip = grid.nodes_where(lambda I, J, K: (I == 6) & (K == 3))
    b = cmg.load_vector(lv, tip, comp=1, value=-1.0)
    u_ref, _ = _reference_solve(OrthotropicElasticity, grid, active, 3, params,
                                lv[0]["free"] > 0, b)
    solver = cmg.create_solver(lv, b, solver_options=_KRYLOV, return_solution=False)
    u = onp.asarray(solver(params))
    assert onp.allclose(u, u_ref, rtol=1e-8, atol=1e-10 * onp.abs(u_ref).max())

    # design gradient (tuple params): implicit-diff vjp == analytic self-adjoint
    c, dc = cmg.compliance_and_dc(lv, params, u)
    assert c == pytest.approx(float(onp.dot(b, u)))
    g = jax.grad(lambda p: jnp.dot(jnp.asarray(b), solver(p)))(params)
    assert len(dc) == 4 and len(g) == 4
    for gi, di in zip(g, dc):
        assert onp.allclose(onp.asarray(gi), di, rtol=1e-6, atol=1e-8 * onp.abs(di).max())
    # TracedParams with several volume_vars
    u_tp = solver(fe.TracedParams(volume_vars=params))
    assert onp.allclose(onp.asarray(u_tp), u)


def test_per_component_dirichlet():
    grid, active = _grid_and_band()

    def fixed(ni, nj, nk, nx, ny, nz):
        m = onp.zeros((ni.size, 3), bool)
        m[ni == 0] = True                    # clamp x = 0
        m[nk == 0, 2] = True                 # roller: only u_z fixed on z = 0
        return m

    cmg = NarrowBandCMG(grid, fixed, bucket=64, material=ScalarMaterial.isotropic_elastic(
        grid.spacing, nu=0.3, scale_fn=simp(1e-9, 1.0, 3.0)))
    lv = cmg.build(active)
    free = lv[0]["free"] > 0
    nz0 = (lv[0]["nk"] == 0) & (lv[0]["ni"] > 0)
    assert free.reshape(-1, 3)[nz0, 0].all() and not free.reshape(-1, 3)[nz0, 2].any()

    rho = jnp.full(active.size, 0.6)
    tip = grid.nodes_where(lambda I, J, K: (I == 6) & (K == 3))
    b = cmg.load_vector(lv, tip, comp=0, value=1.0)
    u_ref, _ = _reference_solve(SimpElasticity, grid, active, 3, (rho,), free, b,
                                additional_info=(0.3,))
    u = onp.asarray(cmg.create_solver(lv, b, solver_options=_KRYLOV, return_solution=False)(rho))
    assert onp.allclose(u, u_ref, rtol=1e-8, atol=1e-10 * onp.abs(u_ref).max())
    assert onp.allclose(u.reshape(-1, 3)[nz0, 2], 0.0)
    assert onp.abs(u.reshape(-1, 3)[nz0, 0]).max() > 0


def test_shared_geometry_between_physics():
    grid, active = _grid_and_band()
    fixed = lambda ni, nj, nk, nx, ny, nz: ni == 0
    geo = build_geometry(grid, active)
    thermal = NarrowBandCMG(grid, fixed, bucket=64,
                            material=ScalarMaterial.isotropic_conduction(grid.spacing))
    elastic = NarrowBandCMG(grid, fixed, bucket=64,
                            material=ScalarMaterial.isotropic_elastic(grid.spacing))
    lt, le = thermal.build(geo), elastic.build(geo)
    assert lt[0]["nodes"] is geo[0]["nodes"] and le[0]["nodes"] is geo[0]["nodes"]
    assert lt[0]["ndof"] == geo[0]["nnode"] and le[0]["ndof"] == 3 * geo[0]["nnode"]
    # identical to the un-shared build
    le2 = elastic.build(active)
    for a, c in zip(le, le2):
        assert onp.array_equal(a["edof_c"], c["edof_c"]) and onp.array_equal(a["free"], c["free"])
    with pytest.raises(ValueError):
        NarrowBandCMG(grid, fixed, n_levels=2, bucket=64).build(geo)


def test_grad_matches_analytic_scalar():
    grid = fe.StructuredGrid((8, 4, 4))
    cmg = NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: ni == 0, bucket=64)
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    lv = cmg.build(active)
    tip = grid.nodes_where(lambda I, J, K: (I == 8) & (K == 0))
    b = cmg.load_vector(lv, tip, comp=2, value=-1.0)
    solver = cmg.create_solver(lv, b, solver_options=_KRYLOV, return_solution=False)
    rho = jnp.asarray(onp.random.default_rng(3).uniform(0.2, 1.0, active.size))
    u = solver(rho)
    c, dc = cmg.compliance_and_dc(lv, rho, u)
    g = jax.grad(lambda r: jnp.dot(jnp.asarray(b), solver(r)))(rho)
    assert c == pytest.approx(float(jnp.dot(jnp.asarray(b), u)))
    assert onp.allclose(onp.asarray(g), dc, rtol=1e-6, atol=1e-8 * onp.abs(dc).max())


@pytest.mark.skipif(_factorize is None, reason="cuDSS (spineax) not available")
def test_cudss_coarse_level_with_padding():
    # regression: the padded coarsest RHS must match the factorized matrix size
    grid = fe.StructuredGrid((8, 4, 4))
    cmg = NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: ni == 0, bucket=64)
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    lv = cmg.build(active)
    tip = grid.nodes_where(lambda I, J, K: (I == 8) & (K == 0))
    b = cmg.load_vector(lv, tip, comp=2, value=-1.0)
    rho = jnp.full(active.size, 0.5)
    u_d = onp.asarray(cmg.create_solver(lv, b, return_solution=False)(rho))
    u_k = onp.asarray(cmg.create_solver(lv, b, solver_options=_KRYLOV,
                                        return_solution=False)(rho))
    assert onp.allclose(u_d, u_k, rtol=1e-7, atol=1e-9 * onp.abs(u_k).max())


# ---------------------------------------------------------------------------
# box_mesh layout: energy-density Problems, face tractions, Dirichlet lift, rhs grad
# ---------------------------------------------------------------------------

class _DiffEnergy(fe.Problem):
    """Scalar diffusion defined by an energy density (no tensor_map)."""

    def get_energy_density(self):
        def psi(g, rho, kx):
            K = jnp.diag(jnp.array([kx, 1.0, 0.5])) * (1e-3 + rho ** 3)
            return 0.5 * jnp.dot(g.ravel(), K @ g.ravel())
        return psi


def _box_setup():
    mesh = fe.mesh.box_mesh(size=(3.0, 2.0, 1.5), mesh_size=0.5, element_type="HEX8")
    grid = fe.StructuredGrid.from_mesh(mesh)
    rng = onp.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.2, 1.0, grid.num_cells))
    kx = jnp.asarray(rng.uniform(0.5, 2.0, grid.num_cells))
    tol = 1e-9
    faces = dict(bottom=lambda p: jnp.isclose(p[1], 0.0, atol=tol),
                 top=lambda p: jnp.isclose(p[1], 2.0, atol=tol))
    return mesh, grid, rho, kx, faces


def test_from_problem_instance_energy_density_and_dirichlet_lift():
    mesh, grid, rho, kx, faces = _box_setup()
    th = _DiffEnergy(mesh, vec=1, dim=3, ele_type="HEX8")
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(faces["top"], "all", 1.0),
                               fe.DirichletBCSpec(faces["bottom"], "all", 0.0)]).create_bc(th)
    ts = fe.TracedStructure.from_problem(th)
    tp = fe.TracedParams(volume_vars=(rho, kx))
    direct = fe.DirectSolverOptions()
    solve_T = fe.create_solver(th, bc, solver_options=direct, adjoint_solver_options=direct,
                               linear=True, traced_structure=ts, traced_params=tp)
    guess = fe.zero_like_initial_guess(th, bc)
    T_ref = onp.asarray(solve_T(tp, guess, traced_structure=ts).dofs).reshape(-1)

    mat = KernelMaterial.from_problem(th, grid, sample_params=(0.5, 1.2))   # instance, psi
    assert mat.vec == 1
    with pytest.raises(ValueError):
        KernelMaterial.from_problem(th, grid, vec=3)
    cmg = NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: (nj == 0) | (nj == ny),
                        material=mat, bucket=64)
    lv = cmg.build(onp.arange(grid.num_cells))
    assert onp.array_equal(lv[0]["nodes"], onp.arange(grid.num_nodes))    # same layout as box_mesh
    u_p = jnp.asarray((lv[0]["nj"] == grid.ny).astype(float))
    solver = cmg.create_solver(lv, solver_options=_KRYLOV, return_solution=False)
    with pytest.raises(ValueError):
        solver((rho, kx))                                                    # no rhs anywhere

    def T_of(r):
        p = (r, kx)
        return u_p + solver(p, b=cmg.dirichlet_rhs(lv, p, u_p))

    T = onp.asarray(T_of(rho))
    assert onp.allclose(T, T_ref, rtol=1e-8, atol=1e-10)
    # design gradient through the design-dependent rhs matches feax's adjoint
    energy = fe.create_energy_fn(th)
    g_c = jax.grad(lambda r: energy(T_of(r), fe.TracedParams(volume_vars=(r, kx))))(rho)
    g_r = jax.grad(lambda r: energy(
        solve_T(fe.TracedParams(volume_vars=(r, kx)), guess, traced_structure=ts).dofs,
        fe.TracedParams(volume_vars=(r, kx))))(rho)
    assert onp.allclose(onp.asarray(g_c), onp.asarray(g_r), rtol=1e-8, atol=1e-12)


def test_face_load_matches_feax_surface_traction_and_rhs_grad():
    mesh, grid, rho, kx, faces = _box_setup()
    tvec = jnp.array([0.3, 1.0, -0.2])

    class Elast(fe.Problem):
        def get_energy_density(self):
            def psi(ug, rho, kx):
                E = 1e-6 + rho ** 3; nu = 0.3
                lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
                A = jnp.diag(jnp.array([kx, 1.0, 1.0]))
                eps = A @ (0.5 * (ug + ug.T)) @ A
                return 0.5 * (lam * jnp.trace(eps) ** 2 + 2 * mu * jnp.sum(eps * eps))
            return psi

        def get_surface_maps(self):
            return [lambda u, x, *a: tvec]

    me = Elast(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=(faces["bottom"],))
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(faces["top"], "all", 0.0)]).create_bc(me)
    ts = fe.TracedStructure.from_problem(me)
    tp = fe.TracedParams(volume_vars=(rho, kx))
    direct = fe.DirectSolverOptions()
    solve_u = fe.create_solver(me, bc, solver_options=direct, adjoint_solver_options=direct,
                               linear=True, traced_structure=ts, traced_params=tp)
    guess = fe.zero_like_initial_guess(me, bc)
    u_ref = onp.asarray(solve_u(tp, guess, traced_structure=ts).dofs).reshape(-1)

    cmg = NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: nj == ny, bucket=64,
                        material=KernelMaterial.from_problem(me, grid, sample_params=(0.5, 1.2)))
    lv = cmg.build(onp.arange(grid.num_cells))
    b = cmg.face_load_vector(lv, axis=1, side=0, traction=-onp.asarray(tvec))   # feax sign
    # total force = traction * face area
    assert onp.allclose(b.reshape(-1, 3).sum(0), -onp.asarray(tvec) * 3.0 * 1.5)
    solver = cmg.create_solver(lv, b, solver_options=_KRYLOV, return_solution=False)
    u = onp.asarray(solver((rho, kx)))
    assert onp.allclose(u, u_ref, rtol=1e-8, atol=1e-10 * onp.abs(u_ref).max())

    # gradient w.r.t. the right-hand side is the adjoint solve
    b2 = jnp.asarray(onp.random.default_rng(1).standard_normal(b.size) * onp.asarray(lv[0]["free"]))
    gb = jax.grad(lambda bb: jnp.dot(b2, solver((rho, kx), b=bb)))(jnp.asarray(b))
    lam = onp.asarray(solver((rho, kx), b=b2))
    assert onp.allclose(onp.asarray(gb), lam, rtol=1e-8, atol=1e-12)


@pytest.mark.skipif(_factorize is None, reason="cuDSS (spineax) not available")
def test_solver_under_outer_jit_with_cudss():
    mesh, grid, rho, kx, faces = _box_setup()
    cmg = NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: nj == ny, bucket=64)
    lv = cmg.build(onp.arange(grid.num_cells))
    b = cmg.face_load_vector(lv, axis=1, side=0, traction=(0.0, -1.0, 0.0))
    solver = cmg.create_solver(lv, b, return_solution=False)
    f = lambda r: jnp.dot(jnp.asarray(b), solver(r))
    c_j, g_j = jax.jit(jax.value_and_grad(f))(rho)
    c_e, g_e = jax.value_and_grad(f)(rho)
    assert float(c_j) == pytest.approx(float(c_e), rel=1e-10)
    assert onp.allclose(onp.asarray(g_j), onp.asarray(g_e), rtol=1e-8)


@pytest.mark.skipif(_factorize is None, reason="cuDSS (spineax) not available")
def test_mixed_precision_matches_double():
    mesh, grid, rho, kx, faces = _box_setup()
    mat = KernelMaterial.from_problem(SimpElasticity, grid, 3, additional_info=(0.3,))
    fixed = lambda ni, nj, nk, nx, ny, nz: nj == ny
    out = {}
    for prec in ("double", "mixed"):
        cmg = NarrowBandCMG(grid, fixed, material=mat, cg_tol=1e-11, precision=prec, bucket=64)
        lv = cmg.build(onp.arange(grid.num_cells))
        b = cmg.face_load_vector(lv, axis=1, side=0, traction=(0.3, -1.0, 0.1))
        solver = cmg.create_solver(lv, b, return_solution=False)
        u = solver(rho)
        g = jax.grad(lambda r: jnp.dot(jnp.asarray(b), solver(r)))(rho)
        k, res = solver.stats(rho)
        assert res < 1e-11 and k < 100
        out[prec] = (onp.asarray(u), onp.asarray(g))
        # Krylov coarse level runs in the preconditioner dtype too
        sk = cmg.create_solver(lv, b, solver_options=_KRYLOV, return_solution=False)
        assert onp.allclose(onp.asarray(sk(rho)), onp.asarray(u), rtol=1e-8, atol=1e-12)
    for i in range(2):
        assert onp.allclose(out["mixed"][i], out["double"][i], rtol=1e-8,
                            atol=1e-10 * onp.abs(out["double"][i]).max())
    with pytest.raises(ValueError):
        NarrowBandCMG(grid, fixed, material=mat, precision="single")
