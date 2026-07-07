"""Tests for feax.solvers.eigen: generalized_eigh, the buckling backends
(dense / sparse-ARPACK / cudss GPU shift-invert), and the differentiable
driver's analytic eigenvalue sensitivity."""

import numpy as onp
import pytest
import scipy.sparse as sp

import jax
import jax.numpy as jnp

import feax as fe
from feax.csr import CSRMatrix
from feax.solvers.eigen import (
    BucklingConvergenceError,
    create_linear_buckling_solver,
    generalized_eigh,
)


def has_gpu():
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def has_spineax():
    try:
        from spineax.cudss.factor_solve import factorize  # noqa: F401
        return has_gpu()
    except ImportError:
        return False


requires_cudss = pytest.mark.skipif(not has_spineax(),
                                    reason="spineax/cuDSS or GPU not available")


# ---------------------------------------------------------------------------
# generalized_eigh
# ---------------------------------------------------------------------------

def test_generalized_eigh_matches_scipy():
    import scipy.linalg as sla
    rng = onp.random.default_rng(0)
    n = 40
    A0 = rng.standard_normal((n, n)); A = (A0 + A0.T) / 2
    B0 = rng.standard_normal((n, n)); B = B0 @ B0.T + n * onp.eye(n)
    w, U = generalized_eigh(jnp.asarray(A), jnp.asarray(B))
    w_ref = sla.eigh(A, B, eigvals_only=True)
    assert onp.allclose(onp.sort(onp.asarray(w)), w_ref, atol=1e-10)
    # B-orthonormal eigenvectors: Uᵀ B U = I
    G = onp.asarray(U).T @ B @ onp.asarray(U)
    assert onp.allclose(G, onp.eye(n), atol=1e-8)


# ---------------------------------------------------------------------------
# buckling pencil fixture: K SPD, Kg symmetric (negative-definite-ish),
# so (K + λ Kg) φ = 0 has known positive eigenvalues via dense reference
# ---------------------------------------------------------------------------

def _pencil(n=150, seed=0, nnz_band=3):
    rng = onp.random.default_rng(seed)
    # K: banded SPD (Laplacian-like with random weights)
    diags = [2.0 + rng.random(n)]
    offs = [0]
    for k in range(1, nnz_band):
        diags.append(-rng.random(n - k) * 0.5)
        offs.append(k)
    K_sp = sp.diags(diags + diags[1:], offs + [-k for k in offs[1:]]).tocsr()
    K_sp = (K_sp + K_sp.T) * 0.5 + 2.0 * sp.eye(n)
    # Kg: banded symmetric negative definite (mass-like, sign-flipped)
    M = sp.diags([4.0 + rng.random(n), onp.ones(n - 1), onp.ones(n - 1)],
                 [0, 1, -1]).tocsr() * (1.0 / n)
    Kg_sp = (-(M + M.T) * 0.5).tocsr()

    def to_csr(A):
        A = A.tocsr(); A.sort_indices()
        return CSRMatrix(jnp.asarray(A.data), jnp.asarray(A.indptr),
                         jnp.asarray(A.indices), A.shape)

    free = onp.arange(3, n - 3)                       # constrain a few end DoFs
    return to_csr(K_sp), to_csr(Kg_sp), free


def _run(solver, num_modes=4, **kw):
    K, Kg, free = _pencil()
    bf = create_linear_buckling_solver(free, num_modes=num_modes, solver=solver, **kw)
    lam, modes = bf(K, Kg)
    return K, Kg, free, bf, onp.asarray(lam), onp.asarray(modes)


def test_dense_buckling_reference():
    K, Kg, free, bf, lam, modes = _run("dense")
    assert lam.shape == (4,) and (onp.diff(lam) >= -1e-12).all() and (lam > 0).all()
    # eigen-residual of the pencil: (K + λ Kg) φ ≈ 0 on the free DoFs
    Kf = onp.asarray(K.todense())[onp.ix_(free, free)]
    Kgf = onp.asarray(Kg.todense())[onp.ix_(free, free)]
    for i in range(4):
        phi = modes[i, free]
        r = onp.linalg.norm(Kf @ phi + lam[i] * (Kgf @ phi)) / onp.linalg.norm(Kf @ phi)
        assert r < 1e-8


def test_sparse_matches_dense():
    _, _, _, _, lam_d, _ = _run("dense")
    _, _, _, _, lam_s, _ = _run("sparse", num_matvecs=60)
    assert onp.allclose(lam_s, lam_d, rtol=1e-6)


@pytest.mark.cuda
@requires_cudss
def test_cudss_matches_dense():
    _, _, _, _, lam_d, _ = _run("dense")
    K, Kg, free, bf, lam_c, modes_c = _run("cudss", num_matvecs=60)
    assert onp.allclose(lam_c, lam_d, rtol=1e-6)
    # modes are K-normalized full-DoF vectors, zero on constrained DoFs
    fixed = onp.setdiff1d(onp.arange(K.shape[0]), free)
    assert onp.allclose(modes_c[:, fixed], 0.0)
    Kf = onp.asarray(K.todense())[onp.ix_(free, free)]
    for i in range(4):
        phi = modes_c[i, free]
        assert phi @ (Kf @ phi) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.cuda
@requires_cudss
def test_cudss_gradients_match_dense():
    # the analytic eigenvalue sensitivity uses the backend's φ: cudss and dense
    # must produce the same gradients w.r.t. the CSR values of K and Kg
    K, Kg, free = _pencil()

    def loss_with(solver, **kw):
        bf = create_linear_buckling_solver(free, num_modes=3, solver=solver, **kw)
        bf(K, Kg)                                     # eager call captures structure
        def loss(kd, gd):
            lam, _ = bf(CSRMatrix(kd, K.indptr, K.indices, K.shape),
                        CSRMatrix(gd, Kg.indptr, Kg.indices, Kg.shape))
            return jnp.sum(lam)
        return jax.grad(loss, argnums=(0, 1))

    gK_d, gG_d = loss_with("dense")(K.data, Kg.data)
    gK_c, gG_c = loss_with("cudss", num_matvecs=60)(K.data, Kg.data)
    sK = float(jnp.max(jnp.abs(gK_d))) + 1e-30
    sG = float(jnp.max(jnp.abs(gG_d))) + 1e-30
    assert float(jnp.max(jnp.abs(gK_c - gK_d))) / sK < 1e-5
    assert float(jnp.max(jnp.abs(gG_c - gG_d))) / sG < 1e-5

    # and the dense gradient itself agrees with finite differences on a
    # DIAGONAL slot (an off-diagonal single-slot perturbation would make K
    # asymmetric, which the backends implicitly re-symmetrize). The random
    # banded pencil has Anderson-localized modes, so pick the diagonal of the
    # dof where the fundamental mode actually lives — elsewhere the gradient
    # is exponentially small and FD compares noise against noise.
    bf_d = create_linear_buckling_solver(free, num_modes=3, solver="dense")
    lam_ref, modes_ref = bf_d(K, Kg)
    lam0 = float(jnp.sum(lam_ref))
    eps = 1e-6
    row = int(onp.argmax(onp.abs(onp.asarray(modes_ref)[0])))
    ip, ix = onp.asarray(K.indptr), onp.asarray(K.indices)
    slot = int(ip[row] + onp.nonzero(ix[ip[row]:ip[row + 1]] == row)[0][0])
    assert abs(float(gK_d[slot])) > 1e-3              # the slot is informative
    kd_p = K.data.at[slot].add(eps)
    lam_p = float(jnp.sum(create_linear_buckling_solver(
        free, num_modes=3, solver="dense")(CSRMatrix(kd_p, K.indptr, K.indices,
                                                     K.shape), Kg)[0]))
    fd = (lam_p - lam0) / eps
    assert float(gK_d[slot]) == pytest.approx(fd, rel=1e-3, abs=1e-9)


@pytest.mark.cuda
@requires_cudss
def test_cudss_no_positive_mode_raises():
    # a positive-definite Kg has no positive buckling factor: the backend must
    # fail loudly instead of returning junk
    K, Kg, free = _pencil()
    Kg_pd = CSRMatrix(-Kg.data, Kg.indptr, Kg.indices, Kg.shape)
    bf = create_linear_buckling_solver(free, num_modes=4, solver="cudss",
                                       num_matvecs=40)
    # the raise happens inside a host callback: under the GPU's async dispatch
    # it surfaces (wrapped) when the result is synchronized
    with pytest.raises(Exception, match="no positive eigenvalue"):
        jax.block_until_ready(bf(K, Kg_pd)[0])


def test_unknown_solver_rejected():
    with pytest.raises(ValueError, match="unknown buckling solver"):
        create_linear_buckling_solver(onp.arange(4), solver="nope")
