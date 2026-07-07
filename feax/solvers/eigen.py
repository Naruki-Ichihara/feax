"""Linear-buckling eigensolvers for feax, in JAX.

Solves the linear-buckling pencil ``(K + λ K_g) φ = 0`` for the lowest positive
buckling factors ``λ`` and their mode shapes, with four forward backends:

* ``solver="sparse"`` (:func:`_sparse_buckling`, **default**) — matrix-free shift-invert
  ``T = -K⁻¹ K_g`` solved by SciPy ARPACK (:func:`scipy.sparse.linalg.eigs`) with one
  sparse factorization for ``K⁻¹``. ``O(nf · ncv)`` memory, never a dense ``N×N``. Runs
  purely host-side (no nested ``jax.pure_callback``), so it is leak-free under repeated
  calls (e.g. inside topology optimization).
* ``solver="dense"`` (:func:`_dense_buckling`) — Cholesky reduction of the generalized
  symmetric-definite pencil to a standard ``jnp.linalg.eigh`` on the free DoFs.
  Exact and robust, but ``O(nf²)`` memory; for small/medium reduced systems.
* ``solver="matfree"`` (:func:`_matfree_buckling`, opt-in) — the same shift-invert via
  matfree's JAX Arnoldi. Equivalent results, but applies ``K⁻¹`` through a nested
  ``jax.pure_callback`` that accumulates in the XLA runtime across calls (memory leak in
  long loops); prefer ``"sparse"`` unless you specifically need the JAX path.
* ``solver="cudss"`` (:func:`_create_cudss_bf`, opt-in, GPU) — the same shift-invert
  with ``K`` factorized ONCE per call on the GPU by cuDSS (spineax
  factor-once/solve-many) and the Arnoldi loop as traced JAX (``K_g`` SpMV + cuDSS
  SOLVE per ``lax.scan`` step); only the tiny Hessenberg eig runs in a host callback.
  No per-call closures; factors live in spineax's fixed-capacity LRU (no accumulation
  over optimization loops). Requires spineax + CUDA.

The differentiable driver :func:`create_linear_buckling_solver` wraps any backend with
an analytical eigenvalue sensitivity, so ``λ`` is ``jax.grad``-able w.r.t. the assembled
``K`` / ``K_g`` (and ``bf`` itself stays jittable — the eigensolve is an opaque host
callback regardless of backend).
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax.scipy.linalg import solve_triangular

from feax.csr import CSRMatrix

try:                                   # matfree: optional opt-in backend (solver="matfree")
    from matfree import eig as _mf_eig, decomp as _mf_decomp
    _HAVE_MATFREE = True
except Exception:                      # pragma: no cover
    _HAVE_MATFREE = False


class BucklingConvergenceError(RuntimeError):
    """Raised when the matrix-free buckling eigensolve fails to converge.

    Lets a caller (e.g. a topology-optimization loop) detect a bad solve early and
    react — instead of silently propagating garbage eigenvalues / sensitivities.
    """


def generalized_eigh(A: jnp.ndarray, B: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the generalized symmetric eigenproblem ``A x = λ B x`` (``B ≻ 0``).

    Parameters
    ----------
    A : (n, n) array
        Symmetric matrix (symmetrised internally for round-off).
    B : (n, n) array
        Symmetric positive-definite matrix.

    Returns
    -------
    w : (n,) array
        Eigenvalues ``λ`` in ascending order.
    X : (n, n) array
        Eigenvectors as columns, B-orthonormal (``Xᵀ B X = I``).

    Notes
    -----
    Implemented via the Cholesky reduction ``C = L⁻¹ A L⁻ᵀ`` with ``B = L Lᵀ``; the
    eigenvalues of ``C`` equal those of the pencil and ``x = L⁻ᵀ u``.
    """
    A = jnp.asarray(A); B = jnp.asarray(B)
    A = 0.5 * (A + A.T)
    L = jnp.linalg.cholesky(B)                       # B = L Lᵀ (L lower-triangular)
    Y = solve_triangular(L, A, lower=True)           # L⁻¹ A
    C = solve_triangular(L, Y.T, lower=True).T       # L⁻¹ A L⁻ᵀ  (symmetric)
    C = 0.5 * (C + C.T)
    w, U = jnp.linalg.eigh(C)                         # standard symmetric eig
    X = solve_triangular(L.T, U, lower=False)        # x = L⁻ᵀ u
    return w, X


def _dense_buckling(K, Kg, free_dofs, num_modes: int = 6,
                    zero_tol: float = 1e-8, verbose: bool = False
                    ) -> Tuple[onp.ndarray, onp.ndarray]:
    """Dense generalized-eigh buckling core (non-differentiable; returns NumPy).

    Used as the forward solve inside :func:`create_linear_buckling_solver` (``solver=
    "dense"``). Prefer that differentiable wrapper in user code.

    Linear buckling factors and modes of ``(K + λ K_g) φ = 0``.

    Solves the pencil on the unconstrained DoFs (``free_dofs``) as the symmetric /
    positive-definite generalized problem ``K_g x = γ K x`` via :func:`generalized_eigh`
    (so ``K`` restricted to the free DoFs must be positive definite — i.e. rigid-body
    and Dirichlet DoFs excluded). The buckling factor is ``λ = -1/γ``; the lowest
    positive ``λ`` scales the reference load to the critical load.

    Parameters
    ----------
    K, Kg : (N, N) CSRMatrix or dense array
        Material tangent stiffness and geometric (initial-stress) stiffness.
    free_dofs : int array
        Indices of the unconstrained DoFs.
    num_modes : int
        Number of lowest positive buckling modes to return.
    zero_tol : float
        Buckling factors with ``|λ|`` below this (rigid/spurious) are discarded.

    Returns
    -------
    lambdas : (m,) array
        Ascending positive buckling factors (``m = min(num_modes, #positive)``).
    modes : (m, N) array
        Mode shapes scattered back onto the full DoF vector (zero on constrained DoFs).
    """
    N = K.shape[0]
    free = onp.asarray(free_dofs)
    nf = len(free)
    if verbose:
        jax.debug.print("[buckling/dense] dense generalized eigh | N={N}, free={nf}, "
                        "constrained={nc}, num_modes={m}",
                        N=N, nf=nf, nc=N - nf, m=num_modes)
    if isinstance(K, CSRMatrix):
        # slice to the free DoFs while still sparse, then densify only the small block
        Kf = jnp.asarray(_restrict_csr(K, free).toarray())
        Kgf = jnp.asarray(_restrict_csr(Kg, free).toarray())
    else:
        fj = jnp.asarray(free)
        Kf = jnp.asarray(K)[jnp.ix_(fj, fj)]
        Kgf = jnp.asarray(Kg)[jnp.ix_(fj, fj)]
    if verbose:
        jax.debug.print("[buckling/dense]   reduced blocks densified {sh}; "
                        "min diag(K_free)={d:.3e} (>0 needed for Cholesky)",
                        sh=Kf.shape, d=jnp.min(jnp.diag(Kf)))

    gamma, Vf = generalized_eigh(Kgf, Kf)            # K_g x = γ K x
    if verbose:
        # eigen-residual ‖K_g x − γ K x‖/‖K_g x‖ for every computed pair
        KgX = Kgf @ Vf; KX = Kf @ Vf
        res_all = (jnp.linalg.norm(KgX - gamma[None, :] * KX, axis=0)
                   / (jnp.linalg.norm(KgX, axis=0) + 1e-30))
        jax.debug.print("[buckling/dense]   Cholesky+eigh done; gamma in "
                        "[{lo:.3e}, {hi:.3e}]; max eig-residual={r:.2e}",
                        lo=jnp.min(gamma), hi=jnp.max(gamma), r=jnp.max(res_all))
    gamma = onp.asarray(gamma); Vf = onp.asarray(Vf)

    lam = -1.0 / gamma
    valid = (lam > zero_tol) & onp.isfinite(lam)
    order = onp.argsort(onp.where(valid, lam, onp.inf))[:num_modes]
    order = order[valid[order]]                      # keep only genuine positive modes
    if verbose:
        Kfn = onp.asarray(Kf); Kgfn = onp.asarray(Kgf)
        jax.debug.print("[buckling/dense]   positive buckling factors found = {n}",
                        n=int(valid.sum()))
        for i, c in enumerate(order):
            x = Vf[:, c]
            res = (onp.linalg.norm(Kgfn @ x - gamma[c] * (Kfn @ x))
                   / (onp.linalg.norm(Kgfn @ x) + 1e-30))
            jax.debug.print("[buckling/dense]   mode {i}: lambda={l:.6g}  gamma={g:.4e}  "
                            "resid={r:.2e}", i=i + 1, l=lam[c], g=gamma[c], r=res)

    modes = onp.zeros((len(order), N))
    modes[:, free] = Vf[:, order].T
    return lam[order], modes


def _to_scipy_csr(M):
    """A :class:`~feax.csr.CSRMatrix` -> SciPy CSR, with no COO re-sort.

    The assembled matrix is already CSR (deduplicated, within-row sorted), so the
    SciPy matrix is built directly from ``(data, indices, indptr)`` — O(nnz), no
    COO -> CSR conversion.
    """
    import scipy.sparse as sp
    return sp.csr_matrix(
        (onp.asarray(M.data), onp.asarray(M.indices), onp.asarray(M.indptr)),
        shape=M.shape)


def _restrict_csr(M, free):
    """``free × free`` sub-matrix of a :class:`~feax.csr.CSRMatrix` as SciPy CSR.

    Pure NumPy/SciPy (used inside the host eigensolve callback, where dispatching
    JAX ops would leak runtime state across repeated calls). The input is already
    CSR, so this is a direct ``csr_matrix(...)[free][:, free]`` — no COO resort.
    """
    free = onp.asarray(free)
    sub = _to_scipy_csr(M)[free][:, free]
    return sub.tocsr()


def _restrict_csr_jax(M, free):
    """``free × free`` sub-matrix of a CSRMatrix as a JAX :class:`~feax.csr.CSRMatrix`.

    Built host-side (SciPy slice) but returned with ``jax.numpy`` leaves so it
    supports ``@`` inside a traced Arnoldi (the ``matfree`` backend). Indices are
    sorted so the CSRMatrix mat-vec stays a deterministic segment-sum.
    """
    sub = _restrict_csr(M, free)
    sub.sort_indices()
    return CSRMatrix(jnp.asarray(sub.data), jnp.asarray(sub.indptr),
                     jnp.asarray(sub.indices), sub.shape)


def _make_host_solve(A):
    """Factorize a reduced SPD stiffness (SciPy sparse ``A``); return a host ``K⁻¹`` apply.

    The returned ``solve(b)`` is plain SciPy/NumPy — **no** JAX. The buckling eigensolve
    already runs host-side (inside the differentiable wrapper's ``pure_callback``); any
    JAX op or nested ``pure_callback`` there accumulates runtime state across calls and
    leaks memory until OOM. Uses CHOLMOD (``sksparse``) when available, else SciPy
    ``splu``.
    """
    import scipy.sparse as sp
    A = sp.csc_matrix(A)
    A = (0.5 * (A + A.T)).tocsc()                     # clean round-off asymmetry

    factor = None                                    # try CHOLMOD (sparse Cholesky)
    try:
        from sksparse.cholmod import cholesky as _chol
        f = _chol(A)
        if callable(f):                              # some builds return a non-Factor; guard
            factor = f
    except Exception:
        factor = None
    if factor is not None:
        return (lambda b: onp.asarray(factor(b), dtype=onp.float64)), "cholmod"
    try:
        lu = sp.linalg.splu(A)                        # SuperLU LU
    except (RuntimeError, MemoryError) as e:          # near-singular / out-of-memory
        raise BucklingConvergenceError(
            f"sparse factorization of K failed ({type(e).__name__}: {e}); K is "
            "likely near-singular on the free DoFs (raise the SIMP void floor or "
            "constrain more rigid modes).") from e
    return (lambda b: onp.asarray(lu.solve(onp.asarray(b, dtype=onp.float64)),
                                  dtype=onp.float64)), "splu"


def _make_kinv_apply(Kf_sp):
    """JAX-callable ``K⁻¹`` apply via :func:`jax.pure_callback` (matfree backend only).

    ``Kf_sp`` is a reduced SciPy CSR/CSC stiffness on the free DoFs.

    .. warning::
        The matfree path runs JAX (its Arnoldi ``lax.scan`` / ``jnp.linalg.eig`` and this
        nested ``K⁻¹`` ``pure_callback``) *inside* the host eigensolve callback. That
        per-call JAX dispatch accumulates in the runtime and leaks memory (~tens of
        MB/iter) over a long loop — it cannot be fully avoided while using matfree here.
        Prefer the default ``solver="sparse"`` (SciPy ARPACK, :func:`_sparse_buckling`),
        which is pure host-side and leak-free; reserve ``"matfree"`` for single solves.
    """
    solve, backend = _make_host_solve(Kf_sp)

    def kinv(b):
        return jax.pure_callback(
            lambda x: solve(onp.asarray(x, dtype=onp.float64)),
            jax.ShapeDtypeStruct(b.shape, b.dtype), b)
    return kinv, backend


def _sparse_buckling(K, Kg, free_dofs, num_modes: int = 6, num_matvecs: int = 30,
                     seed: int = 0, conv_tol: float = 1e-2, verbose: bool = False
                     ) -> Tuple[onp.ndarray, onp.ndarray]:
    """Sparse, matrix-free linear buckling via SciPy ARPACK shift-invert (host-side).

    The default ``solver="sparse"`` backend. Recast ``(K + λ K_g) φ = 0`` as

    .. code-block:: text

        T = -K⁻¹ K_g ,   T φ = μ φ   with   μ = 1/λ,

    and find the largest-magnitude ``μ`` (the smallest positive ``λ``) with ARPACK
    (:func:`scipy.sparse.linalg.eigs`) driving a ``LinearOperator`` ``v ↦ -K⁻¹(K_g v)``.
    ``K⁻¹`` is one sparse factorization (CHOLMOD / SuperLU). Everything runs in plain
    SciPy/NumPy — **no** ``jax.pure_callback`` inside the eigensolve — so repeated calls
    in an optimization loop do not accumulate runtime callbacks / factors (no leak).
    Memory is the sparse factor + an ``O(nf · ncv)`` Krylov basis (never a dense ``N×N``).

    Parameters
    ----------
    K, Kg : (N, N) CSRMatrix
        Material tangent stiffness and geometric (initial-stress) stiffness.
    free_dofs : int array
        Indices of the unconstrained DoFs (``K`` restricted to them must be SPD).
    num_modes : int
        Number of lowest positive buckling modes to return.
    num_matvecs : int
        Lower bound on the ARPACK Arnoldi basis size ``ncv`` (``ncv ≥ 2k+1`` is enforced).
    conv_tol : float or None
        Tolerance on the fundamental Ritz pair's residual ``‖Tφ−μφ‖/‖φ‖``; raise
        :class:`BucklingConvergenceError` early if exceeded / no positive mode found.
        ``None`` disables the check.
    seed, verbose
        RNG seed for the ARPACK starting vector; debug output.

    Returns
    -------
    lambdas : (m,) array       Ascending positive buckling factors.
    modes : (m, N) array       Mode shapes scattered onto the full DoF vector.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    N = K.shape[0]
    free = onp.asarray(free_dofs)
    nf = len(free)
    Kg_sp = _restrict_csr(Kg, free)                  # reduced sub-matrices (pure SciPy)
    solve, backend = _make_host_solve(_restrict_csr(K, free))   # factorize K once (host)

    def matvec(v):                                   # T v = -K⁻¹ (K_g v)
        return -solve(Kg_sp @ onp.asarray(v, dtype=onp.float64))

    T = spla.LinearOperator((nf, nf), matvec=matvec, dtype=onp.float64)
    k = int(min(num_modes + 4, nf - 2))
    if k < 1:
        raise BucklingConvergenceError(
            f"too few free DoFs ({nf}) for the sparse eigensolve; use solver='dense'.")
    ncv = int(min(nf - 1, max(num_matvecs, 2 * k + 1)))
    v0 = onp.random.RandomState(seed).randn(nf)
    if verbose:
        print(f"[buckling/sparse] ARPACK shift-invert (K⁻¹={backend}) | "
              f"N={N}, free={nf}, k={k}, ncv={ncv}")
    try:
        vals, vecs = spla.eigs(T, k=k, which="LM", ncv=ncv, v0=v0)
    except spla.ArpackNoConvergence as e:            # use whatever converged
        vals, vecs = e.eigenvalues, e.eigenvectors
        if vals.size == 0:
            raise BucklingConvergenceError(
                f"ARPACK did not converge (k={k}, ncv={ncv}); increase num_matvecs "
                "or check that K is SPD on the free DoFs.") from e

    mu = vals.real
    real_pair = onp.abs(vals.imag) <= 1e-8 * (onp.abs(vals.real) + 1e-30)
    lam = onp.where(mu != 0.0, 1.0 / mu, onp.inf)
    valid = real_pair & onp.isfinite(lam) & (lam > 1e-12)
    order = onp.argsort(onp.where(valid, lam, onp.inf))
    order = order[valid[order]][:num_modes]

    def _resid(c):                                   # eigen-residual ‖Tφ − μφ‖/‖φ‖
        x = vecs[:, c].real
        return float(onp.linalg.norm(matvec(x) - mu[c] * x) / (onp.linalg.norm(x) + 1e-30))

    if conv_tol is not None:
        if len(order) == 0:
            raise BucklingConvergenceError(
                "sparse buckling found no positive eigenvalue; the solve did not "
                "converge.")
        res0 = _resid(order[0])
        if not onp.isfinite(res0) or res0 > conv_tol:
            raise BucklingConvergenceError(
                f"sparse buckling not converged: fundamental-mode residual {res0:.2e} "
                f"> tol {conv_tol:.1e}. Increase num_matvecs, or check that K is SPD "
                "on the free DoFs.")
    if verbose:
        for i, c in enumerate(order):
            print(f"[buckling/sparse]   mode {i + 1}: lambda={lam[c]:.6g}  "
                  f"mu={mu[c]:.4e}  resid={_resid(c):.2e}")

    modes = onp.zeros((len(order), N))
    modes[:, free] = vecs[:, order].real.T
    return lam[order], modes


def _create_cudss_bf(free_dofs, num_modes: int = 3, num_matvecs: int = 30,
                     seed: int = 0, conv_tol: float = 1e-2, verbose: bool = False,
                     device_id: int = 0, matrix_type: str = "spd"):
    """Build the ``solver="cudss"`` buckling function: GPU shift-invert Arnoldi.

    Same recast as :func:`_sparse_buckling` — ``T = -K⁻¹ K_g``, largest-magnitude
    ``μ`` give the smallest positive ``λ = 1/μ`` — but the forward is TRACED JAX:
    ``K`` (restricted to the free DoFs) is factorized ONCE per call by cuDSS
    (``spineax.cudss.factor_solve``), and each Arnoldi iteration inside a
    ``lax.scan`` is one device SpMV (``K_g @ v``) plus one cuDSS SOLVE reusing
    those factors, with full (two-pass) reorthogonalization. Only the small
    ``(m, m)`` Hessenberg eigendecomposition and the convergence check run in
    host callbacks (pure NumPy — no device work inside a callback, which
    deadlocks the GPU stream; that is why this backend does NOT go through the
    driver's host-eigensolve callback like the others).

    Leak/lifetime notes (this module's recurring theme):

    * The factorization lives in spineax's process-global fixed-capacity LRU
      cache — repeated calls in an optimization loop factorize the current ``K``
      and let the LRU evict stale factors. Nothing accumulates per call.
    * The two host callbacks are single function objects created once here and
      reused every call (no per-call callables registered with the runtime).

    Differentiability comes from the same analytic eigenvalue sensitivity /
    ``custom_vjp`` as the other backends (see
    :func:`create_linear_buckling_solver`); the forward internals need not be
    differentiable. Like the other backends, call ``bf(K, Kg)`` once eagerly
    before jitting so the (constant) CSR structure can be captured.

    Extra kwargs: ``device_id`` (CUDA device) and ``matrix_type``
    (``"spd"`` Cholesky, default, or ``"symmetric"`` LDLᵀ — more robust when a
    SIMP void floor leaves ``K`` near-indefinite).
    """
    try:
        from spineax.cudss.factor_solve import factorize, solve_with
    except ImportError as e:                          # pragma: no cover
        raise ImportError(
            "solver='cudss' requires spineax (spineax.cudss.factor_solve). "
            "Install spineax, or use solver='sparse'/'dense'.") from e
    mtype_ids = {"spd": 3, "symmetric": 1}
    if matrix_type not in mtype_ids:
        raise ValueError(f"matrix_type must be 'spd' or 'symmetric', got {matrix_type!r}")

    free = onp.asarray(free_dofs)
    nf = len(free)
    mo = int(num_modes)
    m = int(min(nf, max(num_matvecs, 2 * mo + 10)))
    if m < 2:
        raise BucklingConvergenceError(
            f"too few free DoFs ({nf}) for the cudss eigensolve; use solver='dense'.")

    # Constant structure, captured from the first eager call: value-gather maps
    # full CSR slots -> reduced (free x free) CSR slots, so the per-call
    # restriction is a traced gather (no SciPy inside the traced forward).
    _s = {}

    def _capture(K, Kg):
        if _s:
            return
        if isinstance(K.indices, jax.core.Tracer) or isinstance(Kg.indices, jax.core.Tracer):
            raise ValueError(
                "create_linear_buckling_solver(solver='cudss'): call bf(K, Kg) once "
                "eagerly before jitting so the (constant) CSR structure can be captured.")
        import scipy.sparse as sp

        def restrict_map(M):
            nnz = int(onp.asarray(M.indices).shape[0])
            A = sp.csr_matrix((onp.arange(1, nnz + 1, dtype=onp.float64),
                               onp.asarray(M.indices), onp.asarray(M.indptr)),
                              shape=M.shape)
            R = A[free][:, free].tocsr()
            R.sort_indices()
            slots = jnp.asarray(R.data.astype(onp.int64) - 1)
            return slots, R

        slotK, RK = restrict_map(K)
        slotG, RG = restrict_map(Kg)
        # transpose permutation on the reduced K pattern (structurally symmetric
        # for FEM operators) -> traced symmetrization kf := (kf + kf[Tperm]) / 2
        P = sp.csr_matrix((onp.arange(1, RK.nnz + 1, dtype=onp.float64),
                           RK.indices, RK.indptr), shape=(nf, nf)).T.tocsr()
        P.sort_indices()
        _s.update(slotK=slotK, slotG=slotG,
                  TpermK=jnp.asarray(P.data.astype(onp.int64) - 1),
                  ipK=jnp.asarray(RK.indptr, jnp.int32),
                  ixK=jnp.asarray(RK.indices, jnp.int32),
                  ipG=jnp.asarray(RG.indptr), ixG=jnp.asarray(RG.indices),
                  N=int(K.shape[0]))

    v0_host = onp.random.RandomState(seed).randn(nf)

    # --- host callbacks: created ONCE, pure NumPy (no device work inside) -----
    def _host_ritz(H):
        H = onp.asarray(H)
        vals, Y = onp.linalg.eig(H[:m, :m])
        mu = vals.real
        real_pair = onp.abs(vals.imag) <= 1e-8 * (onp.abs(vals.real) + 1e-30)
        lam = onp.where(mu != 0.0, 1.0 / mu, onp.inf)
        valid = real_pair & onp.isfinite(lam) & (lam > 1e-12)
        order = onp.argsort(onp.where(valid, lam, onp.inf))
        order = order[valid[order]][:mo]
        if conv_tol is not None and len(order) == 0:
            raise BucklingConvergenceError(
                f"cudss buckling found no positive eigenvalue (num_matvecs={m}); "
                "the eigensolve did not converge.")
        lam_out = onp.full(mo, onp.inf)
        mu_out = onp.zeros(mo)
        coeff = onp.zeros((mo, m + 1))
        for i, c in enumerate(order):
            lam_out[i] = lam[c]
            mu_out[i] = mu[c]
            y = Y[:, c].real
            coeff[i, :m] = y / (onp.linalg.norm(y) + 1e-30)
        return lam_out, mu_out, coeff

    def _host_check(lam, res0):
        lam = onp.asarray(lam)
        res0 = float(res0)
        if conv_tol is not None and (not onp.isfinite(res0) or res0 > conv_tol):
            raise BucklingConvergenceError(
                f"cudss buckling not converged: fundamental-mode residual {res0:.2e} "
                f"> tol {conv_tol:.1e} (num_matvecs={m}). Increase num_matvecs, or "
                "check that K is SPD on the free DoFs.")
        if verbose:
            print(f"[buckling/cudss] modes: lambda={lam.tolist()}  resid0={res0:.2e}")
        return lam

    def _forward(K, Kg):
        _capture(K, Kg)
        N = _s["N"]
        kf = K.data[_s["slotK"]]
        kf = 0.5 * (kf + kf[_s["TpermK"]])           # clean round-off asymmetry
        Kg_red = CSRMatrix(Kg.data[_s["slotG"]], _s["ipG"], _s["ixG"], (nf, nf))
        Kf_red = CSRMatrix(kf, _s["ipK"], _s["ixK"], (nf, nf))
        token = factorize(kf, _s["ipK"], _s["ixK"], device_id=device_id,
                          mtype_id=mtype_ids[matrix_type], mview_id=0)

        def T_apply(v):                              # T v = -K⁻¹ (K_g v), on device
            return -solve_with(token, Kg_red @ v, device_id=device_id)

        v0 = jnp.asarray(v0_host)
        V0 = jnp.zeros((m + 1, nf)).at[0].set(v0 / jnp.linalg.norm(v0))

        def step(carry, j):
            V, H = carry
            w = T_apply(V[j])
            h1 = V @ w                               # rows > j are zero -> no-ops
            w = w - V.T @ h1
            h2 = V @ w                               # second Gram-Schmidt pass
            w = w - V.T @ h2
            beta = jnp.linalg.norm(w)
            H = H.at[:, j].set((h1 + h2).at[j + 1].set(beta))
            V = V.at[j + 1].set(jnp.where(beta > 1e-30, 1.0 / beta, 0.0) * w)
            return (V, H), None

        (V, H), _ = jax.lax.scan(step, (V0, jnp.zeros((m + 1, m))), jnp.arange(m))

        lam, mu, coeff = jax.pure_callback(
            _host_ritz,
            (jax.ShapeDtypeStruct((mo,), jnp.float64),
             jax.ShapeDtypeStruct((mo,), jnp.float64),
             jax.ShapeDtypeStruct((mo, m + 1), jnp.float64)),
            H)

        phi = coeff @ V                              # (mo, nf) Ritz vectors
        # φᵀ K φ = 1 normalization (zero rows from padding stay zero)
        e = jnp.stack([phi[i] @ (Kf_red @ phi[i]) for i in range(mo)])
        s = jnp.where(e > 0.0, 1.0 / jnp.sqrt(jnp.abs(e) + 1e-300), 0.0)
        phi = phi * s[:, None]

        # fundamental-mode eigen-residual ‖Tφ − μφ‖ / ‖φ‖ (traced, one extra solve)
        x0 = phi[0] / (jnp.linalg.norm(phi[0]) + 1e-30)
        res0 = jnp.linalg.norm(T_apply(x0) - mu[0] * x0)
        lam = jax.pure_callback(
            _host_check, jax.ShapeDtypeStruct((mo,), jnp.float64), lam, res0)

        modes = jnp.zeros((mo, N)).at[:, free].set(phi)
        return lam, modes, phi

    @jax.custom_vjp
    def bf(K, Kg):
        """(lambdas, modes): lambdas differentiable w.r.t. K, Kg (analytic
        sensitivity); modes carry no gradient."""
        lam, modes, _ = _forward(K, Kg)
        return lam, jax.lax.stop_gradient(modes)

    def _fwd(K, Kg):
        lam, modes, phi = _forward(K, Kg)
        return (lam, jax.lax.stop_gradient(modes)), (K, Kg, lam, phi)

    def _bwd(res, cts):
        K, Kg, lam, phi = res
        lam_bar, _modes_bar = cts
        # guard the inf-padded (missing) modes out of the sensitivity
        w = jnp.where(jnp.isfinite(lam), lam_bar * lam, 0.0)
        w2 = jnp.where(jnp.isfinite(lam), lam_bar * lam * lam, 0.0)
        K_cot = _outer_cotangent(K, phi, w, free)
        Kg_cot = _outer_cotangent(Kg, phi, w2, free)
        return (K_cot, Kg_cot)

    bf.defvjp(_fwd, _bwd)
    return bf


def _matfree_buckling(K, Kg, free_dofs, num_modes: int = 6, num_matvecs: int = 30,
                     seed: int = 0, conv_tol: float = 1e-2, verbose: bool = False
                     ) -> Tuple[onp.ndarray, onp.ndarray]:
    """Opt-in (``solver="matfree"``) buckling via matfree's JAX Arnoldi eig.

    Equivalent to :func:`_sparse_buckling` but applies ``K⁻¹`` through a nested
    ``jax.pure_callback`` (see :func:`_make_kinv_apply`), which accumulates in the XLA
    runtime across repeated calls and leaks memory in long loops. Prefer the default
    ``solver="sparse"`` (SciPy ARPACK) unless you specifically need the JAX path.

    Recast ``(K + λ K_g) φ = 0`` as the shift-invert standard eigenproblem

    .. code-block:: text

        T = -K⁻¹ K_g ,   T φ = μ φ   with   μ = 1/λ,

    whose largest positive ``μ`` give the smallest positive buckling factors ``λ = 1/μ``.
    ``K⁻¹`` is applied by a single sparse factorization (SuperLU ``splu`` via
    :func:`_make_kinv_apply`); the eigenpairs come from matfree's *non-symmetric*
    Arnoldi factorization (``matfree.eig.eig_partial``).

    Arnoldi works directly on the standard-orthonormal Krylov basis (no reduced
    subspace + generalized Rayleigh–Ritz, whose ``UᵀKU`` Cholesky would corrupt mode
    shapes on ill-conditioned ``K``), so the Ritz eigenvectors are the true buckling
    modes — eigenvalues match the dense generalized solver to near machine precision and
    the modes carry no spurious concentration. ``O(nf · num_matvecs)`` memory.

    Parameters
    ----------
    K, Kg : (N, N) CSRMatrix
        Material tangent stiffness and geometric (initial-stress) stiffness.
    free_dofs : int array
        Indices of the unconstrained DoFs (``K`` restricted to them must be SPD).
    num_modes : int
        Number of lowest positive buckling modes to return.
    num_matvecs : int
        Arnoldi/Krylov subspace size (number of mat-vecs). Must exceed ``num_modes``;
        larger gives more/secondary modes and better-resolved eigenpairs at higher cost.
    conv_tol : float or None
        Convergence tolerance on the fundamental Ritz pair's residual ``‖Tφ−μφ‖/‖φ‖``.
        If exceeded (or no positive mode is found), raise :class:`BucklingConvergenceError`
        early. ``None`` disables the check.
    seed, verbose
        RNG seed for the starting vector; debug output via ``jax.debug.print``.

    Returns
    -------
    lambdas : (m,) array       Ascending positive buckling factors.
    modes : (m, N) array       Mode shapes scattered onto the full DoF vector.
    """
    if not _HAVE_MATFREE:
        raise ImportError("_matfree_buckling requires the 'matfree' package "
                          "(pip install matfree).")
    N = K.shape[0]
    free = onp.asarray(free_dofs)
    nf = len(free)
    Kf_sp = _restrict_csr(K, free)                   # reduced K (SciPy, for the factor)
    Kgf = _restrict_csr_jax(Kg, free)                # reduced K_g (JAX CSRMatrix, for @)
    nkv = int(min(num_matvecs, nf - 1))
    kinv, backend = _make_kinv_apply(Kf_sp)          # factorize K once (host)
    if verbose:
        jax.debug.print("[buckling/sparse] Arnoldi shift-invert (K⁻¹={bk}) | "
                        "N={N}, free={nf}, num_matvecs={k}",
                        bk=backend, N=N, nf=nf, k=nkv)

    v0 = jnp.asarray(onp.random.RandomState(seed).randn(nf))

    def T(v):                                        # μ = 1/λ eigenproblem  T φ = μ φ
        return -kinv(Kgf @ v)

    eig_fun = _mf_eig.eig_partial(_mf_decomp.hessenberg(nkv, reortho="full"))

    # Run the Arnoldi factorization EAGER — do NOT wrap it in a per-call ``jax.jit``.
    # A fresh jitted closure on every call would (a) trigger a full XLA compile each
    # time and (b) retain the compiled executable together with its host K⁻¹ callback
    # (which holds the large sparse factor) in JAX's cache, accumulating across an
    # optimization loop until the host runs out of memory.
    vals, vecs = eig_fun(T, v0)                       # vals (nkv,) complex; vecs (nkv, nf)
    vals = onp.asarray(vals); vecs = onp.asarray(vecs)

    mu = vals.real
    real_pair = onp.abs(vals.imag) <= 1e-8 * (onp.abs(vals.real) + 1e-30)
    lam = onp.where(mu != 0.0, 1.0 / mu, onp.inf)
    valid = real_pair & onp.isfinite(lam) & (lam > 1e-12)
    order = onp.argsort(onp.where(valid, lam, onp.inf))
    order = order[valid[order]][:num_modes]

    def _resid(c):                                   # eigen-residual ‖Tφ − μφ‖/‖φ‖
        x = vecs[c].real
        return float(onp.linalg.norm(onp.asarray(T(jnp.asarray(x))) - mu[c] * x)
                     / (onp.linalg.norm(x) + 1e-30))

    # Fail fast on non-convergence: stop here (before the eigenvalues feed the K_g
    # sensitivity / the optimizer) if no positive mode was found or the fundamental
    # Ritz pair is unconverged. ``conv_tol=None`` disables the check.
    if conv_tol is not None:
        if len(order) == 0:
            raise BucklingConvergenceError(
                f"matfree buckling found no positive eigenvalue (num_matvecs={nkv}); "
                "the eigensolve did not converge.")
        res0 = _resid(order[0])
        if not onp.isfinite(res0) or res0 > conv_tol:
            raise BucklingConvergenceError(
                f"matfree buckling not converged: fundamental-mode residual {res0:.2e} "
                f"> tol {conv_tol:.1e} (num_matvecs={nkv}). Increase num_matvecs, or "
                "check that K is SPD on the free DoFs.")
    if verbose:
        for i, c in enumerate(order):
            jax.debug.print("[buckling/sparse]   mode {i}: lambda={l:.6g}  mu={g:.4e}  "
                            "resid={r:.2e}", i=i + 1, l=lam[c], g=mu[c], r=_resid(c))

    modes = onp.zeros((len(order), N))
    modes[:, free] = vecs[order].real
    return lam[order], modes


def _outer_cotangent(M, phi, w, free):
    """Cotangent :class:`~feax.csr.CSRMatrix` for a matrix input from the eigenvalue
    sensitivity.

    For stored entry ``(r, c)`` of ``M`` the contribution is ``Σ_i w_i φ_i[r] φ_i[c]``
    (rows/cols mapped to the free numbering; non-free entries get 0). The result
    shares ``M``'s structure (``indptr`` / ``indices``) so the custom_vjp cotangent
    matches the primal CSRMatrix pytree exactly.
    """
    N = M.shape[0]
    remap = -onp.ones(N, dtype=onp.int64)
    remap[onp.asarray(free)] = onp.arange(len(free))
    remap = jnp.asarray(remap)
    rows = M._row_of_slot(); cols = M.indices         # row/col of every CSR slot
    rf = remap[rows]; cf = remap[cols]
    ok = (rf >= 0) & (cf >= 0)
    rf = jnp.where(ok, rf, 0); cf = jnp.where(ok, cf, 0)
    contrib = jnp.sum(w[:, None] * phi[:, rf] * phi[:, cf], axis=0)
    return CSRMatrix(jnp.where(ok, contrib, 0.0), M.indptr, M.indices, M.shape)


def create_linear_buckling_solver(free_dofs, num_modes: int = 3, solver: str = "sparse",
                                  **solver_kw):
    """Return a ``jax.grad``-able function ``bf(K, Kg) -> λ`` for buckling factors.

    The forward eigensolve (sparse matrix-free :func:`_matfree_buckling` or dense
    :func:`_dense_buckling`, selected by ``solver``) runs through a
    :func:`jax.pure_callback`; differentiability is supplied by a
    :func:`jax.custom_vjp` using the **analytical eigenvalue sensitivity**

    .. code-block:: text

        (K + λ K_g) φ = 0,   φᵀ K φ = 1   ⇒   dλ = λ φᵀ dK φ + λ² φᵀ dK_g φ.

    Because only eigenvalues (not eigenvectors) are differentiated, this is robust to
    the degenerate buckling-mode pairs that make a naive ``eigh`` VJP blow up. Gradients
    flow to ``K.data`` and ``K_g.data`` (hence to any upstream design parameters that
    assemble them), e.g. for buckling-load optimization.

    Parameters
    ----------
    free_dofs : int array
        Unconstrained DoFs (captured; the returned function differentiates only K, Kg).
    num_modes : int
        Number of buckling factors returned.
    solver : {"sparse", "dense", "matfree", "cudss"}
        Forward eigensolver. ``"sparse"`` (default) forwards ``solver_kw`` to
        :func:`_sparse_buckling` (SciPy ARPACK matrix-free shift-invert — low memory,
        leak-free; e.g. ``num_matvecs=40``); ``"dense"`` uses :func:`_dense_buckling`
        (exact, ``O(nf²)`` memory); ``"matfree"`` uses :func:`_matfree_buckling` (JAX
        matfree Arnoldi — equivalent, opt-in; leaks memory in long loops);
        ``"cudss"`` uses :func:`_create_cudss_bf` (GPU shift-invert: cuDSS
        factor-once/solve-many + traced device Arnoldi — requires spineax +
        CUDA; kwargs e.g. ``num_matvecs=40``, ``matrix_type="symmetric"``).

    Returns
    -------
    bf : Callable[[CSRMatrix, CSRMatrix], Tuple[jax.Array, jax.Array]]
        ``bf(K, Kg) -> (lambdas, modes)``. ``lambdas`` is the ``(num_modes,)`` ascending
        positive buckling factors, **differentiable** w.r.t. ``K`` and ``Kg``. ``modes``
        is the ``(num_modes, N)`` full-DoF mode shapes for visualization and carries
        **no gradient** (``stop_gradient``; eigenvector derivatives are unstable at the
        degenerate buckling pairs).

    Notes
    -----
    ``bf`` is jittable (``jax.jit(bf)`` / ``jax.jit(jax.grad(...))``): the forward
    eigensolve runs host-side in a :func:`jax.pure_callback` and only ``K.data`` /
    ``K_g.data`` are traced. The (constant) sparsity ``indices`` are captured on the
    first **eager** call, so call ``bf(K, Kg)`` once outside ``jit`` before jitting it
    (a clear error is raised otherwise).
    """
    if solver not in ("sparse", "dense", "matfree", "cudss"):
        raise ValueError(
            f"unknown buckling solver {solver!r}; expected 'sparse' (default, SciPy "
            "ARPACK matrix-free), 'dense', 'matfree' (opt-in JAX/matfree backend), "
            "or 'cudss' (GPU shift-invert via spineax cuDSS).")
    if solver == "cudss":
        # Traced-JAX forward (cuDSS factor + device Arnoldi) with its own
        # custom_vjp wiring — GPU work must NOT run inside this driver's host
        # eigensolve callback (device calls from a GPU host callback deadlock
        # the stream).
        return _create_cudss_bf(free_dofs, num_modes, **solver_kw)
    free = onp.asarray(free_dofs)
    nf = len(free)
    m = num_modes

    # The CSR sparsity (indptr / indices) is fixed for a given problem, so the host
    # callback can reuse the concrete structure captured on the first (eager) call.
    # This lets `bf` itself be jitted: under jit ``K.indptr`` / ``K.indices`` are
    # tracers, but the callback only needs ``K.data`` (the traced input) — the
    # constant structure comes from this cache.
    _idx_cache = {}                                  # name -> (indptr, indices)
    _meta = {}                                       # constant K/Kg shapes (set on 1st call)

    def _concrete_indices(name, M):
        if isinstance(M.indices, jax.core.Tracer):   # tracing (e.g. under jax.jit)
            if name not in _idx_cache:
                raise ValueError(
                    "create_linear_buckling_solver: call bf(K, Kg) once eagerly before "
                    "jitting so the (constant) CSR structure can be captured for the "
                    "host eigensolve callback.")
            return _idx_cache[name]
        _idx_cache[name] = (onp.asarray(M.indptr), onp.asarray(M.indices))
        return _idx_cache[name]

    def _solve_host(K, Kg):                          # concrete CSRMatrix -> (λ, modes, φ_free)
        N = K.shape[0]
        if solver == "dense":
            lam, modes = _dense_buckling(K, Kg, free, num_modes=m)
        elif solver == "matfree":                    # opt-in JAX/matfree backend
            lam, modes = _matfree_buckling(K, Kg, free, num_modes=m, **solver_kw)
        else:                                        # "sparse" -> SciPy ARPACK (default)
            lam, modes = _sparse_buckling(K, Kg, free, num_modes=m, **solver_kw)
        modes = onp.asarray(modes)                   # (m', N)
        phi = modes[:, free]                         # (m', nf)
        Kf_sp = _restrict_csr(K, free)               # SciPy (no JAX in the host callback)
        Kphi = (Kf_sp @ phi.T).T                     # (m', nf)
        scale = 1.0 / onp.sqrt(onp.abs(onp.sum(phi * Kphi, axis=1)))  # φᵀ K φ = 1
        phi = phi * scale[:, None]
        modes = modes * scale[:, None]
        out_m = lam.shape[0]                          # pad if fewer positive modes found
        if out_m < m:
            lam = onp.concatenate([lam, onp.full(m - out_m, onp.inf)])
            phi = onp.concatenate([phi, onp.zeros((m - out_m, nf))], axis=0)
            modes = onp.concatenate([modes, onp.zeros((m - out_m, N))], axis=0)
        return (onp.asarray(lam[:m], onp.float64),
                onp.asarray(modes[:m], onp.float64),
                onp.asarray(phi[:m], onp.float64))

    # Persistent host callback: ONE function object reused for every call. Passing a
    # fresh ``lambda`` to ``jax.pure_callback`` on each call registers a new host callable
    # with the runtime that is never released — it accumulates (with whatever its host
    # execution allocates, e.g. the sparse factor) and leaks memory across a long loop
    # (topology optimization). Reading the constant indices/shapes from the caches keeps
    # this callable identical across calls.
    def _host_eval(kd, kgd):
        kip, kix = _idx_cache["K"]
        gip, gix = _idx_cache["Kg"]
        return _solve_host(
            CSRMatrix(onp.asarray(kd), kip, kix, _meta["K"]),
            CSRMatrix(onp.asarray(kgd), gip, gix, _meta["Kg"]))

    def _callback(K, Kg):
        N = K.shape[0]
        _concrete_indices("K", K)                    # populate index cache (jittable)
        _concrete_indices("Kg", Kg)
        _meta["K"] = K.shape; _meta["Kg"] = Kg.shape
        return jax.pure_callback(
            _host_eval,                              # same object every call -> no leak
            (jax.ShapeDtypeStruct((m,), jnp.float64),
             jax.ShapeDtypeStruct((m, N), jnp.float64),
             jax.ShapeDtypeStruct((m, nf), jnp.float64)),
            K.data, Kg.data)

    @jax.custom_vjp
    def bf(K, Kg):
        """Return ``(lambdas, modes)``: ``lambdas`` is differentiable w.r.t. K, Kg;
        ``modes`` (full-DoF mode shapes) carries no gradient (stop_gradient)."""
        lam, modes, _ = _callback(K, Kg)
        return lam, jax.lax.stop_gradient(modes)

    def _fwd(K, Kg):
        lam, modes, phi = _callback(K, Kg)
        return (lam, jax.lax.stop_gradient(modes)), (K, Kg, lam, phi)

    def _bwd(res, cts):
        K, Kg, lam, phi = res
        lam_bar, _modes_bar = cts                     # mode-shape cotangent ignored
        K_cot = _outer_cotangent(K, phi, lam_bar * lam, free)
        Kg_cot = _outer_cotangent(Kg, phi, lam_bar * lam * lam, free)
        return (K_cot, Kg_cot)

    bf.defvjp(_fwd, _bwd)
    return bf


__all__ = ["generalized_eigh", "create_linear_buckling_solver",
           "BucklingConvergenceError"]
