"""Preconditioners for the Krylov solvers: one callable, one contract.

A preconditioner is a function ``preconditioner(op) -> M`` that is called
*inside* the solve, with the operator of that solve, and returns the
preconditioner action ``M(r) -> z`` (an approximate inverse) for the Krylov
method. ``op`` is a :class:`LinearOperator`: its matvec is always available,
and its assembled CSR values / diagonal are assembled on demand, so a
preconditioner can be built from the current operator even on the matrix-free
Krylov path. Because it runs under tracing, anything built from ``op`` (a
Jacobi diagonal, a factorization of a batch mean, ...) is rebuilt for every
solve and never goes stale.

This single entry point is passed as ``preconditioner=`` to
:func:`feax.create_solver` (and may be overridden per call,
``solver(tp, u0, preconditioner=P)``); it replaces the former option fields and
the preconditioner state that used to ride along ``TracedStructure``.

Built-in preconditioners
------------------------
* :func:`jacobi` — diagonal scaling from the current operator.
* :func:`amg` — a smoothed-aggregation AMG hierarchy built once on the host
  (PyAMG, from a sample operator) and applied as one multigrid cycle. The
  result, :class:`AMGPreconditioner`, is a pytree, so it can be a traced
  argument of a jitted solve instead of a closure constant.
* a fixed ``M``: ``lambda op: M``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as onp

from ..csr import CSRMatrix, CSRTranspose


# ---------------------------------------------------------------------------
# The operator handed to a preconditioner
# ---------------------------------------------------------------------------

class LinearOperator:
    """The operator of one linear solve, as a preconditioner sees it.

    Attributes
    ----------
    matvec : callable
        ``A @ v`` (also available as ``op @ v`` / ``op(v)``).
    csr : CSRMatrix
        The assembled, BC-applied operator. Assembled on first access when the
        solve is matrix-free (one assembly; the matvec stays matrix-free).
    values, indptr, indices : arrays
        The CSR triple of ``csr``.
    diag : array
        The diagonal of ``csr`` (gathered through the problem's diagonal slot
        map when available).
    structure, problem, bc :
        The :class:`feax.TracedStructure` (or None), the Problem and the
        DirichletBC of the solve, for preconditioners that need them.
    transpose : bool
        True for the adjoint operator ``Jᵀ``. ``csr`` then still holds ``J``
        (its diagonal is the same; AMG and Jacobi do not need the transpose).
    """

    def __init__(self, matvec=None, csr=None, assemble=None, structure=None,
                 problem=None, bc=None, transpose=False):
        if matvec is None and csr is None:
            raise ValueError("LinearOperator needs a matvec or an assembled CSR matrix.")
        self._matvec = matvec
        self._csr = csr
        self._assemble = assemble
        self.structure = structure
        self.problem = problem
        self.bc = bc
        self.transpose = transpose

    @classmethod
    def wrap(cls, A, **kw):
        """Wrap whatever a solve function receives (CSRMatrix, its transpose,
        a matvec callable, a sparse/dense matrix) into a LinearOperator."""
        if isinstance(A, cls):
            return A
        if isinstance(A, CSRTranspose):
            return cls(matvec=lambda v: A @ v, csr=A.parent, transpose=True, **kw)
        if isinstance(A, CSRMatrix):
            return cls(csr=A, **kw)
        if callable(A):
            return cls(matvec=A, **kw)
        return cls(matvec=lambda v: A @ v, **kw)

    # -- action --------------------------------------------------------------
    def matvec(self, v):
        if self._matvec is not None:
            return self._matvec(v)
        return self._csr @ v

    def __matmul__(self, v):
        return self.matvec(v)

    def __call__(self, v):
        return self.matvec(v)

    # -- assembled form ------------------------------------------------------
    @property
    def has_csr(self) -> bool:
        return self._csr is not None

    @property
    def csr(self) -> CSRMatrix:
        if self._csr is None:
            if self._assemble is None:
                raise ValueError(
                    "This operator is matrix-free and cannot be assembled here, so a "
                    "preconditioner that needs its values (e.g. jacobi) is unavailable "
                    "on this path. Use a fixed preconditioner (lambda op: M) or a "
                    "direct solver.")
            self._csr = self._assemble()
        return self._csr

    @property
    def values(self):
        return self.csr.data

    @property
    def indptr(self):
        return self.csr.indptr

    @property
    def indices(self):
        return self.csr.indices

    @property
    def shape(self):
        return self.csr.shape

    @property
    def diag(self):
        src = self.structure if self.structure is not None else self.problem
        slots = getattr(src, "csr_diag_slots", None) if src is not None else None
        if slots is not None and self.has_csr_pattern_of(src):
            return self.values[slots]
        csr = self.csr
        n = csr.shape[0]
        nnz = csr.indices.shape[0]
        rows = jnp.repeat(jnp.arange(n), jnp.diff(csr.indptr), total_repeat_length=nnz)
        on_diag = rows == csr.indices
        return jnp.zeros(n, csr.data.dtype).at[jnp.where(on_diag, rows, n)].add(
            jnp.where(on_diag, csr.data, 0.0))

    def has_csr_pattern_of(self, src) -> bool:
        """Whether ``csr`` is on the problem's own CSR pattern (so the slot
        maps of ``src`` apply) rather than e.g. a merged or reduced pattern."""
        indptr = getattr(src, "csr_indptr", None)
        if indptr is None:
            return False
        return self.csr.indptr.shape == jnp.shape(indptr) and \
            self.csr.indices.shape == jnp.shape(getattr(src, "csr_indices"))


# ---------------------------------------------------------------------------
# Passing a preconditioner through custom_vjp / jit as an argument
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class _StaticPreconditioner:
    """A plain callable as a leafless pytree, so it can be an argument of the
    custom_vjp solve (it is kept static; a pytree preconditioner such as
    :class:`AMGPreconditioner` passes its arrays as traced leaves instead)."""

    def __init__(self, fn):
        self.fn = fn

    def tree_flatten(self):
        return (), (self.fn,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux[0])

    def __call__(self, op):
        return self.fn(op)

    def __repr__(self):
        return f"_StaticPreconditioner({self.fn!r})"


def as_pytree(preconditioner):
    """Return ``preconditioner`` in a form whose pytree leaves are all arrays
    (``None`` passes through)."""
    if preconditioner is None:
        return None
    leaves = jax.tree_util.tree_leaves(preconditioner)
    if all(isinstance(l, (jax.Array, onp.ndarray, onp.generic, int, float, bool))
           for l in leaves):
        return preconditioner
    if not callable(preconditioner):
        raise TypeError(
            f"preconditioner must be callable (preconditioner(op) -> M), got "
            f"{type(preconditioner).__name__}.")
    return _StaticPreconditioner(preconditioner)


# ---------------------------------------------------------------------------
# Jacobi
# ---------------------------------------------------------------------------

def jacobi(shift: float = 1e-12) -> Callable:
    """Diagonal (Jacobi) preconditioner, ``M = diag(A)⁻¹``, from the current operator.

    The diagonal is read from the assembled operator of *each* solve (one
    assembly on the matrix-free Krylov path), so it follows the parameters —
    e.g. the SIMP stiffness contrast in topology optimization. ``shift``
    regularizes near-zero diagonal entries.
    """
    def preconditioner(op: LinearOperator):
        inv = 1.0 / (op.diag + shift)
        return lambda r: inv * r

    preconditioner.__name__ = "jacobi"
    return preconditioner


# ---------------------------------------------------------------------------
# AMG
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AMGOptions:
    """Configuration of the smoothed-aggregation AMG preconditioner (:func:`amg`).

    Parameters
    ----------
    near_nullspace : array, str, or None, default None
        The AMG near-null-space ``B`` — the low-energy modes the coarse grid must
        represent. Accepts three kinds of input:

        - **array** ``(n_dof, k)``: a user-defined near-null-space, used verbatim.
        - **str preset** of a known physics:
            - ``"rigid_body"``: rigid body modes built from the mesh node
              coordinates (6 in 3D, 3 in 2D); the right choice for continuum
              elasticity (single ``vec == dim`` field).
            - ``"constant"``: the constant near-null-space (PyAMG default; correct
              for scalar elliptic problems like Poisson/heat).
            - ``"adaptive_sa"``: *estimate* the near-null-space numerically via
              adaptive smoothed aggregation (PyAMG ``adaptive_sa_solver``).
              ``num_nullspace`` sets how many candidates.
        - **None** (default): rigid body modes for a single ``vec == dim`` field
          (elasticity), else the constant near-null-space.
    num_nullspace : int, optional
        Number of candidate vectors for ``near_nullspace="adaptive_sa"``.
    cycle : str, default "V"
        Multigrid cycle type ("V", "W", "F").
    smoother_omega : float, default 0.67
        Damped-Jacobi relaxation factor (undamped Jacobi diverges on elasticity).
    smoother_sweeps : int, default 2
        Jacobi sweeps per pre/post smoothing.
    coarse_solver : str, default "pinv"
        AMJax coarse-grid solver ("pinv", "lu", "qr", "jacobi").
    strength : float, optional
        SA strength-of-connection threshold (PyAMG default when None).
    rebuild_every : int or None, default None
        Newton-only: how often to rebuild the hierarchy from the *current*
        tangent during a nonlinear solve (the near-null-space is always reused).

        - ``None`` (default): **adaptive lag** — reuse the hierarchy across Newton
          steps and rebuild it only when it has gone stale, detected by the linear
          solve failing to reach ``lag_tol`` (then rebuild + re-solve that step).
          Host Newton loop (eager, not jit/vmap).
        - ``0``: fixed hierarchy (the one built by :func:`amg`) for every Newton
          step. Cheapest and fully traced/jit/vmap/grad-able.
        - ``k >= 1``: host Newton loop, rebuild every ``k`` iterations.

        On the host loop the adjoint rebuilds its hierarchy from the converged
        tangent. Linear solves always use the fixed hierarchy.
    lag_tol : float, default 1e-3
        Adaptive-lag trigger: rebuild when the per-step linear solve's relative
        residual exceeds this.
    """
    near_nullspace: Optional[object] = None
    num_nullspace: Optional[int] = None
    cycle: str = "V"
    smoother_omega: float = 0.67
    smoother_sweeps: int = 2
    coarse_solver: str = "pinv"
    strength: Optional[float] = None
    rebuild_every: Optional[int] = None
    lag_tol: float = 1e-3

    def __post_init__(self):
        if isinstance(self.near_nullspace, str):
            valid_ns = ("rigid_body", "constant", "adaptive_sa")
            if self.near_nullspace.lower() not in valid_ns:
                raise ValueError(
                    f"Invalid near_nullspace preset: {self.near_nullspace!r}. "
                    f"Choose from {valid_ns}, or pass an (n_dof, k) array."
                )


@jax.tree_util.register_pytree_node_class
class AMGPreconditioner:
    """A built AMG hierarchy as a preconditioner (``preconditioner(op) -> M``).

    Created by :func:`amg`. A pytree whose leaves are the hierarchy's arrays, so
    a jitted caller can pass it as an argument of the solve
    (``solver(tp, u0, preconditioner=P)``) instead of baking every level in as a
    constant. ``rebuild(A)`` builds a new hierarchy from a concrete operator on
    the host (used by the Newton host loop; not traceable).
    """

    def __init__(self, hierarchy, options: AMGOptions, problem, bc, projection=None):
        self.hierarchy = hierarchy
        self.options = options
        self.problem = problem
        self.bc = bc
        self.projection = projection

    # pytree: the hierarchy's arrays are the leaves, everything else is static
    def tree_flatten(self):
        return (self.hierarchy,), (self.options, self.problem, self.bc, self.projection)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], *aux)

    def __call__(self, op=None):
        return self.hierarchy.aspreconditioner(cycle=self.options.cycle)

    @property
    def rebuild_every(self):
        return self.options.rebuild_every

    @property
    def lag_tol(self):
        return self.options.lag_tol

    def rebuild(self, A) -> "AMGPreconditioner":
        """A new preconditioner whose hierarchy is built from the concrete
        (BC-applied) CSR operator ``A`` — host-side, not traceable."""
        from .amg import build_amg_hierarchy
        if isinstance(A, LinearOperator):
            A = A.csr
        return AMGPreconditioner(
            build_amg_hierarchy(self.problem, self.bc, A, self.options),
            self.options, self.problem, self.bc, self.projection)

    def __repr__(self):
        return f"AMGPreconditioner(levels={len(self.hierarchy.levels)}, options={self.options})"


def amg(problem, bc, traced_params, options: Optional[AMGOptions] = None, *,
        traced_structure=None, symmetric_elimination: bool = True,
        projection=None) -> AMGPreconditioner:
    """Build a smoothed-aggregation AMG preconditioner from a sample operator.

    Assembles one BC-applied Jacobian at ``traced_params`` (on the host, via
    PyAMG — requires the optional ``feax[amg]`` extra) and converts the
    hierarchy to a JAX-native ``amjax.MultilevelSolver``. The returned
    :class:`AMGPreconditioner` applies one multigrid cycle as ``M``; the outer
    Krylov applies the current operator matrix-free, so the fixed hierarchy
    stays usable as the operator changes between solves (parameter sweeps,
    Newton steps — see ``AMGOptions.rebuild_every`` for rebuilding it).

    Parameters
    ----------
    problem, bc :
        The feax Problem and its DirichletBC.
    traced_params : TracedParams
        Sample parameters the hierarchy is built at.
    options : AMGOptions, optional
        Hierarchy / smoother / near-null-space configuration.
    traced_structure : TracedStructure, optional
        Assemble the sample on the TracedStructure path (recommended when the
        solver uses one).
    symmetric_elimination : bool, default True
        Must match the solver's Dirichlet elimination.
    projection : sparse matrix, optional
        The reduced solver's projection (``create_solver(projection=...)``):
        the sample is then the reduced operator ``PᵀJP`` and the near-null-space
        is mapped into the reduced space.

    Examples
    --------
    >>> P = fe.amg(problem, bc, tp, fe.AMGOptions(near_nullspace="rigid_body"))
    >>> solver = fe.create_solver(problem, bc, fe.KrylovSolverOptions(solver="cg"),
    ...                           preconditioner=P, linear=True,
    ...                           traced_params=tp, traced_structure=ts)
    >>> u = jax.jit(lambda tp, ts, P: solver(tp, u0, traced_structure=ts,
    ...                                      preconditioner=P))(tp, ts, P)
    """
    from ..utils import zero_like_initial_guess
    from .amg import build_amg_hierarchy

    if options is None:
        options = AMGOptions()
    if not isinstance(options, AMGOptions):
        raise TypeError(f"options must be AMGOptions, got {type(options).__name__}.")
    if traced_params is None:
        raise ValueError("amg() needs sample traced_params to assemble the operator "
                         "the hierarchy is built from.")

    initial = zero_like_initial_guess(problem, bc)
    if projection is None:
        from ..assembler import create_J_bc_csr_parametric
        J_csr = create_J_bc_csr_parametric(problem, symmetric=symmetric_elimination)
        sample = J_csr(initial, traced_params, bc, traced_structure)
        build_options = options
    else:
        from ..asd import connectivity_pattern, operator_assembler, reduced_operator_pattern
        from ..assembler import create_matfree_res_J_parametric
        from .reduced import _reduced_amg_options
        P = projection
        matfree = create_matfree_res_J_parametric(problem, symmetric=True)
        _, J_mv = matfree(initial, traced_params, bc, traced_structure)
        assemble = operator_assembler(reduced_operator_pattern(P, connectivity_pattern(problem)))
        sample = assemble(lambda v: P.T @ J_mv(P @ v))
        build_options = _reduced_amg_options(options, problem, bc, P, int(P.shape[1]))

    hierarchy = build_amg_hierarchy(problem, bc, sample, build_options)
    return AMGPreconditioner(hierarchy, build_options, problem, bc, projection)


__all__ = [
    "AMGOptions",
    "AMGPreconditioner",
    "LinearOperator",
    "amg",
    "jacobi",
    "as_pytree",
]
