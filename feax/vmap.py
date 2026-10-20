"""``feax.vmap``: batch a solver over its parameters, solving the batch well.

``jax.vmap(solver)`` already batches a feax solver, but it does not know what
the batch means: when the batched parameters change only the right-hand side
(load cases), a direct solver factorizes once and solves many; when they
change the stiffness matrix (designs), it factorizes every case — ``N``
factorizations held at once, which runs out of memory quickly.

:func:`vmap` looks at which parameters are batched, decides which of the two
it is, and picks the solve accordingly:

* **rhs batch** (the operator does not depend on the batched parameters):
  plain ``jax.vmap`` of the solver — with a direct solver that is one
  factorization and a multi-right-hand-side solve.
* **lhs batch** (one operator per case): the ``N`` operators are assembled on
  their shared sparsity pattern and solved together by
  :func:`feax.solvers.batched.batched_pcg`, preconditioned by a batch-level
  preconditioner built from all of them — by default one shared factorization
  of their mean (:func:`feax.shared_factor`), or :func:`feax.jacobi` without
  cuDSS. Each case reports its CG iteration count.

A Newton solver is batched the same way: all cases iterate together in one
traced loop, with one shared factorization of the mean tangent per Newton step
and a per-case line search (a nonlinear batch is always an lhs batch).

The returned callable is differentiable (custom VJP: the adjoint batch is
solved the same way) and composes with ``jax.jit``.
"""

import warnings
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as onp

from .solver import Solver
from .solvers.batched import BatchedNewtonInfo, BatchedSolveInfo, batched_pcg, batched_spmv
from .solvers.common import _safe_negate
from .solvers.options import DirectSolverOptions, KrylovSolverOptions, has_cudss
from .solvers.preconditioners import AMGPreconditioner, LinearOperator, jacobi, shared_factor

__all__ = ["BatchedSolver", "vmap"]


# ---------------------------------------------------------------------------
# in_axes helpers (the batch is over traced_params only)
# ---------------------------------------------------------------------------

def _is_axis_leaf(x):
    return x is None or isinstance(x, int)


def _axes_tree(in_axes, tp):
    """Broadcast an ``in_axes`` prefix (int / None / a TracedParams-shaped
    prefix of ints and None) to every leaf of ``tp``."""
    if _is_axis_leaf(in_axes):
        return jax.tree_util.tree_map(lambda _: in_axes, tp)
    return jax.tree_util.tree_map(
        lambda ax, sub: jax.tree_util.tree_map(lambda _: ax, sub),
        in_axes, tp, is_leaf=_is_axis_leaf)


def _axis_leaves(axes_tree):
    return jax.tree_util.tree_leaves(axes_tree, is_leaf=_is_axis_leaf)


def _to_front(tp, axes_tree):
    """Move every batched leaf's batch axis to the front; return the
    normalized tree, the 0/None axes tree and the batch size."""
    leaves, treedef = jax.tree_util.tree_flatten(tp)
    axes = _axis_leaves(axes_tree)
    if len(axes) != len(leaves):
        raise ValueError("in_axes does not match the structure of traced_params.")
    moved, N = [], None
    for leaf, ax in zip(leaves, axes):
        if ax is None:
            moved.append(leaf)
            continue
        leaf = jnp.moveaxis(jnp.asarray(leaf), ax, 0)
        if N is None:
            N = leaf.shape[0]
        elif leaf.shape[0] != N:
            raise ValueError(f"batched leaves have different batch sizes ({N} and {leaf.shape[0]}).")
        moved.append(leaf)
    if N is None:
        raise ValueError("in_axes batches nothing: at least one leaf of traced_params must have an axis.")
    front = jax.tree_util.tree_unflatten(treedef, [0 if ax is not None else None for ax in axes])
    return jax.tree_util.tree_unflatten(treedef, moved), front, N


# ---------------------------------------------------------------------------
# rhs / lhs detection
# ---------------------------------------------------------------------------

def _operator_depends_on(solver: Solver, axes_tree, u0) -> bool:
    """Whether the operator changes with the batched leaves of traced_params:
    a JVP of the assembled CSR values at the solver's sample parameters, in a
    random direction of those leaves."""
    tp = solver.sample_params
    leaves, treedef = jax.tree_util.tree_flatten(tp)
    axes = _axis_leaves(axes_tree)
    key = jax.random.PRNGKey(0)
    tangents = []
    for i, (leaf, ax) in enumerate(zip(leaves, axes)):
        leaf = jnp.asarray(leaf)
        if ax is None or not jnp.issubdtype(leaf.dtype, jnp.floating):
            tangents.append(jnp.zeros_like(leaf))
        else:
            tangents.append(jax.random.normal(jax.random.fold_in(key, i), leaf.shape, leaf.dtype))

    def values(leaves_):
        return solver.assemble(u0, jax.tree_util.tree_unflatten(treedef, leaves_)).data

    v, dv = jax.jvp(values, (leaves,), (tangents,))
    scale = float(jnp.max(jnp.abs(v))) or 1.0
    return float(jnp.max(jnp.abs(dv))) > 1e-12 * scale


class _CsrPattern:
    """The CSR pattern shared by the batch's operators.

    Read from the TracedStructure of the call when there is one: its arrays are
    traced leaves, so nothing nnz-sized is baked into a jitted caller as a
    constant (at 500k DOF the pattern alone is ~10^8 bytes, and XLA folds such
    constants on one thread). Without a TracedStructure it comes from one
    concrete assembly at the solver's sample parameters, made here (eagerly:
    under a trace it would be a tracer) -- the call falls back to the solver's
    own TracedStructure, so a call can only miss one when the solver has none.
    """

    def __init__(self, solver, u0, sample):
        self._fallback = None
        if solver.traced_structure is None:
            K0 = solver.assemble(u0, sample)
            self._fallback = (K0.indptr, K0.indices)

    def __call__(self, ts):
        if ts is not None:
            return ts.csr_indptr, ts.csr_indices
        return self._fallback


# ---------------------------------------------------------------------------
# the batched solver
# ---------------------------------------------------------------------------

class BatchedSolver:
    """What :func:`vmap` returns. ``kind`` is ``"rhs"`` or ``"lhs"``."""

    def __init__(self, fn, *, solver, kind, in_axes, options, preconditioner, chunk, return_info,
                 share=None, factorize_fn=None):
        self._fn = fn
        self._factorize_fn = factorize_fn
        self._factorize_jit = None
        self.solver = solver
        self.kind = kind
        self.in_axes = in_axes
        self.options = options
        self.preconditioner = preconditioner
        self.chunk = chunk
        self.share = share
        self.return_info = return_info
        self._signature = None

    def __call__(self, traced_params, initial_guess=None, traced_structure=None, factors=None):
        """Solve the batch. ``factors`` (from :meth:`factorize`) are the
        group factorizations to precondition with -- no mean is assembled and
        nothing is factorized; without them each call factorizes its own."""
        from .solution import Solution
        if isinstance(initial_guess, Solution):
            initial_guess = initial_guess.dofs
        if factors is None:
            U, info = self._fn(traced_params, initial_guess, traced_structure)
        elif self._factorize_fn is None:
            raise ValueError("factors= applies to an lhs CG batch with fe.shared_factor().")
        else:
            U, info = self._fn(traced_params, initial_guess, traced_structure, factors)
        if self.solver.return_solution:
            U = Solution(U, self.solver.layout)
        return (U, info) if self.return_info else U

    def factorize(self, traced_params, initial_guess=None, traced_structure=None):
        """Factorize the mean operator of every group of ``share`` cases (one
        cuDSS entry each, kept alive) and return them as ``factors`` for
        ``batched(..., factors=factors)``.

        The preconditioner only has to be close, not current: reuse the
        factors over several calls -- e.g. the iterations of an optimization
        -- and factorize again (at the current parameters) when the CG counts
        have grown too much. The batch must keep its order and ``share``
        while the factors are reused. spineax keeps ``SPINEAX_FACTOR_CACHE``
        entries (read at each new entry): set it to at least the number of
        groups, or an evicted group is re-factorized from its stored mean on
        every use (correct, but no reuse)."""
        if self._factorize_fn is None:
            raise ValueError("factorize applies to an lhs CG batch with fe.shared_factor().")
        if self._factorize_jit is None:
            self._factorize_jit = jax.jit(self._factorize_fn)
        from .utils import zero_like_initial_guess
        u0 = (zero_like_initial_guess(self.solver.problem, self.solver.bc)
              if initial_guess is None else initial_guess)
        ts = self.solver.traced_structure if traced_structure is None else traced_structure
        return self._factorize_jit(traced_params, u0, ts)

    def operator_signature(self, traced_params, initial_guess=None, traced_structure=None):
        """A signature of every case's operator, ``(B, n_dof)``: the log of the
        diagonal of its (BC-applied) stiffness matrix at the initial guess.

        Independent of how the design is parametrized (density, modulus,
        orientation, ...): whatever enters the operator shows up here. Feed it
        to :func:`feax.gene.similarity_order` to order the batch so that
        ``chunk`` groups similar operators for the shared preconditioner."""
        from .utils import zero_like_initial_guess
        u0 = zero_like_initial_guess(self.solver.problem, self.solver.bc) if initial_guess is None else initial_guess
        ts = self.solver.traced_structure if traced_structure is None else traced_structure
        tp, front, _ = _to_front(traced_params, _axes_tree(self.in_axes, self.solver.sample_params))
        if self._signature is None:
            # jitted once, with the mesh-sized arrays as arguments; assembled
            # ``chunk`` operators at a time (all N at once would hold N
            # operators: 29 GiB for 100 cases at 500k DOF)
            def signature(tp, u0, ts):
                slots = (ts if ts is not None else self.solver.problem).csr_diag_slots

                def one(t):
                    diag = self.solver.assemble(u0, t, traced_structure=ts).data[slots]
                    return jnp.log(jnp.maximum(diag, 1e-300))
                group = lambda c: jax.vmap(one, in_axes=(front,))(c)
                n_cases = jax.tree_util.tree_leaves(jax.tree_util.tree_map(
                    lambda leaf, ax: None if ax is None else leaf.shape[0], tp, front,
                    is_leaf=_is_axis_leaf))[0]
                if self.chunk is None or self.chunk >= n_cases:
                    return group(tp)
                return _chunked(group, tp, front, n_cases, self.chunk)
            self._signature = jax.jit(signature)
        return self._signature(tp, u0, ts)

    def __repr__(self):
        share = "" if self.share is None else f", share={self.share}"
        return (f"BatchedSolver({self.kind}, {self.solver!r}, chunk={self.chunk}{share}, "
                f"preconditioner={getattr(self.preconditioner, '__name__', self.preconditioner)!r})")


def vmap(solver: Solver, in_axes: Any = 0, *, batch: str = "auto",
         options: Optional[Any] = None, preconditioner: Optional[Callable] = None,
         chunk: Optional[int] = None, share: Optional[int] = None,
         return_info: bool = False, self_adjoint: bool = False) -> BatchedSolver:
    """Batch a feax solver over ``traced_params``, solving the batch well.

    Parameters
    ----------
    solver : Solver
        A solver from :func:`feax.create_solver`: linear (``linear=True``) or
        the standard traced Newton solver (no ``extra_residual_fn``, no
        host-side AMG rebuild). A Newton batch is always an lhs batch (the
        tangent depends on the state): all cases iterate together, with one
        shared factorization of the mean tangent per Newton step and a per-case
        line search; ``info`` is then a :class:`BatchedNewtonInfo` (Newton
        iterations, ``converged``, ``line_search_failed``, CG iterations) and
        every case's final iterate is returned, converged or not.
    in_axes : int, None or a TracedParams-shaped prefix
        Which leaves of ``traced_params`` carry the batch axis (as for
        ``jax.vmap``; the initial guess and the TracedStructure are shared).
    batch : {"auto", "rhs", "lhs"}
        What the batch changes: ``"rhs"`` only the right-hand side (same
        operator for every case), ``"lhs"`` the operator too. ``"auto"``
        detects it from the solver's sample parameters.
    options : KrylovSolverOptions or DirectSolverOptions, optional
        How an *lhs* batch is solved. Default: batched CG (``tol``, ``atol``,
        ``maxiter`` from the solver's own Krylov options, or cg 1e-8 / 1e-10 /
        1000). ``DirectSolverOptions`` solves each case with its own
        factorization (``N`` at once unless ``chunk`` is set — a warning).
    preconditioner : callable, optional
        The batch-level preconditioner of an lhs batch, ``preconditioner(op) -> M``
        where ``op.values`` is ``(N, nnz)`` (every case's operator on the
        shared pattern) and ``M`` acts on ``(N, n)`` arrays. Default:
        :func:`feax.shared_factor` (one factorization of the mean operator)
        with cuDSS, else :func:`feax.jacobi`. Any callable of the same
        contract (e.g. a per-group factorization) can be used. A solver built
        with ``preconditioner=fe.jacobi()`` / ``fe.shared_factor()`` / an
        ``AMGPreconditioner`` keeps it; any other solver preconditioner is
        ignored here (it may not accept a batched operator).
    chunk : int, optional
        Process the batch in groups of this size (``lax.map`` over groups): a
        memory bound of ``chunk`` operators, and -- unless ``share`` is set --
        a preconditioner per group.
    share : int, optional
        Cases per shared preconditioner, independent of ``chunk`` (lhs linear
        batch with :func:`feax.shared_factor`). The batch is split into groups
        of ``share`` cases; each group's mean operator is accumulated ``chunk``
        operators at a time, factorized once, and that one ``M``
        preconditions the CG of all of the group's chunks. So
        ``share=N, chunk=4`` is one factorization for the whole batch with
        only 4 operators in memory. The groups re-factorize one cuDSS entry
        in turn (one symbolic analysis, one factorization's memory), and the
        adjoint runs the groups in reverse from the forward's last
        factorization (with one group, the adjoint factorizes nothing).
        Default: ``chunk`` (a preconditioner per chunk). Typical use: start
        with the whole batch while the cases are alike and lower ``share`` as
        the CG counts grow.
    self_adjoint : bool
        The objective is self-adjoint (compliance, ``c_i = f_i . u_i``): its
        adjoint right-hand side is the forward one, so ``λ_i = u_i``. The
        adjoint CG then starts from ``α_i (u_i - u0)`` with ``α_i = (u_i -
        u0)ᵀ g_i / (u_i - u0)ᵀ K_i (u_i - u0)`` (one SpMV), which is the exact
        solution and converges at iteration 0. Only a starting guess: for an
        objective that is not self-adjoint CG continues from there and the
        gradient stays exact. Linear lhs batches only.
    return_info : bool
        Also return a :class:`feax.solvers.batched.BatchedSolveInfo` with the
        per-case CG iteration counts (``rhs``: zeros).

    Returns
    -------
    BatchedSolver
        ``batched(traced_params, initial_guess=None, traced_structure=None)
        -> solution`` (``(solution, info)`` with ``return_info``), with the
        batch axis in front of the solution; differentiable w.r.t.
        ``traced_params`` and jit-able.
    """
    if not isinstance(solver, Solver):
        raise TypeError("feax.vmap takes the solver returned by feax.create_solver "
                        f"(got {type(solver).__name__}); jax.vmap batches any function.")
    if not solver.linear and not solver.traced_newton:
        raise NotImplementedError(
            "feax.vmap batches the standard traced Newton solver; this solver runs a host "
            "loop (extra_residual_fn, or an AMG preconditioner that rebuilds its hierarchy). "
            "Batch it with jax.vmap.")
    if solver.projection is not None:
        raise NotImplementedError("feax.vmap does not support reduced (projection=) solvers yet.")
    if batch not in ("auto", "rhs", "lhs"):
        raise ValueError("batch must be 'auto', 'rhs' or 'lhs'.")
    if chunk is not None and int(chunk) < 1:
        raise ValueError("chunk must be a positive integer.")
    if share is not None and int(share) < 1:
        raise ValueError("share must be a positive integer.")

    from .utils import zero_like_initial_guess
    u0_default = zero_like_initial_guess(solver.problem, solver.bc)
    problem, bc, ts0 = solver.problem, solver.bc, solver.traced_structure

    sample = solver.sample_params
    if sample is None:
        raise ValueError("feax.vmap needs the solver's sample traced_params "
                         "(create_solver(..., traced_params=tp)) to see the parameter structure.")
    axes_tree = _axes_tree(in_axes, sample)

    kind = batch
    if kind == "auto":
        # A nonlinear tangent depends on the state, which depends on every
        # batched parameter: a Newton batch always changes the operator.
        kind = "lhs" if (not solver.linear or
                         _operator_depends_on(solver, axes_tree, u0_default)) else "rhs"

    if self_adjoint and (kind == "rhs" or isinstance(options, DirectSolverOptions)
                         or not solver.linear):
        raise NotImplementedError("self_adjoint applies to the linear lhs batch solved by CG.")
    if share is not None and (kind == "rhs" or isinstance(options, DirectSolverOptions)
                              or not solver.linear):
        raise NotImplementedError(
            "share= applies to the linear lhs batch solved by CG (one preconditioner per "
            "group of cases); this batch is "
            + ("rhs" if kind == "rhs" else "direct" if isinstance(options, DirectSolverOptions)
               else "Newton") + ".")

    # ---- rhs: the operator is shared, jax.vmap does the right thing -------
    if kind == "rhs":
        if preconditioner is not None:
            raise ValueError("preconditioner is for an lhs batch; this batch only changes the "
                             "right-hand side (pass batch='lhs' to force it).")
        if options is not None:
            raise ValueError("options are for an lhs batch; the rhs batch uses the solver's own options.")

        def solve_rhs(tp, u0, ts):
            u0 = u0_default if u0 is None else u0
            ts = ts0 if ts is None else ts
            fn = lambda tp_: solver._fn(tp_, u0, traced_structure=ts)
            tp, front, N = _to_front(tp, axes_tree)
            if chunk is None or chunk >= N:
                U = jax.vmap(fn, in_axes=(front,))(tp)
            else:
                U = _chunked(lambda c: jax.vmap(fn, in_axes=(front,))(c), tp, front, N, chunk)
            info = BatchedSolveInfo(jnp.zeros(N, jnp.int32), jnp.ones(N, bool), jnp.zeros(N))
            return U, info
        return BatchedSolver(solve_rhs, solver=solver, kind="rhs", in_axes=in_axes,
                             options=None, preconditioner=None, chunk=chunk, return_info=return_info)

    # ---- lhs: one operator per case ----------------------------------------
    if options is None:
        if isinstance(solver.options, KrylovSolverOptions):
            options = solver.options
        else:
            options = KrylovSolverOptions(solver="cg", tol=1e-8, atol=1e-10, maxiter=1000)

    if isinstance(options, DirectSolverOptions):
        if preconditioner is not None:
            raise ValueError("preconditioner has no effect with a direct solver.")
        if chunk is None:
            warnings.warn(
                "feax.vmap: this batch changes the operator and options is a direct solver, "
                "so every case is factorized and all N factorizations are held at once "
                "(an A100 ran out of memory at N = 10 for 254k DOF). Set chunk= to bound "
                "the memory, or use KrylovSolverOptions (batched CG with a shared "
                "factorization).", stacklevel=2)
        direct_solver = solver if isinstance(solver.options, DirectSolverOptions) else None
        if direct_solver is None:
            from .solver import create_solver
            direct_solver = create_solver(problem, bc, options, linear=solver.linear,
                                          newton_options=solver.newton_options,
                                          traced_params=sample, traced_structure=ts0,
                                          symmetric_elimination=solver.symmetric_elimination,
                                          return_solution=False)

        def solve_direct(tp, u0, ts):
            u0 = u0_default if u0 is None else u0
            ts = ts0 if ts is None else ts
            fn = lambda tp_: direct_solver._fn(tp_, u0, traced_structure=ts)
            tp, front, N = _to_front(tp, axes_tree)
            if chunk is None or chunk >= N:
                U = jax.vmap(fn, in_axes=(front,))(tp)
            else:
                U = _chunked(lambda c: jax.vmap(fn, in_axes=(front,))(c), tp, front, N, chunk)
            info = BatchedSolveInfo(jnp.ones(N, jnp.int32), jnp.ones(N, bool), jnp.zeros(N))
            return U, info
        return BatchedSolver(solve_direct, solver=solver, kind="lhs", in_axes=in_axes,
                             options=options, preconditioner=None, chunk=chunk, return_info=return_info)

    if not isinstance(options, KrylovSolverOptions):
        raise TypeError(f"options must be KrylovSolverOptions or DirectSolverOptions, got {type(options).__name__}.")
    if not solver.linear:
        return _newton_vmap(solver, in_axes, axes_tree, options, preconditioner, chunk,
                            return_info, u0_default, ts0)
    if options.solver not in ("auto", "cg"):
        raise NotImplementedError(
            f"the batched lhs solve is conjugate gradients (SPD operators); got solver={options.solver!r}.")
    if not solver.symmetric_elimination:
        raise NotImplementedError("the batched lhs solve needs a symmetric operator (symmetric_elimination=True).")

    if preconditioner is None:
        # The solver's own preconditioner is reused only when it is known to
        # handle a batched operator (a fixed single-solve M would not).
        own = solver.preconditioner
        if isinstance(own, AMGPreconditioner) or getattr(own, "__name__", None) in ("jacobi", "shared_factor"):
            preconditioner = own
    if preconditioner is None:
        preconditioner = shared_factor() if has_cudss() else jacobi()
    if isinstance(preconditioner, AMGPreconditioner):
        _amg = preconditioner
        preconditioner = lambda op: jax.vmap(_amg(op))
        preconditioner.__name__ = "amg"
    if not callable(preconditioner):
        raise TypeError("preconditioner must be callable, preconditioner(op) -> M.")
    # A mean-based preconditioner with token access (fe.shared_factor) takes
    # the chained path: per group of ``share`` cases the mean operator is
    # accumulated ``chunk`` operators at a time and factorized, re-factorizing
    # ONE cuDSS entry from group to group (one symbolic analysis per
    # evaluation), and the adjoint starts from the forward's last
    # factorization. Any other preconditioner is built per group from all of
    # the group's operators at once.
    chained = callable(getattr(preconditioner, "factor", None))
    if not chained and share is not None and (chunk is None or int(chunk) < int(share)):
        raise ValueError(
            f"share={share} > chunk={chunk} needs a preconditioner built from the mean operator "
            f"(fe.shared_factor()); {getattr(preconditioner, '__name__', preconditioner)!r} "
            "needs every operator of its group at once (set chunk >= share).")

    pattern = _CsrPattern(solver, u0_default, sample)
    tol, atol, maxiter = options.tol, options.atol, options.maxiter
    # which leaves carry the batch (axis 0 once normalized): fixed by in_axes
    front = jax.tree_util.tree_map(lambda ax: None if ax is None else 0, axes_tree, is_leaf=_is_axis_leaf)

    def _operator(values, ts):
        indptr, indices = pattern(ts)
        return LinearOperator(values=values, indptr=indptr, indices=indices, structure=ts,
                              problem=problem, bc=bc, batched=True)

    def _cg(values, tp_c, u0, ts, rhs, M):
        """Batched CG of one chunk. ``rhs`` is None (forward: b = -r(u0)), the
        (C, n) adjoint right-hand sides, or ``(rhs, guess)``: start from the
        best multiple of ``guess`` (self-adjoint mode)."""
        indptr, indices = pattern(ts)
        x0 = None
        if rhs is None:
            rhs = -jax.vmap(lambda t: solver.residual(u0, t, traced_structure=ts),
                            in_axes=(front,))(tp_c)
        elif isinstance(rhs, tuple):
            rhs, guess = rhs
            x0 = _projected_guess(values, indptr, indices, guess, rhs)
        return batched_pcg(values, indptr, indices, rhs, M, tol=tol, atol=atol, maxiter=maxiter,
                           x0=x0)

    def _solve_group(tp_g, u0, ts, rhs):
        """One group: assemble its operators, build M, batched CG."""
        values = jax.vmap(lambda t: solver.assemble(u0, t, traced_structure=ts).data,
                          in_axes=(front,))(tp_g)
        return _cg(values, tp_g, u0, ts, rhs, preconditioner(_operator(values, ts)))

    def _grouped(tp, N, u0, ts, rhs=None, reverse=False):
        """The batch laid out in groups of ``share`` cases, ``chunk`` at a
        time (padded): ``(G, xs, mean_of, solve_with, scatter)``. ``xs`` =
        ``(leaves, valid, rhs)`` stacked by group; ``mean_of(leaves_g,
        valid_g)`` accumulates a group's mean operator chunk by chunk;
        ``solve_with(token, leaves_g, rhs_g)`` runs a group's CG with the
        factorization ``token``; ``scatter`` maps the per-group outputs back
        to case order."""
        S = min(int(share or chunk or N), N)
        C = min(int(chunk or S), S)
        src, dst, valid = _share_layout(N, S, C)                          # (G, K, C) each
        if reverse:
            src, dst, valid = src[::-1], dst[::-1], valid[::-1]
        leaves, treedef = jax.tree_util.tree_flatten(tp)
        axes = _axis_leaves(front)
        scanned = [jnp.asarray(leaf)[src] for leaf, ax in zip(leaves, axes) if ax is not None]
        g_rhs = None if rhs is None else jax.tree_util.tree_map(lambda a: jnp.asarray(a)[src], rhs)
        indptr, indices = pattern(ts)

        def rebuild(group_leaves):
            it = iter(group_leaves)
            return jax.tree_util.tree_unflatten(
                treedef, [next(it) if ax is not None else leaf for leaf, ax in zip(leaves, axes)])

        def values_of(tp_c):
            return jax.vmap(lambda t: solver.assemble(u0, t, traced_structure=ts).data,
                            in_axes=(front,))(tp_c)

        def mean_of(g_leaves, g_valid):
            def acc(total, a):
                c_leaves, c_valid = a
                v = values_of(rebuild(c_leaves))
                return total + jnp.sum(jnp.where(c_valid[:, None], v, 0.0), axis=0), None
            total, _ = jax.lax.scan(acc, jnp.zeros(indices.shape[0], u0.dtype), (g_leaves, g_valid))
            return total / jnp.maximum(jnp.sum(g_valid), 1)

        def solve_with(token, g_leaves, g_rhs_):
            M = preconditioner.apply(token)

            def solve_chunk(a):
                c_leaves, c_rhs = a
                tp_c = rebuild(c_leaves)
                return _cg(values_of(tp_c), tp_c, u0, ts, c_rhs, M)
            return jax.lax.map(solve_chunk, (g_leaves, g_rhs_))

        flat = jnp.asarray(dst).reshape(-1)                               # padding -> N, dropped

        def scatter(out):                                                 # leaves (G, K, C, ...)
            def one(o):
                o = o.reshape((flat.shape[0],) + o.shape[3:])
                return jnp.zeros((N,) + o.shape[1:], o.dtype).at[flat].set(o, mode="drop")
            return jax.tree_util.tree_map(one, out)
        return src.shape[0], (scanned, jnp.asarray(valid), g_rhs), mean_of, solve_with, scatter

    def _solve_chained(tp, N, u0, ts, rhs=None, token=None):
        """No factorizations given: each group's is made here, re-factorizing
        ONE cuDSS entry from group to group. Returns ``(X, info, token)``;
        ``token`` is the factorization of the last group processed. The
        forward (``rhs`` None) runs the groups in order; the adjoint (``rhs``
        given) runs them in reverse, starting with the forward's last group,
        whose factorization ``token`` still holds -- so a single-group batch
        factorizes once per forward + adjoint."""
        G, (scanned, valid, g_rhs), mean_of, solve_with, scatter = _grouped(
            tp, N, u0, ts, rhs, reverse=rhs is not None)
        indptr, indices = pattern(ts)

        def solve_group(token, g_leaves, g_rhs_):
            out = solve_with(token, g_leaves, g_rhs_)
            # The next group re-factorizes this very cuDSS entry: order that
            # after every solve of this group.
            return jax.lax.optimization_barrier((token, out))

        def body(token, a):
            g_leaves, g_valid, g_rhs_ = a
            token = preconditioner.factor(mean_of(g_leaves, g_valid), indptr, indices, token)
            return solve_group(token, g_leaves, g_rhs_)

        head = lambda t: jax.tree_util.tree_map(lambda x: x[0], t)
        tail = lambda t: jax.tree_util.tree_map(lambda x: x[1:], t)
        if token is None:
            token = preconditioner.factor(mean_of(head(scanned), valid[0]), indptr, indices)
        token, out0 = solve_group(token, head(scanned), head(g_rhs))
        if G > 1:
            token, outs = jax.lax.scan(body, token, (tail(scanned), valid[1:], tail(g_rhs)))
            out = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b]), out0, outs)
        else:
            out = jax.tree_util.tree_map(lambda a: a[None], out0)        # (G, K, C, ...)
        X, info = scatter(out)
        return X, info, token

    def _factorize(tp, N, u0, ts):
        """One factorization of the mean operator per group, stacked (one
        cuDSS entry each, all kept): the ``factors`` a later solve reuses."""
        G, (scanned, valid, _), mean_of, _, _ = _grouped(tp, N, u0, ts)
        indptr, indices = pattern(ts)
        return jax.lax.map(lambda a: preconditioner.factor(mean_of(*a), indptr, indices),
                           (scanned, valid))

    def _solve_factored(tp, N, u0, ts, factors, rhs=None):
        """Solve with given per-group ``factors``: no mean, no factorization."""
        G, (scanned, _, g_rhs), _, solve_with, scatter = _grouped(tp, N, u0, ts, rhs)
        n_factors = jax.tree_util.tree_leaves(factors)[0].shape[0]
        if n_factors != G:
            raise ValueError(f"factors holds {n_factors} group factorizations; this batch "
                             f"has {G} groups (N={N}, share={share}): call factorize again.")
        out = jax.lax.map(lambda a: solve_with(*a), (factors, scanned, g_rhs))
        return scatter(out)

    def _solve_batch(tp, N, u0, ts, rhs=None):
        group = chunk if share is None else share
        if group is None or group >= N:
            return _solve_group(tp, u0, ts, rhs)
        if rhs is None:
            return _chunked(lambda c: _solve_group(c, u0, ts, None), tp, front, N, group)
        if isinstance(rhs, tuple):                                        # chunk both together
            RG = jnp.concatenate([rhs[0][..., None], rhs[1][..., None]], axis=-1)
            return _chunked(lambda c, r: _solve_group(c, u0, ts, (r[..., 0], r[..., 1])),
                            tp, front, N, group, extra=RG)
        return _chunked(lambda c, r: _solve_group(c, u0, ts, r), tp, front, N, group, extra=rhs)

    def _batch_size(tp):
        return jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda leaf, ax: None if ax is None else leaf.shape[0],
                                   tp, front, is_leaf=_is_axis_leaf))[0]

    def _forward(tp, u0, ts, factors):
        N = _batch_size(tp)
        if factors is not None:
            X, info = _solve_factored(tp, N, u0, ts, factors)
            return X, info, factors
        if chained:
            return _solve_chained(tp, N, u0, ts)
        X, info = _solve_batch(tp, N, u0, ts)
        return X, info, None

    @jax.custom_vjp
    def solve_lhs(tp, u0, ts, factors):
        X, info, _ = _forward(tp, u0, ts, factors)
        return u0 + X, info

    def fwd(tp, u0, ts, factors):
        X, info, token = _forward(tp, u0, ts, factors)
        out = (u0 + X, info)
        return out, (tp, out[0], u0, ts, factors, token)

    def bwd(res, g):
        tp, U, u0, ts, factors, token = res
        gU, _ = g
        N = gU.shape[0]
        # adjoint: K_i λ_i = v_i (symmetric operators), solved like the forward
        # batch; self-adjoint: started from the forward solution's direction
        rhs = (gU, U - u0) if self_adjoint else gU
        if factors is not None:
            lam, _ = _solve_factored(tp, N, u0, ts, factors, rhs=rhs)
        elif chained:
            lam, _, _ = _solve_chained(tp, N, u0, ts, rhs=rhs, token=token)
        else:
            lam, _ = _solve_batch(tp, N, u0, ts, rhs=rhs)
        return _residual_cotangents(solver, U, lam, tp, ts, front, chunk), None, None, None

    solve_lhs.defvjp(fwd, bwd)

    def solve(tp, u0, ts, factors=None):
        u0 = u0_default if u0 is None else u0
        ts = ts0 if ts is None else ts
        if factors is not None and not chained:
            raise ValueError("factors= needs a preconditioner with token access (fe.shared_factor()).")
        tp, _, _ = _to_front(tp, axes_tree)
        return solve_lhs(tp, u0, ts, factors)

    def factorize(tp, u0, ts):
        u0 = u0_default if u0 is None else u0
        ts = ts0 if ts is None else ts
        tp, _, _ = _to_front(tp, axes_tree)
        return _factorize(tp, _batch_size(tp), u0, ts)

    return BatchedSolver(solve, solver=solver, kind="lhs", in_axes=in_axes, options=options,
                         preconditioner=preconditioner, chunk=chunk, share=share,
                         return_info=return_info, factorize_fn=factorize if chained else None)


def _projected_guess(values, indptr, indices, x, b):
    """``α_i x_i`` with ``α_i = x_iᵀ b_i / x_iᵀ K_i x_i``: the best multiple of
    ``x_i`` in the energy norm (Galerkin on span{x_i}); exactly ``K_i⁻¹ b_i``
    when ``b_i`` is a multiple of ``K_i x_i``."""
    Kx = batched_spmv(values, indptr, indices)(values, x)
    xKx = jnp.sum(x * Kx, axis=1)
    ok = xKx > 0
    alpha = jnp.where(ok, jnp.sum(x * b, axis=1) / jnp.where(ok, xKx, 1.0), 0.0)
    return alpha[:, None] * x


def _share_layout(N, share, chunk):
    """Index layout of ``share``-case groups processed ``chunk`` cases at a
    time: ``(src, dst, valid)``, each ``(G, K, C)`` with ``G`` groups of ``K``
    chunks of ``C = chunk``. ``src`` gathers the cases (padding repeats case
    ``N - 1``), ``valid`` marks the real ones (padding is kept out of the
    group means), ``dst`` scatters results back (padding goes to ``N``, out of
    bounds, and is dropped)."""
    G, K = -(-N // share), -(-share // chunk)
    j = onp.arange(K * chunk)
    case = onp.arange(G)[:, None] * share + j[None, :]
    valid = (j[None, :] < share) & (case < N)
    src = onp.where(valid, case, N - 1).reshape(G, K, chunk)
    dst = onp.where(valid, case, N).reshape(G, K, chunk)
    return src, dst, valid.reshape(G, K, chunk)


def _residual_cotangents(solver, U, lam, tp, ts, front, chunk):
    """``-(dr/dtp)^T λ`` of every case: the traced_params cotangents of the
    batched solve, from the solutions ``U`` and adjoints ``lam`` (both
    ``(N, n)``). Computed in groups of ``chunk`` cases like the solves: one
    case's residual VJP carries element-sized intermediates, so a vmap over
    the whole batch would hold N of them at once (17 GiB for 100 cases at
    500k DOF). Unbatched leaves get the sum over the cases."""
    def one(u_i, lam_i, tp_i):
        _, f_vjp = jax.vjp(lambda t: solver.residual(u_i, t, traced_structure=ts), tp_i)
        return jax.tree_util.tree_map(_safe_negate, f_vjp(lam_i)[0])

    def group(tp_g, U_g, lam_g):
        return jax.vmap(one, in_axes=(0, 0, front))(U_g, lam_g, tp_g)
    N = U.shape[0]
    if chunk is None or chunk >= N:
        ct = group(tp, U, lam)
    else:
        UL = jnp.concatenate([U[..., None], lam[..., None]], axis=-1)   # chunk both together
        ct = _chunked(lambda c, ul: group(c, ul[..., 0], ul[..., 1]), tp, front, N, chunk, extra=UL)
    return jax.tree_util.tree_map(
        lambda c, ax: c if ax is not None else jnp.sum(c, axis=0), ct, front, is_leaf=_is_axis_leaf)


def _chunked(group_fn, tp, front, N, chunk, extra=None):
    """``lax.map`` ``group_fn`` over groups of ``chunk`` cases of a
    front-batched tree: only the batched leaves are scanned (the last group is
    padded by repeating the last case; padded results are dropped), the
    unbatched leaves are closed over. ``extra`` is an optional ``(N, ...)``
    array chunked alongside; ``group_fn(tp_group[, extra_group])`` returns an
    array or a tuple of arrays with the group axis in front."""
    C = int(chunk)
    G = -(-N // C)
    idx = jnp.minimum(jnp.arange(G * C), N - 1)
    leaves, treedef = jax.tree_util.tree_flatten(tp)
    axes = _axis_leaves(front)

    def split(leaf):
        leaf = jnp.asarray(leaf)[idx]
        return leaf.reshape((G, C) + leaf.shape[1:])
    scanned = [split(leaf) for leaf, ax in zip(leaves, axes) if ax is not None]

    def rebuild(group_leaves):
        it = iter(group_leaves)
        return jax.tree_util.tree_unflatten(
            treedef, [next(it) if ax is not None else leaf for leaf, ax in zip(leaves, axes)])

    if extra is None:
        out = jax.lax.map(lambda g: group_fn(rebuild(g)), scanned)
    else:
        extra_groups = jnp.asarray(extra)[idx].reshape((G, C) + jnp.shape(extra)[1:])
        out = jax.lax.map(lambda a: group_fn(rebuild(a[0]), a[1]), (scanned, extra_groups))

    def merge(o):
        return o.reshape((G * C,) + o.shape[2:])[:N]
    return jax.tree_util.tree_map(merge, out)


# ---------------------------------------------------------------------------
# Newton: a batched Newton loop with one shared factorization per step
# ---------------------------------------------------------------------------

def _newton_vmap(solver, in_axes, axes_tree, options, preconditioner, chunk, return_info,
                 u0_default, ts0):
    """lhs batch of the standard traced Newton solver.

    All cases iterate together: per Newton step the N tangents are assembled on
    the shared pattern, the batch-level preconditioner is built from them (by
    default one factorization of their mean), the N Newton directions come from
    one batched CG, and the Armijo line search runs per case. A converged case
    (or one whose line search failed) keeps its iterate while the others go on.
    Every case's final iterate is returned, converged or not; the
    :class:`BatchedNewtonInfo` says which cases converged, which stopped on a
    failed line search, and which ran out of ``max_iter``. The custom VJP
    solves the N adjoint systems at the returned iterates the same way
    (symmetric tangents).
    """
    from .solvers.newton import create_armijo_line_search_scan
    from .solvers.options import has_cudss

    problem, bc = solver.problem, solver.bc
    newton = solver.newton_options
    if options.solver not in ("auto", "cg"):
        raise NotImplementedError(
            f"the batched Newton solve uses conjugate gradients (SPD tangents); got solver={options.solver!r}.")
    if not solver.symmetric_elimination:
        raise NotImplementedError("the batched Newton solve needs a symmetric tangent (symmetric_elimination=True).")
    if preconditioner is None:
        own = solver.preconditioner
        if isinstance(own, AMGPreconditioner) or getattr(own, "__name__", None) in ("jacobi", "shared_factor"):
            preconditioner = own
    if preconditioner is None:
        preconditioner = shared_factor() if has_cudss() else jacobi()
    if isinstance(preconditioner, AMGPreconditioner):
        _amg = preconditioner
        preconditioner = lambda op: jax.vmap(_amg(op))
        preconditioner.__name__ = "amg"
    if not callable(preconditioner):
        raise TypeError("preconditioner must be callable, preconditioner(op) -> M.")

    pattern = _CsrPattern(solver, u0_default, solver.sample_params)
    tol, atol, maxiter = options.tol, options.atol, options.maxiter
    front = jax.tree_util.tree_map(lambda ax: None if ax is None else 0, axes_tree, is_leaf=_is_axis_leaf)

    def _operator(values, ts):
        indptr, indices = pattern(ts)
        return LinearOperator(values=values, indptr=indptr, indices=indices, structure=ts,
                              problem=problem, bc=bc, batched=True)

    def _fns(ts):
        res = lambda s, t: solver.residual(s, t, traced_structure=ts)
        v_res = jax.vmap(res, in_axes=(0, front))
        v_asm = jax.vmap(lambda s, t: solver.assemble(s, t, traced_structure=ts).data, in_axes=(0, front))
        armijo = create_armijo_line_search_scan(
            res, c1=newton.line_search_c1, rho=newton.line_search_rho,
            max_backtracks=newton.line_search_max_backtracks)
        v_ls = jax.vmap(lambda s, d, r, rn, t: armijo(s, d, r, rn, traced_params=t),
                        in_axes=(0, 0, 0, 0, front))
        return v_res, v_asm, v_ls

    def _batched_solve(values, rhs, ts):
        M = preconditioner(_operator(values, ts))
        indptr, indices = pattern(ts)
        return batched_pcg(values, indptr, indices, rhs, M, tol=tol, atol=atol, maxiter=maxiter)

    def _newton_group(tp_g, U0, ts):
        """The batched Newton loop of one group. ``U0`` is ``(n,)`` (shared) or ``(C, n)``."""
        v_res, v_asm, v_ls = _fns(ts)
        C = jax.tree_util.tree_leaves(jax.tree_util.tree_map(
            lambda leaf, ax: None if ax is None else leaf.shape[0], tp_g, front, is_leaf=_is_axis_leaf))[0]
        U = jnp.broadcast_to(U0, (C,) + U0.shape[-1:]) if U0.ndim == 1 else U0
        R = v_res(U, tp_g)
        rn = jnp.linalg.norm(R, axis=1)
        rn0 = jnp.maximum(rn, 1e-30)
        converged = lambda rn_: (rn_ <= newton.tol) | (rn_ / rn0 <= newton.rel_tol)
        state = (U, rn, jnp.zeros(C, jnp.int32), converged(rn), jnp.ones(C, bool),
                 jnp.zeros(C, jnp.int32), jnp.array(0, jnp.int32))

        def cond(st):
            return (st[6] < newton.max_iter) & ~jnp.all(st[3] | ~st[4])

        def body(st):
            U, rn, its, done, found, cg_its, k = st
            R = v_res(U, tp_g)
            values = v_asm(U, tp_g)
            X, info = _batched_solve(values, -R, ts)
            newU, newn, _alpha, ok = v_ls(U, X, R, rn, tp_g)
            active = ~(done | ~found)
            U = jnp.where(active[:, None], newU, U)
            rn = jnp.where(active, newn, rn)
            its = jnp.where(active, k + 1, its)
            cg_its = jnp.where(active, cg_its + info.iterations, cg_its)
            found = jnp.where(active, ok, found)
            done = done | converged(rn)
            return U, rn, its, done, found, cg_its, k + 1

        U, rn, its, done, found, cg_its, _ = jax.lax.while_loop(cond, body, state)
        # The final iterate of every case is returned (converged or not).
        info = BatchedNewtonInfo(iterations=its, converged=done, line_search_failed=~found,
                                 cg_iterations=cg_its, residual=rn / rn0)
        return U, info

    def _forward(tp, N, U0, ts):
        if chunk is None or chunk >= N:
            return _newton_group(tp, U0, ts)
        if U0.ndim == 1:
            return _chunked(lambda c: _newton_group(c, U0, ts), tp, front, N, chunk)
        return _chunked(lambda c, u: _newton_group(c, u, ts), tp, front, N, chunk, extra=U0)

    def _adjoint(tp, U, gU, ts):
        _, v_asm, _ = _fns(ts)

        def group(tp_g, U_g, gU_g):
            values = v_asm(U_g, tp_g)
            lam, _ = _batched_solve(values, gU_g, ts)
            return lam
        N = gU.shape[0]
        if chunk is None or chunk >= N:
            return group(tp, U, gU)
        UG = jnp.concatenate([U[..., None], gU[..., None]], axis=-1)   # chunk both together
        return _chunked(lambda c, ug: group(c, ug[..., 0], ug[..., 1]), tp, front, N, chunk, extra=UG)

    def _batch_size(tp):
        return jax.tree_util.tree_leaves(jax.tree_util.tree_map(
            lambda leaf, ax: None if ax is None else leaf.shape[0], tp, front, is_leaf=_is_axis_leaf))[0]

    @jax.custom_vjp
    def solve_newton(tp, U0, ts):
        return _forward(tp, _batch_size(tp), U0, ts)

    def fwd(tp, U0, ts):
        out = solve_newton(tp, U0, ts)
        return out, (tp, out[0], ts)

    def bwd(res, g):
        tp, U, ts = res
        gU, _ = g
        lam = _adjoint(tp, U, gU, ts)

        return _residual_cotangents(solver, U, lam, tp, ts, front, chunk), None, None

    solve_newton.defvjp(fwd, bwd)

    def solve(tp, u0, ts):
        u0 = u0_default if u0 is None else jnp.asarray(u0)
        ts = ts0 if ts is None else ts
        tp, _, _ = _to_front(tp, axes_tree)
        return solve_newton(tp, u0, ts)

    return BatchedSolver(solve, solver=solver, kind="lhs", in_axes=in_axes, options=options,
                         preconditioner=preconditioner, chunk=chunk, return_info=return_info)
