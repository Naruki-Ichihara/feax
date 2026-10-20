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

from .solver import Solver
from .solvers.batched import BatchedNewtonInfo, BatchedSolveInfo, batched_pcg
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


# ---------------------------------------------------------------------------
# the batched solver
# ---------------------------------------------------------------------------

class BatchedSolver:
    """What :func:`vmap` returns. ``kind`` is ``"rhs"`` or ``"lhs"``."""

    def __init__(self, fn, *, solver, kind, in_axes, options, preconditioner, chunk, return_info):
        self._fn = fn
        self.solver = solver
        self.kind = kind
        self.in_axes = in_axes
        self.options = options
        self.preconditioner = preconditioner
        self.chunk = chunk
        self.return_info = return_info

    def __call__(self, traced_params, initial_guess=None, traced_structure=None):
        from .solution import Solution
        if isinstance(initial_guess, Solution):
            initial_guess = initial_guess.dofs
        U, info = self._fn(traced_params, initial_guess, traced_structure)
        if self.solver.return_solution:
            U = Solution(U, self.solver.layout)
        return (U, info) if self.return_info else U

    def __repr__(self):
        return (f"BatchedSolver({self.kind}, {self.solver!r}, chunk={self.chunk}, "
                f"preconditioner={getattr(self.preconditioner, '__name__', self.preconditioner)!r})")


def vmap(solver: Solver, in_axes: Any = 0, *, batch: str = "auto",
         options: Optional[Any] = None, preconditioner: Optional[Callable] = None,
         chunk: Optional[int] = None, return_info: bool = False) -> BatchedSolver:
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
        preconditioner per group and a memory bound of ``chunk`` operators.
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

    # the shared pattern, from one concrete assembly at the sample parameters
    K0 = solver.assemble(u0_default, sample)
    indptr, indices = K0.indptr, K0.indices
    tol, atol, maxiter = options.tol, options.atol, options.maxiter
    # which leaves carry the batch (axis 0 once normalized): fixed by in_axes
    front = jax.tree_util.tree_map(lambda ax: None if ax is None else 0, axes_tree, is_leaf=_is_axis_leaf)

    def _operator(values, ts):
        return LinearOperator(values=values, indptr=indptr, indices=indices, structure=ts,
                              problem=problem, bc=bc, batched=True)

    def _solve_group(tp_g, u0, ts, rhs):
        """One group: assemble its operators, build M, batched CG. ``rhs`` is
        None (forward: b = -r(u0)) or the (C, n) adjoint right-hand sides."""
        values = jax.vmap(lambda t: solver.assemble(u0, t, traced_structure=ts).data,
                          in_axes=(front,))(tp_g)
        if rhs is None:
            rhs = -jax.vmap(lambda t: solver.residual(u0, t, traced_structure=ts),
                            in_axes=(front,))(tp_g)
        M = preconditioner(_operator(values, ts))
        return batched_pcg(values, indptr, indices, rhs, M, tol=tol, atol=atol, maxiter=maxiter)

    def _solve_batch(tp, N, u0, ts, rhs=None):
        if chunk is None or chunk >= N:
            return _solve_group(tp, u0, ts, rhs)
        if rhs is None:
            return _chunked(lambda c: _solve_group(c, u0, ts, None), tp, front, N, chunk)
        return _chunked(lambda c, r: _solve_group(c, u0, ts, r), tp, front, N, chunk, extra=rhs)

    def _batch_size(tp):
        return jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda leaf, ax: None if ax is None else leaf.shape[0],
                                   tp, front, is_leaf=_is_axis_leaf))[0]

    @jax.custom_vjp
    def solve_lhs(tp, u0, ts):
        X, info = _solve_batch(tp, _batch_size(tp), u0, ts)
        return u0 + X, info

    def fwd(tp, u0, ts):
        out = solve_lhs(tp, u0, ts)
        return out, (tp, out[0], u0, ts)

    def bwd(res, g):
        tp, U, u0, ts = res
        gU, _ = g
        # adjoint: K_i λ_i = v_i (symmetric operators), solved like the forward batch
        lam, _ = _solve_batch(tp, gU.shape[0], u0, ts, rhs=gU)

        def one(u_i, lam_i, tp_i):
            _, f_vjp = jax.vjp(lambda t: solver.residual(u_i, t, traced_structure=ts), tp_i)
            return jax.tree_util.tree_map(_safe_negate, f_vjp(lam_i)[0])
        ct = jax.vmap(one, in_axes=(0, 0, front))(U, lam, tp)
        ct = jax.tree_util.tree_map(
            lambda c, ax: c if ax is not None else jnp.sum(c, axis=0), ct, front, is_leaf=_is_axis_leaf)
        return ct, None, None

    solve_lhs.defvjp(fwd, bwd)

    def solve(tp, u0, ts):
        u0 = u0_default if u0 is None else u0
        ts = ts0 if ts is None else ts
        tp, _, _ = _to_front(tp, axes_tree)
        return solve_lhs(tp, u0, ts)

    return BatchedSolver(solve, solver=solver, kind="lhs", in_axes=in_axes, options=options,
                         preconditioner=preconditioner, chunk=chunk, return_info=return_info)


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

    K0 = solver.assemble(u0_default, solver.sample_params)
    indptr, indices = K0.indptr, K0.indices
    tol, atol, maxiter = options.tol, options.atol, options.maxiter
    front = jax.tree_util.tree_map(lambda ax: None if ax is None else 0, axes_tree, is_leaf=_is_axis_leaf)

    def _operator(values, ts):
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

        def one(u_i, lam_i, tp_i):
            _, f_vjp = jax.vjp(lambda t: solver.residual(u_i, t, traced_structure=ts), tp_i)
            return jax.tree_util.tree_map(_safe_negate, f_vjp(lam_i)[0])
        ct = jax.vmap(one, in_axes=(0, 0, front))(U, lam, tp)
        ct = jax.tree_util.tree_map(
            lambda c, ax: c if ax is not None else jnp.sum(c, axis=0), ct, front, is_leaf=_is_axis_leaf)
        return ct, None, None

    solve_newton.defvjp(fwd, bwd)

    def solve(tp, u0, ts):
        u0 = u0_default if u0 is None else jnp.asarray(u0)
        ts = ts0 if ts is None else ts
        tp, _, _ = _to_front(tp, axes_tree)
        return solve_newton(tp, u0, ts)

    return BatchedSolver(solve, solver=solver, kind="lhs", in_axes=in_axes, options=options,
                         preconditioner=preconditioner, chunk=chunk, return_info=return_info)
