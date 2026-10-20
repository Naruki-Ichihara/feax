---
sidebar_label: mma
title: feax.experimental.mma
---

MMA / GCMMA in JAX (experimental).

Pure-JAX implementation of Svanberg&#x27;s *Method of Moving Asymptotes* (MMA) and
its *Globally Convergent* variant (GCMMA), following the reference MATLAB
code (``mmasub.m``, ``gcmmasub.m``, ``subsolv.m``, ``asymp.m``, ``concheck.m``,
``raaupdate.m``, ``kktcheck.m``, version September 2007) [1]_ [2]_.

Problem statement (Svanberg&#x27;s standard form)::

    minimise    f0(x) + a0*z + sum_i (c_i*y_i + 0.5*d_i*y_i**2)
    subject to  f_i(x) - a_i*z - y_i &lt;= 0,   i = 1..m
                xmin &lt;= x &lt;= xmax,  y &gt;= 0,  z &gt;= 0

With the defaults ``a0 = 1``, ``a = 0``, ``c = 1000``, ``d = 1`` this is the
usual ``min f0(x) s.t. f_i(x) &lt;= 0`` with elastic (penalised) constraints.

.. warning::
   ``c`` is the elastic penalty on the constraint slacks ``y``.  At the
   optimum ``y_i &gt; 0`` whenever the true multiplier of constraint ``i``
   exceeds ``c_i``, and the constraint is then *left violated*.  Scale the
   objective and the constraints so that the expected multipliers stay well
   below ``c`` (or raise ``c``).  ``f0`` of order 1 and constraints written
   as ``g/g_target - 1`` is a safe recipe.

Accuracy is limited by the interior point barrier: the final iterate is
``O(epsimin)`` away from the exact optimum, so the default ``epsimin=1e-7``
of the reference code gives roughly 8 correct digits.  Lower it for tighter
convergence.

**Everything is traced.**  Every loop in the algorithm — the interior point
barrier loop, its Newton iteration, the Newton line search, the GCMMA inner
(conservativity) loop, and the outer MMA iteration itself — is a
``lax.while_loop`` or ``lax.scan``.  No loop bound is a Python ``for``, so the
XLA graph does not grow with the iteration count and a *complete* optimisation
run compiles into a single kernel launch sequence.  The subproblem uses the
same primal-dual Newton method as ``subsolv.m``, with the ``m &lt; n``
Schur-complement branch, so the dense linear solve is only ``(m+1)x(m+1)`` for
the typical topology-optimisation case ``m &lt;&lt; n``.

Usage — the whole optimisation inside one ``jit``::

    from feax.experimental.mma import GCMMA

    def value_and_grad(x):
        # f0: (), df0: (n,), f: (m,), df: (m, n)     with constraints f &lt;= 0
        return f0, df0, f, df

    opt = GCMMA(value_and_grad, xmin, xmax, n_constraints=m)
    state = opt.solve(x0, max_iter=200, kkt_tol=1e-6)   # traced end-to-end
    x_opt = state.x

``solve`` is itself ``jax.jit``-compatible (and ``jax.vmap``-compatible over
``x0`` and extra arguments), so a batch of designs can be optimised at once::

    solve = jax.jit(lambda x0: opt.solve(x0, max_iter=200))
    batch = jax.vmap(solve)(x0_batch)

Use :meth:``0 to get per-iteration convergence data out of
the same compiled program, and :meth:``1 for a Python-level loop with a
callback (for VTU snapshots, logging, adaptive remeshing, ...).

The plain ``MMA`` class has the same interface (no inner loop, move limit
``move=0.5`` by default) and, because it never re-evaluates the functions
inside a traced loop, also works un-jitted with arbitrary Python callables via
:meth:``1.

References
----------
.. [1] K. Svanberg, &quot;The method of moving asymptotes — a new method for
       structural optimization&quot;, Int. J. Numer. Meth. Engng 24 (1987) 359–373.
.. [2] K. Svanberg, &quot;A class of globally convergent optimization methods based
       on conservative convex separable approximations&quot;, SIAM J. Optim. 12
       (2002) 555–573.  Implementation notes: &quot;MMA and GCMMA – two methods for
       nonlinear optimization&quot; (2007).

## Subproblem Objects

```python
class Subproblem(NamedTuple)
```

MMA/GCMMA convex separable approximation around the current iterate.

``f0app(x) = r0 + p0/(upp-x) + q0/(x-low)``,
``fapp(x)  = r  + P/(upp-x)  + Q/(x-low)`` (row-wise), with the trust
region ``alfa &lt;= x &lt;= beta``.

## SubproblemSolution Objects

```python
class SubproblemSolution(NamedTuple)
```

Primal-dual solution of the subproblem (variables of ``subsolv.m``).

## MMAState Objects

```python
class MMAState(NamedTuple)
```

Optimiser state carried between outer iterations.

#### f0

f0(x)

#### df0dx

(n,)

#### fval

(m,)

#### dfdx

(m, n)

#### sol

duals of the last subproblem (for the KKT check)

#### iteration

number of outer iterations performed

#### inner\_iterations

inner iterations used in the last outer step

#### kkt\_norm

||KKT residual||_2 at x

#### kkt\_max

||KKT residual||_inf at x

#### f0app

last subproblem objective value at x

#### fapp

last subproblem constraint values at x

#### aux

user pytree from value_and_grad_fn (has_aux)

#### update\_asymptotes

```python
def update_asymptotes(iteration,
                      x,
                      xold1,
                      xold2,
                      xmin,
                      xmax,
                      low,
                      upp,
                      *,
                      asyinit=0.5,
                      asyincr=1.2,
                      asydecr=0.7,
                      asymin=0.01,
                      asymax=10.0)
```

Moving asymptotes update (``asymp.m`` / first part of ``mmasub.m``).

``iteration`` is the 1-based outer iteration counter; for the first two
iterations the asymptotes are initialised at ``x -/+ asyinit*(xmax-xmin)``.

#### move\_bounds

```python
def move_bounds(x, xmin, xmax, low, upp, *, albefa=0.1, move=None)
```

Subproblem box ``alfa &lt;= x &lt;= beta`` (``move=None`` → no move limit).

#### initial\_raa

```python
def initial_raa(df0dx, dfdx, xmin, xmax, *, raa0eps=1e-6, raaeps=1e-6)
```

GCMMA initial ``raa0``, ``raa`` for an outer iteration (``asymp.m``).

#### build\_subproblem

```python
def build_subproblem(x, xmin, xmax, low, upp, alfa, beta, f0, df0dx, fval,
                     dfdx, raa0, raa) -> Subproblem
```

Convex separable approximation (``gcmmasub.m`` / ``mmasub.m``).

``raa0`` (scalar) and ``raa`` (m,) are the curvature parameters; for
plain MMA they are all ``1e-5``.

#### evaluate\_approximation

```python
def evaluate_approximation(sub: Subproblem, x)
```

Values ``(f0app, fapp)`` of the separable approximations at ``x``.

#### solve\_subproblem

```python
def solve_subproblem(sub: Subproblem,
                     a0,
                     a,
                     c,
                     d,
                     *,
                     epsimin=1e-7,
                     max_newton=200,
                     max_linesearch=50) -> SubproblemSolution
```

Primal-dual interior point solver for the subproblem (``subsolv.m``).

Three nested ``lax.while_loop``s: barrier parameter ``epsi`` (1 → epsimin,
factor 0.1), Newton iterations (until ``||res||_inf &lt;= 0.9*epsi``), and a
residual-norm halving line search.

#### is\_conservative

```python
def is_conservative(f0app, f0new, fapp, fnew, *, epsimin=1e-7)
```

``concheck.m``: are the approximations conservative at the new point?

#### update\_raa

```python
def update_raa(xnew,
               x,
               xmin,
               xmax,
               low,
               upp,
               f0new,
               fnew,
               f0app,
               fapp,
               raa0,
               raa,
               *,
               epsimin=1e-7,
               raacofmin=1e-12)
```

``raaupdate.m``: increase curvature of non-conservative approximations.

#### kkt\_residual

```python
def kkt_residual(sol: SubproblemSolution, xmin, xmax, df0dx, fval, dfdx, a0, a,
                 c, d)
```

``kktcheck.m``: KKT residual of the *original* problem at ``sol.x``
using the subproblem multipliers.  Returns ``(norm2, norminf)``.

## MMA Objects

```python
class MMA()
```

Method of Moving Asymptotes (Svanberg 1987, 2007 ``mmasub.m``).

Parameters
----------
- **value_and_grad_fn** (*callable*): ``fn(x, *args) -&gt; (f0, df0dx, fval, dfdx)`` with shapes ``(), (n,), (m,), (m, n)``.  Constraints are ``fval &lt;= 0``.
- **n_constraints** (*int*): Number of constraints ``m`` (``fval`` may be empty when ``m = 0``).
- **move** (*float or None*): Move limit as a fraction of ``xmax - xmin`` (``None`` = no limit).
- **has_aux** (*bool*): If ``True``, ``value_and_grad_fn`` returns a fifth element -- any JAX pytree -- which is carried in ``MMAState.aux``.  Use it to get derived quantities (diagnostics, logged fields, intermediate solutions) out of the same evaluation that produced the gradients, instead of paying for a second one.
- **epsimin** (*float*): Final barrier parameter of the subproblem solver.
- **raa0** (*float*): Fixed curvature parameter of plain MMA (``1e-5`` in ``mmasub.m``).


#### init

```python
def init(x0, *args) -> MMAState
```

Create the initial state (evaluates the functions at ``x0``).

#### step

```python
def step(state: MMAState, *args) -> MMAState
```

One outer MMA iteration.

#### solve

```python
def solve(x0,
          *args,
          max_iter: int = 100,
          kkt_tol: float = 0.0,
          xtol: float = 0.0,
          ftol: float = 0.0) -> MMAState
```

Fully traced driver — the entire optimisation is one XLA program.

The outer loop is a ``lax.while_loop``, so the compiled graph is
independent of ``max_iter`` and the whole solve runs without returning
to Python.  ``solve`` composes with ``jax.jit``, ``jax.vmap`` and
``jax.pmap``; the user functions must therefore be JAX-traceable.

Parameters
----------
- **max_iter** (*int*): Hard cap on outer iterations (static).


Returns
-------
MMAState
    Final state; ``state.iteration`` reports how many outer iterations
    were actually performed.

#### solve\_with\_history

```python
def solve_with_history(x0,
                       *args,
                       n_iter: int = 100) -> Tuple[MMAState, Dict[str, Array]]
```

Traced driver with per-iteration history (``lax.scan``).

Runs exactly ``n_iter`` outer iterations (no early exit — that is what
``scan`` buys) and returns the final state plus stacked arrays of
length ``n_iter`` for ``objective``, ``constraints``, ``kkt_norm``,
``kkt_max`` and ``inner_iterations``.  Useful for convergence plots
and CSV logs without breaking out of the compiled program.

#### run

```python
def run(x0,
        *args,
        max_iter=100,
        kkt_tol=0.0,
        xtol=0.0,
        ftol=0.0,
        jit=True,
        callback: Optional[Callable[[MMAState], None]] = None) -> MMAState
```

Python-loop driver, for callbacks and non-JAX callables.

Prefer :meth:`solve` unless you need to run Python between iterations
(VTU snapshots, logging, continuation, adaptive remeshing) or your
``value_and_grad_fn`` is not traceable.

Iterates :meth:`step` until ``max_iter`` is reached or one of the
(disabled by default) stopping tests fires:

* ``kkt_tol`` — ``kkt_norm &lt; kkt_tol``.  Note that ``kkt_norm`` floors
  at ``O(epsimin)``, so a tolerance below that never triggers.
* ``xtol`` — ``max|x - x_prev| &lt; xtol * max(1, max|x|)``.
* ``ftol`` — ``|f0 - f0_prev| &lt; ftol * max(1, |f0|)``.

``callback`` receives the state after every outer iteration.

## GCMMA Objects

```python
class GCMMA(MMA)
```

Globally Convergent MMA (Svanberg 2002; 2007 ``gcmmasub.m`` et al.).

Same interface as :class:`MMA`.  Each outer iteration solves the
subproblem, evaluates the functions at the candidate and, while the
approximations are not conservative (``concheck``), increases the
curvature parameters (``raaupdate``) and re-solves — up to
``max_inner`` times.

Parameters (in addition to :class:`MMA`)
----------
value_fn : callable or None
    ``fn(x, *args) -&gt; (f0, fval)`` used for the inner-loop evaluations.
    If ``None`` (default) the inner loop calls ``value_and_grad_fn`` and
    keeps the gradients of the accepted point, so no extra evaluation is
    needed at the end of the outer iteration.  Supply ``value_fn`` when a
    value-only evaluation is much cheaper than value-and-gradient.
max_inner : int
    Maximum number of inner iterations (15 in the reference driver).
raa0eps, raaeps : float
    Lower bounds of the curvature parameters.

#### step

```python
def step(state: MMAState, *args) -> MMAState
```

One outer GCMMA iteration (with inner conservativity loop).

## BatchedGCMMA Objects

```python
class BatchedGCMMA(GCMMA)
```

GCMMA over a batch of ``B`` independent problems that are *evaluated together*.

Use it when the ``B`` objective evaluations are cheapest as one call — the
designs of a :func:`feax.vmap` batch, whose ``B`` finite-element solves
share one factorization. ``jax.vmap`` of :class:`GCMMA` would evaluate
every design on its own; here every evaluation is one call on the whole
batch, while the subproblems, asymptotes and the conservativity loop are
per problem (vectorized, with the inner loop masked so that a problem whose
approximation is already conservative keeps its candidate while the others
re-solve).

Parameters
----------
- **value_and_grad_fn** (*callable*): ``fn(X, *args) -&gt; (f0, df0dx, fval, dfdx)`` on the batch, with shapes ``(B,), (B, n), (B, m), (B, m, n)`` (``+ aux`` with ``has_aux``); constraints are ``fval &lt;= 0`` per problem.
- **value_fn** (*callable or None*): ``fn(X, *args) -&gt; (f0, fval)`` on the batch, for the inner loop.
- **and** (*meth:``8 work unchanged; a stopping test fires when *every**)


#### init

```python
def init(X0, *args) -> MMAState
```

Create the initial state (one evaluation of the whole batch).

#### step

```python
def step(state: MMAState, *args) -> MMAState
```

One outer iteration of every problem, with a masked inner loop.

#### run

```python
def run(X0,
        *args,
        max_iter=100,
        kkt_tol=0.0,
        xtol=0.0,
        ftol=0.0,
        jit=True,
        callback: Optional[Callable[[MMAState], None]] = None) -> MMAState
```

Python-loop driver (see :meth:`MMA.run`); stops when every problem
passes a stopping test.

