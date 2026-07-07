---
sidebar_label: newton
title: feax.solvers.newton
---

Newton solver for FEAX.

Provides:

* :func:`create_newton_solver` — the entry point used by ``feax.solver``. With
  ``extra_residual_fn`` it runs the hybrid matrix-free Newton-Krylov path;
  otherwise it delegates to the unified traced solver below.
* :func:`create_callback_newton_solver` — the standard nonlinear solver. The
  iteration is a traced ``jax.lax.while_loop`` (one Newton step per loop body;
  the graph does not grow with ``max_iter``), so it composes natively with
  ``jax.jit`` / ``jax.vmap`` / ``jax.grad``, with a CSR-direct fused
  residual+Jacobian forward and a single traced adjoint solve.
* :func:``6 — the (vmap-friendly) line search.

## NewtonLineSearchError Objects

```python
class NewtonLineSearchError(RuntimeError)
```

Raised when Armijo line search exhausts backtracks without a descent step.

A failed line search means the proposed Newton direction is not a
descent direction for the residual merit function ½‖r‖².  This is
almost always a symptom of an inconsistent Jacobian, a bad linear
solve, or a degenerate state, rather than a problem the line search
itself can recover from — so the Python-loop Newton path raises this
by default rather than silently truncating the iteration.

#### create\_newton\_solver

```python
def create_newton_solver(problem,
                         bc,
                         linear_options,
                         adjoint_linear_options,
                         newton_options: Optional[NewtonOptions] = None,
                         traced_params=None,
                         extra_residual_fn=None,
                         symmetric_elimination: bool = True,
                         traced_structure=None)
```

Create a differentiable nonlinear Newton solver.

The Newton iteration always runs adaptively (to ``newton_options.tol`` /
``rel_tol``, capped at ``max_iter``). For a single linear solve use
:func:`feax.solvers.linear.create_linear_solver` instead.

Parameters
----------
- **extra_residual_fn** (*callable, optional*): Additional residual: ``extra_residual_fn(sol_flat) -&gt; residual_flat``. Dirichlet BC rows of the extra residual are zeroed automatically. Two paths, chosen by ``linear_options``:


#### create\_armijo\_line\_search\_scan

```python
def create_armijo_line_search_scan(res_bc_applied,
                                   c1=1e-4,
                                   rho=0.5,
                                   max_backtracks=30)
```

Create JAX scan-based Armijo line search (vmap-friendly).

#### create\_callback\_newton\_solver

```python
def create_callback_newton_solver(problem,
                                  bc,
                                  linear_options,
                                  adjoint_linear_options,
                                  newton_options: NewtonOptions = None,
                                  traced_params=None,
                                  symmetric_elimination: bool = True,
                                  traced_structure=None)
```

Create a differentiable, jit/vmap-safe Newton solver (traced loop).

The Newton iteration (inherently data-dependent) is a
:func:`jax.lax.while_loop` whose body is one Newton step (fused CSR
residual+Jacobian assembly + linear solve + Armijo line search). The body
is traced once, so the compiled graph holds a single step regardless of
``max_iter``, and the solver composes natively with ``jax.jit`` /
``jax.vmap`` / ``jax.grad``. The backward pass is a single adjoint linear
solve plus the residual VJP (custom_vjp — nothing differentiates through
the loop).

``raise_on_line_search_failure`` is honored for eager calls only; under
jit/vmap a failed line search stops the iteration (no exception can be
raised from traced values).

Returns ``solver(traced_params, initial_guess, bc=None, traced_structure=None)
-&gt; solution``.

