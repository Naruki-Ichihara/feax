---
sidebar_label: newton
title: feax.solvers.newton
---

Newton solver for FEAX.

Provides:

* :func:`create_newton_solver` — the entry point used by ``feax.solver``. With
  ``extra_residual_fn`` it runs the hybrid matrix-free Newton-Krylov path;
  otherwise it delegates to the unified callback solver below.
* :func:`create_callback_newton_solver` — the standard nonlinear solver. The
  iteration runs as a host loop inside a single ``jax.pure_callback`` (so it
  composes with an outer ``jax.jit`` / ``jax.vmap`` / ``jax.grad``), with a
  CSR-direct fused residual+Jacobian forward and a single traced adjoint solve.
* :func:``4 — the (vmap-friendly) line search.

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
                         internal_vars=None,
                         extra_residual_fn=None,
                         symmetric_bc: bool = True)
```

Create a differentiable nonlinear Newton solver.

The Newton iteration always runs adaptively (to ``newton_options.tol`` /
``rel_tol``, capped at ``max_iter``). For a single linear solve use
:func:`feax.solvers.linear.create_linear_solver` instead.

Parameters
----------
- **extra_residual_fn** (*callable, optional*): Additional residual: ``extra_residual_fn(sol_flat) -&gt; residual_flat``. When provided, uses hybrid matrix-free Newton-Krylov: feax assembles the bulk Jacobian (sparse), and the extra contribution&#x27;s JVP is computed via ``jax.jvp``.  The combined matvec is: ``J_total @ v = J_bulk @ v + jvp(extra_res_bc, sol, v)``. Dirichlet BC rows of the extra residual are zeroed automatically.


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
                                  internal_vars=None,
                                  symmetric_bc: bool = True)
```

Create a differentiable, jit/vmap-safe Newton solver via a host callback.

The Newton iteration (inherently data-dependent) runs as a host Python loop
that dispatches the compiled per-iteration kernels (fused CSR
residual+Jacobian assembly + linear solve). The whole loop is wrapped in a
single :func:`jax.pure_callback`, so tracing it under an outer ``jax.jit``
inserts one callback node — no giant fused compile, no ``float(norm)``
tracing error. A ``custom_vmap`` rule routes batched calls to a vectorized
host loop (block-diagonal direct solves). The backward pass is a single
adjoint linear solve plus the residual VJP — ordinary traced JAX, so it is
natively jit/vmap/grad compatible.

Returns ``solver(internal_vars, initial_guess, bc=None) -&gt; solution``.

