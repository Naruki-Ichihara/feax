---
sidebar_label: matrix_free
title: feax.solvers.matrix_free
---

Fully matrix-free Newton solver for nonlinear problems.

This module provides a matrix-free Newton solver where the tangent
operator is computed via JVP (Jacobian-vector product) of the residual,
eliminating the need for sparse matrix assembly.

The inner linear system at each Newton step is solved by an iterative
solver (CG, BiCGSTAB, or GMRES) applied to the JVP-based matvec.

The solver is suitable for problems with custom energy contributions
(e.g., cohesive zones, phase-field fracture) that are difficult to
express within the standard FE assembly framework.

Usage pattern:

```python
from feax.solvers.matrix_free import MatrixFreeOptions, newton_solve

# Define total energy as a pure JAX function
def total_energy(u_flat, *args):
    return elastic_energy(u_flat) + custom_energy(u_flat, *args)

# Solve
u, info = newton_solve(total_energy, u0, fixed_dofs, args=(delta_max,))
```

## LinearSolverOptions Objects

```python
@dataclass(frozen=True)
class LinearSolverOptions()
```

Options for the inner iterative linear solver.

Parameters
----------
- **solver** (*str, default &quot;cg&quot;*): Iterative solver algorithm: &quot;cg&quot;, &quot;bicgstab&quot;, or &quot;gmres&quot;.
- **tol** (*float, default 1e-10*): Relative tolerance for the iterative solver.
- **atol** (*float, default 1e-8*): Absolute tolerance for the iterative solver.
- **maxiter** (*int, default 200*): Maximum number of iterations.
- **restart** (*int, default 200*): Restart parameter for GMRES. Only used when solver=&quot;gmres&quot;.


## MatrixFreeOptions Objects

```python
@dataclass(frozen=True)
class MatrixFreeOptions(AbstractSolverOptions)
```

Options for the matrix-free Newton solver.

Can be passed as ``solver_options`` to :func:`feax.create_solver` to use
the fully matrix-free Newton path, or used directly with
:func:`newton_solve`.

Parameters
----------
- **newton_tol** (*float, default 1e-8*): Absolute tolerance on the Newton residual norm.
- **newton_max_iter** (*int, default 200*): Maximum number of Newton iterations.
- **linear_solver** (*LinearSolverOptions, default LinearSolverOptions()*): Options for the inner iterative linear solver.
- **verbose** (*bool*): Inherited from AbstractSolverOptions.  Log Newton iteration info via jax.debug.print (visible inside JIT).


#### solver

```python
@property
def solver() -> str
```

Solver identifier for dispatch in create_solver.

#### uses\_x0

```python
def uses_x0() -> bool
```

Matrix-free solver uses an initial iterate.

## NewtonInfo Objects

```python
@dataclass
class NewtonInfo()
```

Convergence information returned by newton_solve.

Attributes
----------
- **converged** (*bool*): Whether the solver converged within tolerance.
- **res_norm** (*float*): Final residual norm.
- **n_iter** (*int*): Number of Newton iterations performed.


#### newton\_solve

```python
def newton_solve(
    energy_fn: Callable,
    u0: jax.Array,
    fixed_dofs: jax.Array,
    args: tuple = (),
    options: Optional[MatrixFreeOptions] = None
) -> Tuple[jax.Array, NewtonInfo]
```

Fully matrix-free Newton solver.

Solves the nonlinear problem ``grad(energy_fn)(u, *args) = 0``
subject to Dirichlet boundary conditions (fixed DOFs).

The tangent operator is computed via JVP of the gradient,
requiring no sparse matrix assembly. The inner linear system
is solved by an iterative solver (CG, BiCGSTAB, or GMRES).

Parameters
----------
- **energy_fn** (*callable*)
- **u0** (*jax.Array*)
- **fixed_dofs** (*jax.Array*)
- **args** (*tuple, optional*)
- **options** (*MatrixFreeOptions, optional*)


Returns
-------
- **u** (*jax.Array*)
- **info** (*NewtonInfo*)


Examples
--------
```python
>>> def energy(u, delta_max):
...     return elastic_energy(u) + cohesive_energy(u, delta_max)
>>> u_sol, info = newton_solve(energy, u0, fixed_dofs, args=(delta_max,))
>>> if not info.converged:
...     print(f&quot;Warning: did not converge, res={`info.res_norm:.2e`}&quot;)
```

#### create\_energy\_fn

```python
def create_energy_fn(problem) -> Callable
```

Create an energy integration function from a feax Problem.

Builds a pure JAX function that computes the total energy by
integrating the problem&#x27;s energy density over the domain:

```python
E(u) = ∫ ψ(∇u) dΩ
```

The energy density is obtained from ``problem.get_energy_density()``.

Parameters
----------
- **problem** (*feax.Problem*): A feax problem. Must define ``get_energy_density()`` returning a non-None callable.


Returns
-------
- **energy** (*callable*): Function ``energy(u_flat) -&gt; scalar`` that computes the total energy ∫ ψ(∇u) dΩ.


Raises
------
ValueError
    If ``problem.get_energy_density()`` returns None.

#### create\_matrix\_free\_solver

```python
def create_matrix_free_solver(problem,
                              bc,
                              options: Optional[MatrixFreeOptions] = None,
                              energy_fn: Optional[Callable] = None)
```

Create a differentiable matrix-free Newton solver.

Returns a callable with the same ``(internal_vars, initial_guess) -&gt; solution``
signature used by :func:`feax.create_solver`, enabling use as a drop-in
replacement for assembly-based solvers.

The tangent operator is computed via JVP (forward-mode AD) of the residual,
and the inner linear system is solved by an iterative Krylov method.

Parameters
----------
- **problem** (*feax.Problem*): A feax problem.  When ``energy_fn`` is not provided, the problem must define ``get_energy_density()`` returning a non-None callable.
- **bc** (*feax.DirichletBC*): Dirichlet boundary conditions.
- **options** (*MatrixFreeOptions, optional*): Solver options.  Uses defaults if not provided.
- **energy_fn** (*callable, optional*): Custom total energy function with signature ``energy_fn(u_flat, internal_vars) -&gt; scalar``. When provided, this is used instead of auto-generating from ``problem.get_energy_density()``.  This is useful for problems with extra energy contributions (e.g. cohesive zones):


Returns
-------
- **solver** (*callable*): ``solver(internal_vars, initial_guess) -&gt; solution``. Supports ``jax.grad`` via a custom VJP (adjoint method).


Examples
--------
```python
>>> # Simple: auto-generate energy from problem
>>> solver = create_matrix_free_solver(problem, bc, MatrixFreeOptions(newton_tol=1e-6))
>>> sol = solver(internal_vars, initial_guess)
```
```python
>>> # Custom energy (e.g. elastic + cohesive)
>>> def total_energy(u_flat, delta_max):
...     return elastic_energy(u_flat) + cohesive_energy(u_flat, delta_max)
>>> solver = create_matrix_free_solver(problem, bc, options, energy_fn=total_energy)
>>> sol = solver(delta_max, initial_guess)
```

