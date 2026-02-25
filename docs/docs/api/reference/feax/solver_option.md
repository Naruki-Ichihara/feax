---
sidebar_label: solver_option
title: feax.solver_option
---

Solver configuration options for FEAX finite element framework.

This module provides configuration dataclasses and enums for controlling
solver behavior, including linear solver selection, tolerances, and
CUDA-specific options.

## Backend Objects

```python
class Backend(Enum)
```

JAX compute backend.

#### CUDA

JAX reports GPU as &quot;gpu&quot;; we expose it as CUDA

#### detect\_backend

```python
def detect_backend() -> Backend
```

Detect the active JAX backend.

Returns
-------
Backend
    Backend.CUDA if a CUDA device is active, Backend.CPU otherwise.

#### is\_cuda

```python
def is_cuda() -> bool
```

Check if the active backend is CUDA.

#### is\_cpu

```python
def is_cpu() -> bool
```

Check if the active backend is CPU.

#### has\_cudss

```python
def has_cudss() -> bool
```

Check if the cuDSS direct solver is available.

Requires CUDA backend and the ``spineax`` package.

#### has\_spsolve

```python
def has_spsolve() -> bool
```

Check if JAX&#x27;s experimental spsolve is available.

Only supported on the CPU backend.

## MatrixProperty Objects

```python
class MatrixProperty(Enum)
```

Algebraic properties of the system matrix.

Used by auto solver selection to choose the best algorithm.

- GENERAL: No special structure (LU / GMRES)
- SYMMETRIC: Symmetric A = A^T (LDLT / BICGSTAB)
- SPD: Symmetric Positive Definite (Cholesky / CG)

#### view

```python
def view()
```

Return the recommended MatrixView for this property.

Returns
-------
MatrixView
    GENERAL → FULL, SYMMETRIC/SPD → UPPER.

#### detect\_matrix\_property

```python
def detect_matrix_property(A,
                           sym_tol: float = 1e-8,
                           matrix_view=None) -> MatrixProperty
```

Detect matrix property from an assembled sparse matrix.

Performs numerical checks on the matrix:
1. Symmetry: compares ``A @ x`` vs ``A^T @ x`` for a random vector.
   Skipped when ``matrix_view`` is UPPER or LOWER (the matrix is
   symmetric by construction).
2. Positive definiteness (heuristic): checks that all diagonal entries
   are positive, which is a necessary condition for SPD.

Parameters
----------
- **A** (*BCOO sparse matrix*): The assembled system matrix.
- **sym_tol** (*float, default 1e-8*): Relative tolerance for the symmetry check.
- **matrix_view** (*MatrixView, optional*): Storage format of the matrix.  When UPPER or LOWER, the matrix stores only one triangular half and is symmetric by definition, so the symmetry check is skipped.


Returns
-------
MatrixProperty
    SPD if symmetric with all-positive diagonal,
    SYMMETRIC if symmetric but diagonal has non-positive entries,
    GENERAL otherwise.

Notes
-----
The diagonal positivity test is a necessary but not sufficient
condition for positive definiteness.  For FEM stiffness matrices
from well-posed problems this is a reliable indicator.

## CUDSSMatrixType Objects

```python
class CUDSSMatrixType(Enum)
```

Matrix type for cuDSS solver.

Determines which factorization method cuDSS will use internally.

#### GENERAL

Non-symmetric matrix (uses LU factorization)

#### SYMMETRIC

Symmetric matrix (uses LDLT factorization)

#### HERMITIAN

Hermitian matrix (uses LDLT for complex)

#### SPD

Symmetric Positive Definite (uses Cholesky factorization)

#### HPD

Hermitian Positive Definite (uses Cholesky for complex)

## CUDSSMatrixView Objects

```python
class CUDSSMatrixView(Enum)
```

Matrix view (which portion of matrix is provided) for cuDSS solver.

#### FULL

Full matrix provided

#### UPPER

Upper triangular portion only

#### LOWER

Lower triangular portion only

## CUDSSOptions Objects

```python
@dataclass(frozen=True)
class CUDSSOptions()
```

Options specific to the NVIDIA cuDSS direct solver.

cuDSS matrix configuration is defined by two orthogonal properties:
1) Matrix Type (algebraic structure determining factorization method)
2) Matrix View (which triangular portion is provided)

Parameters
----------
- **matrix_type** (*CUDSSMatrixType or str or int, default SYMMETRIC*): Matrix type determining factorization method: - GENERAL: Non-symmetric (uses LU factorization) - SYMMETRIC: Symmetric indefinite (uses LDLT factorization) - HERMITIAN: Hermitian indefinite (uses LDLT for complex) - SPD: Symmetric Positive Definite (uses Cholesky factorization) - HPD: Hermitian Positive Definite (uses Cholesky for complex)
- **matrix_view** (*CUDSSMatrixView or str or int, default FULL*): Which portion of the matrix is provided: - FULL: All entries provided - UPPER: Upper triangular portion only - LOWER: Lower triangular portion only
- **device_id** (*int, default 0*): CUDA device ID to use


Examples
--------
```python
>>> # Using enums (recommended)
>>> opts = CUDSSOptions(
...     matrix_type=CUDSSMatrixType.SPD,
...     matrix_view=CUDSSMatrixView.FULL
... )
>>>
>>> # Using strings (convenient)
>>> opts = CUDSSOptions(matrix_type=&quot;SPD&quot;, matrix_view=&quot;FULL&quot;)
>>>
>>> # Using integers (backward compatible)
>>> opts = CUDSSOptions(matrix_type=3, matrix_view=0)
```

Notes
-----
Typical combinations:
- General sparse: matrix_type=GENERAL, matrix_view=FULL
- Symmetric: matrix_type=SYMMETRIC, matrix_view=FULL
- SPD (Cholesky): matrix_type=SPD, matrix_view=FULL

For GENERAL matrices, cuDSS treats the view as FULL (the view setting is
effectively ignored).

With symmetric elimination (default in FEAX), matrix_view should always be FULL
since both upper and lower triangular portions are modified.

#### \_\_post\_init\_\_

```python
def __post_init__()
```

Validate and convert matrix_type and matrix_view to enums if needed.

#### mtype\_id

```python
@property
def mtype_id() -> int
```

Get matrix type as integer ID for cuDSS.

#### mview\_id

```python
@property
def mview_id() -> int
```

Get matrix view as integer ID for cuDSS.

## AbstractSolverOptions Objects

```python
@dataclass(frozen=True)
class AbstractSolverOptions()
```

Base class for all solver option types.

Common parameters shared by DirectSolverOptions,
IterativeSolverOptions, and SolverOptions.

Parameters
----------
- **verbose** (*bool, default False*): Whether to print solver diagnostics.
- **check_convergence** (*bool, default False*): Whether to verify solution by checking residual norm.
- **convergence_threshold** (*float, default 0.1*): Maximum allowable relative residual for convergence check. Only used when check_convergence=True.


## SolverOptions Objects

```python
@dataclass(frozen=True)
class SolverOptions(AbstractSolverOptions)
```

Configuration options for the Newton solver.

Parameters
----------
- **tol** (*float, default 1e-6*): Absolute tolerance for residual vector (l2 norm)
- **rel_tol** (*float, default 1e-8*): Relative tolerance for residual vector (l2 norm)
- **max_iter** (*int, default 100*): Maximum number of Newton iterations
- **linear_solver** (*str, optional*): Linear solver type. If not specified, automatically selects based on backend: - GPU backend: &quot;cudss&quot; (cuDSS direct solver, requires CUDA) - CPU backend: &quot;cg&quot; (Conjugate Gradient, JAX-native) Manual options: &quot;cg&quot;, &quot;bicgstab&quot;, &quot;gmres&quot;, &quot;spsolve&quot;, &quot;cudss&quot;, &quot;lineax&quot;
- **preconditioner** (*callable, optional*): Preconditioner function for linear solver
- **use_jacobi_preconditioner** (*bool, default False*): Whether to use Jacobi (diagonal) preconditioner automatically
- **jacobi_shift** (*float, default 1e-12*): Regularization parameter for Jacobi preconditioner
- **linear_solver_tol** (*float, default 1e-10*): Tolerance for linear solver
- **linear_solver_atol** (*float, default 1e-10*): Absolute tolerance for linear solver
- **linear_solver_maxiter** (*int, default 10000*): Maximum iterations for linear solver
- **linear_solver_x0_fn** (*callable, optional*): Custom function to compute initial guess: f(current_sol) -&gt; x0
- **cudss_options** (*CUDSSOptions, optional*): CuDSS-specific options (mtype_id, mview_id, device_id)
- **line_search_max_backtracks** (*int, default 30*): Maximum number of backtracking steps in Armijo line search
- **line_search_c1** (*float, default 1e-4*): Armijo constant for sufficient decrease condition
- **line_search_rho** (*float, default 0.5*): Backtracking factor for line search (alpha *= rho each iteration)


#### linear\_solver

Auto-detected based on backend if not specified

#### linear\_solver\_x0\_fn

Function to compute initial guess: f(current_sol) -&gt; x0

#### cudss\_options

Will be set to default in __post_init__

#### \_\_post\_init\_\_

```python
def __post_init__()
```

Auto-detect backend and set appropriate defaults.

## DirectSolverOptions Objects

```python
@dataclass(frozen=True)
class DirectSolverOptions(AbstractSolverOptions)
```

Configuration for direct linear solvers.

Parameters
----------
- **solver** (*str, default &quot;auto&quot;*): Direct solver algorithm: - &quot;auto&quot;: Automatically selected based on backend and matrix property   (CUDA -&gt; cudss, CPU -&gt; spsolve). Resolved at create_solver time. - &quot;cudss&quot;: NVIDIA cuDSS direct solver (GPU only) - &quot;spsolve&quot;: JAX experimental sparse solve (CPU only) - &quot;cholesky&quot;: Cholesky decomposition via lineax (SPD matrices) - &quot;lu&quot;: LU decomposition via lineax (general matrices) - &quot;qr&quot;: QR decomposition via lineax (general/rectangular matrices)
- **cudss_options** (*CUDSSOptions, optional*): cuDSS-specific configuration. Only used when solver=&quot;cudss&quot;. Auto-configured when solver=&quot;auto&quot; and backend is CUDA.


#### resolve\_direct\_solver

```python
def resolve_direct_solver(options: DirectSolverOptions,
                          matrix_property: MatrixProperty,
                          matrix_view=None) -> DirectSolverOptions
```

Resolve &quot;auto&quot; to a concrete direct solver based on backend and matrix property.

Parameters
----------
- **options** (*DirectSolverOptions*): Options with solver possibly set to &quot;auto&quot;.
- **matrix_property** (*MatrixProperty*): Detected matrix property (SPD, SYMMETRIC, GENERAL).
- **matrix_view** (*MatrixView, optional*): Problem&#x27;s matrix storage format.  When UPPER or LOWER, the cuDSS solver is automatically configured to match.


Returns
-------
DirectSolverOptions
    Options with solver resolved to a concrete algorithm.
    If solver != &quot;auto&quot;, returns the input unchanged.

## IterativeSolverOptions Objects

```python
@dataclass(frozen=True)
class IterativeSolverOptions(AbstractSolverOptions)
```

Configuration for iterative linear solvers (cg, bicgstab, gmres).

Parameters
----------
- **solver** (*str, default &quot;auto&quot;*): Iterative solver algorithm: - &quot;auto&quot;: Automatically selected based on matrix property.   SPD -&gt; cg, SYMMETRIC -&gt; bicgstab, GENERAL -&gt; gmres.   Resolved at create_solver time. - &quot;cg&quot;: Conjugate Gradient (for SPD matrices) - &quot;bicgstab&quot;: BiCGSTAB (for symmetric/general matrices) - &quot;gmres&quot;: GMRES (for general matrices)
- **tol** (*float, default 1e-10*): Relative tolerance for the iterative solver.
- **atol** (*float, default 1e-10*): Absolute tolerance for the iterative solver.
- **maxiter** (*int, default 10000*): Maximum number of iterations.
- **preconditioner** (*callable, optional*): Custom preconditioner function M(x) -&gt; y.
- **use_jacobi_preconditioner** (*bool, default False*): Whether to auto-create Jacobi (diagonal) preconditioner.
- **jacobi_shift** (*float, default 1e-12*): Regularization parameter for Jacobi preconditioner.
- **x0_fn** (*callable, optional*): Custom function to compute initial guess: f(current_sol) -&gt; x0.


#### resolve\_iterative\_solver

```python
def resolve_iterative_solver(
        options: IterativeSolverOptions,
        matrix_property: MatrixProperty) -> IterativeSolverOptions
```

Resolve &quot;auto&quot; to a concrete iterative solver based on matrix property.

Parameters
----------
- **options** (*IterativeSolverOptions*): Options with solver possibly set to &quot;auto&quot;.
- **matrix_property** (*MatrixProperty*): Detected matrix property (SPD, SYMMETRIC, GENERAL).


Returns
-------
IterativeSolverOptions
    Options with solver resolved to a concrete algorithm.
    If solver != &quot;auto&quot;, returns the input unchanged.

