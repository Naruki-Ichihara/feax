---
sidebar_label: Overview
title: feax.solvers - Solver Infrastructure
---

# Solver Infrastructure

Low-level solver implementations and configuration for linear and nonlinear finite element problems.

## Modules

### [options](./options.md)
Solver configuration classes: `DirectSolverOptions`, `IterativeSolverOptions`, `NewtonOptions`. Includes backend detection (`detect_backend()`) and automatic solver resolution.

### [linear](./linear.md)
Linear solver creation and adjoint-based differentiation. `create_linear_solver()` builds JIT-compiled solvers with automatic backend selection (CUDSS on GPU, SciPy on CPU).

### [newton](./newton.md)
Newton-Raphson nonlinear solver with Armijo line search. `create_newton_solver()` provides configurable nonlinear solving with multiple line search strategies.

### [common](./common.md)
Shared utilities: preconditioners (`create_jacobi_preconditioner()`), initial guess creation (`create_x0()`), convergence checking, and direct/iterative solve function factories.

### [reduced](./reduced.md)
Reduced-space solver for problems with periodic boundary conditions via prolongation matrices.
