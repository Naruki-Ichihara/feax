---
sidebar_label: mma
title: feax.gene.mma
---

Copied and modified from https://github.com/UW-ERSL/AuTO
Under GNU General Public License v3.0

Original copy from https://github.com/arjendeetman/GCMMA-MMA-Python/blob/master/Code/MMA.py

Improvement is made to avoid N^2 memory operation so that the MMA solver is more scalable.

Note: This MMA optimizer is NOT JIT-compiled and this is intentional.
The optimizer uses control flow (while loops, conditionals) that are
difficult to JIT compile. JAX arrays are used for GPU compatibility
and automatic differentiation, but the MMA algorithm itself runs eagerly.

#### compute\_filter\_kd\_tree

```python
def compute_filter_kd_tree(fe)
```

This function is created by Tianju. Not from the original code.
We use k-d tree algorithm to compute the filter.

#### subsolv

```python
def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
```

Solve MMA subproblem using primal-dual interior point method.

Note: This function is NOT JIT-compiled due to dynamic control flow
(nested while loops with data-dependent iteration counts).
This is acceptable as the MMA optimizer is called once per
optimization iteration, not in performance-critical inner loops.

#### optimize

```python
def optimize(fe, rho_ini, optimizationParams, objectiveHandle, consHandle,
             numConstraints)
```

Performs topology optimization using the Method of Moving Asymptotes (MMA).

Parameters
----------
- **fe** (*FiniteElement*): Finite element object.
- **rho_ini** (*NumpyArray*): Initial density distribution. Shape is (num_rho_vars, 1).
- **optimizationParams** (*dict*): Dictionary containing optimization parameters:
- **objectiveHandle** (*callable*): Function that computes the objective value and its gradient. Signature: ``J, dJ = objectiveHandle(rho_physical)``
- **consHandle** (*callable*): Function that computes constraint values and their gradients. Signature: ``vc, dvc = consHandle(rho_physical, iter)``
- **numConstraints** (*int*): Number of constraints in the optimization problem.


Returns
-------
- **rho** (*NumpyArray*): Optimized density distribution after completing all iterations. Same shape as ``rho_ini``.


Notes
-----
TODO: Scale objective function value to be always within 1-100
(``8_).

