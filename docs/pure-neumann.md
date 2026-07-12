# Pure-Neumann Poisson (v1)

For a scalar linear Poisson problem with no Dirichlet conditions, declare
its constant nullspace explicitly and select CG:

```python
constraint = feax.NullspaceConstraint.constant_mean_zero()
solver = feax.create_solver(
    problem,
    empty_bc,
    linear=True,
    solver_options=feax.KrylovSolverOptions(solver="cg"),
    nullspace=constraint,
)
solution = solver(traced_params, initial_guess)
```

The returned solution satisfies the mass-lumped physical gauge
`m.T @ solution == 0` on every connected mesh component. A disconnected mesh
therefore has one independent constant null mode per component. FEAX assumes
the user supplied a compatible pure-Neumann load on every component and does
not validate or modify that load.

v1 is limited to one scalar field, empty Dirichlet conditions, symmetric
linear operators, and matrix-free CG. Direct solvers, AMG, nonlinear solves,
reduced/prolongation systems, and general user-provided nullspace bases are
not supported.
