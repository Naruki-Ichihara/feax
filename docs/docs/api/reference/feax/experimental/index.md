---
sidebar_label: Overview
title: feax.experimental - Experimental Features
---

# Experimental Features

⚠️ **Experimental** - API may change in future versions.

Experimental symbolic interface for defining physics using mathematical notation instead of low-level kernel code.

## Modules

### [symbolic](./symbolic.md)
Symbolic DSL inspired by UFL/FEniCS for defining physics. Create finite element problems using mathematical notation with automatic kernel generation.

## Features

- ✅ Single-variable problems (Poisson, elasticity, Neumann BCs)
- ✅ Multi-variable problems (coupled Poisson, mixed formulations)
- ✅ Source terms and body forces
- ✅ Volume and surface integrals
- ✅ Gradient-based and value-based weak forms

## Quick Start

```python
from feax.experimental import SymbolicProblem
from feax.experimental.symbolic import (
    TrialFunction, TestFunction, Constant,
    grad, epsilon, inner, Identity, dx
)

# Define symbolic variables
u = TrialFunction(vec=3, name='displacement')
v = TestFunction(vec=3, name='v')
E = Constant(name='E', vec=1)
nu = Constant(name='nu', vec=1)

# Define physics using mathematical notation
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
eps_u = epsilon(u)
sigma = 2 * mu * eps_u + lmbda * tr(eps_u) * Identity(3)

# Define weak form
F = inner(sigma, epsilon(v)) * dx

# Create problem (automatically generates kernels)
problem = SymbolicProblem(F, mesh, dim=3, ele_type='HEX8', gauss_order=2)
```

## Status

Recent updates (2025):
- Multi-variable support implemented and validated
- Fixed sign tracking in weak form parsing
- Fixed nested multiplication handling for source terms
- Validated against JAX-FEM (matches to machine precision)
