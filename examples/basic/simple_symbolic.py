"""Minimal symbolic DSL example - Linear Elasticity"""
import sys
sys.path.insert(0, '/workspace')

import jax.numpy as np
from feax.experimental import SymbolicProblem, create_internal_vars_from_dict
from feax.experimental.symbolic import (TrialFunction, TestFunction, Constant,
                                         epsilon, tr, inner, Identity, dx)
from feax.mesh import box_mesh
from feax import DirichletBC, DirichletBCSpec, create_solver, SolverOptions, InternalVars
from feax.utils import zero_like_initial_guess

# Mesh
mesh = box_mesh(size=1.0, mesh_size=0.2, element_type='HEX8')

# Symbolic weak form
u = TrialFunction(vec=3, name='u')
v = TestFunction(vec=3, name='v')
E = Constant(name='E', vec=1)
nu = Constant(name='nu', vec=1)

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
sigma = 2 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(3)

F = inner(sigma, epsilon(v)) * dx

# Problem
problem = SymbolicProblem(F, mesh, dim=3, ele_type='HEX8')

# BCs: fixed left, stretch right
bc = DirichletBC.from_specs(problem, [
    DirichletBCSpec(lambda x: np.isclose(x[0], 0.0, atol=1e-5), 'all', 0.0),
    DirichletBCSpec(lambda x: np.isclose(x[0], 1.0, atol=1e-5), 0, 0.01),
])

# Material properties
E_array = InternalVars.create_cell_var(problem, 210e9)
nu_array = InternalVars.create_cell_var(problem, 0.3)
internal_vars = create_internal_vars_from_dict(problem, {'E': E_array, 'nu': nu_array})

# Solve
solver = create_solver(problem, bc, SolverOptions(linear_solver="cg"), iter_num=1)
sol = solver(internal_vars, zero_like_initial_guess(problem, bc))

print(f"Max displacement: {np.max(np.abs(sol)):.6e} m")
