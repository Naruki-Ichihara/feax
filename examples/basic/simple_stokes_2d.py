"""Minimal symbolic DSL example - 2D Stokes flow in channel"""
import sys
sys.path.insert(0, '/workspace')

import jax.numpy as np
from feax.experimental import SymbolicProblem, create_internal_vars_from_dict
from feax.experimental.symbolic import (TrialFunction, TestFunction, Constant,
                                         epsilon, div, inner, sigma, dx)
from feax.mesh import rectangle_mesh
from feax import DirichletBC, DirichletBCSpec, create_solver, SolverOptions, InternalVars
from feax.utils import zero_like_initial_guess, save_sol

# 2D channel mesh
mesh = rectangle_mesh(Nx=32, Ny=16, domain_x=4.0, domain_y=1.0)
print(f"Mesh: {mesh.cells.shape[0]} cells, {mesh.points.shape[0]} nodes")

# Symbolic weak form (2D Stokes)
u = TrialFunction(vec=2, name='velocity', index=0)
p = TrialFunction(vec=1, name='pressure', index=1)
v = TestFunction(vec=2, name='v', index=0)
q = TestFunction(vec=1, name='q', index=1)
mu = Constant(name='viscosity', vec=1)

# Stokes: -div(sigma) = 0, div(u) = 0
# sigma = 2*mu*epsilon(u) - p*I
# Weak form: ∫ sigma:grad(v) dx = 0, ∫ q*div(u) dx = 0
F = inner(2 * mu * epsilon(u), epsilon(v)) * dx - p * div(v) * dx + q * div(u) * dx

# Problem
problem = SymbolicProblem(F, mesh, dim=2, ele_type='QUAD4')

# BCs: parabolic inlet, no-slip walls
def inlet(x): return np.isclose(x[0], 0.0, atol=1e-5)
def walls(x): return np.isclose(x[1], 0.0, atol=1e-5) | np.isclose(x[1], 1.0, atol=1e-5)

bc = DirichletBC.from_specs(problem, [
    DirichletBCSpec(inlet, 0, lambda x: 4*x[1]*(1-x[1])),  # Parabolic inlet u_x
    DirichletBCSpec(inlet, 1, 0.0),                        # Zero inlet u_y
    DirichletBCSpec(walls, 'all', 0.0),                    # No-slip walls
])

# Material
mu_array = InternalVars.create_cell_var(problem, 0.01)
internal_vars = create_internal_vars_from_dict(problem, {'viscosity': mu_array})

# Solve
solver = create_solver(problem, bc, SolverOptions(linear_solver="bicgstab", tol=1e-6), iter_num=1)
sol = solver(internal_vars, zero_like_initial_guess(problem, bc))

sol_list = problem.unflatten_fn_sol_list(sol)
velocity = sol_list[0]
pressure = sol_list[1]

print(f"Velocity: [{np.min(velocity):.3f}, {np.max(velocity):.3f}]")
print(f"Pressure: [{np.min(pressure):.3f}, {np.max(pressure):.3f}]")

save_sol(mesh, "stokes_2d.vtu",
         point_infos=[("velocity", velocity), ("pressure", pressure)])
print("Saved to stokes_2d.vtu")
