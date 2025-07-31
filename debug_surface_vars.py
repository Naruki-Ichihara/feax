"""
Debug script to understand internal_vars surface structure
"""

import jax.numpy as np
from feax import Problem, InternalVars
from feax import Mesh, DirichletBC
from feax.mesh import box_mesh_gmsh

# Simple problem setup
E = 70e3
nu = 0.3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, E_quad):
            mu = E_quad / (2. * (1. + nu))
            lmbda = E_quad * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

# Create mesh and problem
meshio_mesh = box_mesh_gmsh(2, 2, 2, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

def zero_disp(point):
    return 0.0

# Boundary conditions
dirichlet_bc_info = [[left] * 3, [0, 1, 2], 
                     [zero_disp, zero_disp, zero_disp]]

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

print("=== Problem Analysis ===")
print(f"Number of boundaries: {len(problem.boundary_inds_list)}")
print(f"Boundary indices list: {[len(b) for b in problem.boundary_inds_list]} (showing lengths)")

# Test different InternalVars configurations
print("\n=== Testing InternalVars configurations ===")

# Test 1: Empty surface vars
try:
    E_array = InternalVars.create_uniform_volume_var(problem, E)
    internal_vars1 = InternalVars(
        volume_vars=(E_array,),
        surface_vars=()  # Empty surface vars
    )
    print(f"Empty surface_vars: {internal_vars1.surface_vars}")
    print(f"Length: {len(internal_vars1.surface_vars)}")
except Exception as e:
    print(f"Empty surface vars failed: {e}")

# Test 2: Single surface var (what's probably intended)
try:
    E_array = InternalVars.create_uniform_volume_var(problem, E)
    traction_array = InternalVars.create_uniform_surface_var(problem, 1.0)
    internal_vars2 = InternalVars(
        volume_vars=(E_array,),
        surface_vars=[(traction_array,)]  # Single tuple in list
    )
    print(f"Single surface_vars: {internal_vars2.surface_vars}")
    print(f"Length: {len(internal_vars2.surface_vars)}")
    print(f"First element: {internal_vars2.surface_vars[0]}")
    print(f"Length of first element: {len(internal_vars2.surface_vars[0])}")
except Exception as e:
    print(f"Single surface vars failed: {e}")

# Test 3: What might be needed for multiple boundaries
try:
    E_array = InternalVars.create_uniform_volume_var(problem, E)
    traction_array = InternalVars.create_uniform_surface_var(problem, 1.0)
    
    # Create surface vars for each boundary
    surface_vars_list = []
    for i in range(len(problem.boundary_inds_list)):
        surface_vars_list.append((traction_array,))
    
    internal_vars3 = InternalVars(
        volume_vars=(E_array,),
        surface_vars=surface_vars_list
    )
    print(f"Multiple surface_vars: {internal_vars3.surface_vars}")
    print(f"Length: {len(internal_vars3.surface_vars)}")
    for i, sv in enumerate(internal_vars3.surface_vars):
        print(f"  Boundary {i}: {len(sv)} variables")
except Exception as e:
    print(f"Multiple surface vars failed: {e}")

print("\n=== Recommended Fix ===")
print("The issue occurs when:")
print("1. problem.boundary_inds_list has multiple boundaries")
print("2. internal_vars.surface_vars doesn't have enough elements for each boundary")
print("3. The code tries to access internal_vars_surfaces[i] where i >= len(surface_vars)")
print("\nSolution: Ensure surface_vars has one entry for each boundary, or handle empty case.")