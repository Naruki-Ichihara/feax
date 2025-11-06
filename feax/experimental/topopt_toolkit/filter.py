"""Helmholtz filter for topology optimization using FEAX framework.

Simple implementation using InternalVars pattern.
"""

import jax
import jax.numpy as np
from typing import Callable, Optional

from feax.problem import Problem
from feax.internal_vars import InternalVars
from feax.assembler import get_res, get_J
from feax.solver import linear_solve, SolverOptions
from feax.DCboundary import DirichletBC


class HelmholtzProblem(Problem):
    """Helmholtz equation problem for design variable filtering."""
    
    def __post_init__(self):
        super().__post_init__()
        # Get radius from additional_info
        if self.additional_info:
            self.radius = self.additional_info[0]
        else:
            self.radius = 0.05
            
    def get_tensor_map(self):
        """Get the diffusion tensor mapping for the Helmholtz equation."""
        def diffusion(u_grad, design_variable):
            """Compute diffusion term r²∇u."""
            return self.radius**2 * u_grad
        return diffusion
        
    def get_mass_map(self):
        """Get the mass term mapping for the Helmholtz equation."""
        def mass_term(u, x, design_variable):
            """Compute mass term u - design_variable."""
            return u - design_variable
        return mass_term


def create_helmholtz_filter(base_problem: Problem, 
                           radius: float = 0.05) -> Callable:
    """Create a Helmholtz filter function for quadrature point design variables.
    
    Args:
        base_problem: Base FE problem defining mesh and element structure
        radius: Filter radius controlling smoothing length scale
        
    Returns:
        Filter function that applies Helmholtz filtering to design variables at quadrature points
    """
    
    # Create Helmholtz problem with vec=[1] for scalar density field
    helmholtz_problem = HelmholtzProblem(
        mesh=[base_problem.mesh] if not isinstance(base_problem.mesh, list) else [base_problem.mesh[0]],
        vec=[1],  # Scalar field for density
        dim=base_problem.dim,
        ele_type=[base_problem.ele_type] if not isinstance(base_problem.ele_type, list) else [base_problem.ele_type[0]],
        gauss_order=base_problem.gauss_order,
        additional_info=(radius,)
    )
    
    # Get problem dimensions
    cells = helmholtz_problem.fes[0].cells
    shape_vals = helmholtz_problem.fes[0].shape_vals
    num_nodes = helmholtz_problem.fes[0].num_total_nodes
    vec = helmholtz_problem.vec if isinstance(helmholtz_problem.vec, int) else helmholtz_problem.vec[0]
    
    @jax.jit
    def apply_filter(design_vars_quad: np.ndarray) -> np.ndarray:
        """Apply Helmholtz filter to design variables at quadrature points.
        
        Args:
            design_vars_quad: Design variable values at quadrature points (num_cells, num_quads)
                       
        Returns:
            Filtered design variables at quadrature points (num_cells, num_quads)
        """
        
        # Create internal variables with design field as source
        internal_vars = InternalVars(volume_vars=(design_vars_quad,))
        
        # Create residual and Jacobian functions with correct signatures
        def res_fn(sol, internal_vars_arg):
            sol_list = [sol.reshape(num_nodes, 1)]
            res_list = get_res(helmholtz_problem, sol_list, internal_vars_arg)
            return res_list[0].flatten()
        
        def jac_fn(sol, internal_vars_arg):
            sol_list = [sol.reshape(num_nodes, 1)]
            return get_J(helmholtz_problem, sol_list, internal_vars_arg)
        
        # Create empty boundary conditions (no constraints for filtering)
        bc = DirichletBC(
            bc_rows=np.array([], dtype=np.int32),
            bc_mask=np.zeros(num_nodes, dtype=bool),
            bc_vals=np.array([]),
            total_dofs=num_nodes
        )
        
        # Solve using linear solver (Helmholtz is linear)
        sol_init = np.zeros(num_nodes)
        solver_options = SolverOptions()
        sol = linear_solve(
            jac_fn, res_fn, sol_init, bc,
            solver_options=solver_options,
            internal_vars=internal_vars
        )
        
        # Handle solver return format (might be tuple or array)
        if isinstance(sol, tuple):
            filtered_nodal = sol[0].flatten()
        else:
            filtered_nodal = sol.flatten()
        
        # Convert filtered nodal values back to quadrature points
        filtered_cells = filtered_nodal[cells]  # (num_cells, num_nodes_per_cell)
        filtered_quads = np.einsum("qn,cn->cq", shape_vals, filtered_cells)  # (num_cells, num_quads)
        
        return filtered_quads
    
    return apply_filter

def create_helmholtz_transform(problem: Problem, key: str, radius: float = 0.05):
    """Create a simplified Helmholtz filtering transformation for optax.
    
    This creates an optax-compatible gradient transformation that applies 
    Helmholtz filtering to specific design variable updates with no conditional overhead.
    
    Args:
        problem: Base FE problem defining mesh and element structure
        key: Key name for the design variable to filter (e.g., 'rho', 'density')
        radius: Filter radius controlling smoothing length scale
        
    Returns:
        Optax gradient transformation that applies Helmholtz filtering
        
    Example:
        >>> filter_transform = create_helmholtz_transform(problem, 'rho', radius=0.05)
        >>> optimizer = optax.chain(
        ...     optax.adam(0.01),
        ...     filter_transform
        ... )
    """
    import optax
    
    filter_fn = create_helmholtz_filter(problem, radius)
    
    def init_fn(params):
        return optax.EmptyState()
    
    def update_fn(updates, state, params=None):
        """Apply Helmholtz filter to specified design variable updates."""
        # Direct key-based filtering without conditionals in compiled code
        if key in updates:
            # Create new updates dict with filtered value for the specific key
            filtered_updates = {**updates, key: filter_fn(updates[key])}
        else:
            # Key not present, return updates unchanged
            filtered_updates = updates
        
        return filtered_updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)

def create_box_projection_transform(key: str, lower: float = 0.0, upper: float = 1.0):
    """Create a box projection transformation for optax.
    
    This creates an optax-compatible gradient transformation that modifies updates
    to ensure parameters stay within specified bounds after the update is applied.
    
    The transform clips the update such that param + update stays within [lower, upper].
    
    Args:
        key: Key name for the design variable to project (e.g., 'rho', 'density')
        lower: Lower bound for projection
        upper: Upper bound for projection
        
    Returns:
        Optax gradient transformation that clips updates to maintain bounds
        
    Example:
        >>> box_transform = create_box_projection_transform('rho', lower=0.0, upper=1.0)
        >>> optimizer = optax.chain(
        ...     optax.adam(0.01),
        ...     create_helmholtz_transform(problem, 'rho', radius=0.05),
        ...     box_transform
        ... )
    """
    import optax
    
    def init_fn(params):
        return optax.EmptyState()
    
    def update_fn(updates, state, params=None):
        """Clip updates to ensure parameters stay within bounds."""
        if key in updates and params is not None and key in params:
            current_param = params[key]
            update = updates[key]
            
            # Clip the update to ensure param + update stays within bounds
            # If param + update < lower, then update = lower - param
            # If param + update > upper, then update = upper - param
            new_param = current_param + update
            clipped_param = np.clip(new_param, lower, upper)
            clipped_update = clipped_param - current_param
            
            # Create new updates dict with clipped update for the specific key
            clipped_updates = {**updates, key: clipped_update}
        else:
            # Key not present, return updates unchanged
            clipped_updates = updates
        
        return clipped_updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)


def create_sigmoid_transform(key: str, scale: float = 5.0):
    """Create a sigmoid transformation for design variables.
    
    This transformation maintains an unconstrained variable internally and 
    applies sigmoid to map it to [0, 1]. The gradients are automatically
    adjusted through the chain rule.
    
    The sigmoid function used is: rho = 1 / (1 + exp(-scale * x))
    where x is the unconstrained variable.
    
    Args:
        key: Key name for the design variable (e.g., 'rho')
        scale: Scaling factor for sigmoid steepness (higher = steeper transition)
        
    Returns:
        Optax gradient transformation that handles sigmoid reparameterization
        
    Example:
        >>> sigmoid_transform = create_sigmoid_transform('rho', scale=5.0)
        >>> optimizer = optax.chain(
        ...     optax.adam(0.01),
        ...     mdmm.optax_prepare_update(),
        ...     sigmoid_transform,
        ...     filter_transform
        ... )
    """
    import optax
    
    def init_fn(params):
        return optax.EmptyState()
    
    def update_fn(updates, state, params=None):
        """Transform gradients for sigmoid-parameterized variables."""
        if key in updates and params is not None and key in params:
            # Get current sigmoid-transformed parameter
            rho = params[key]
            
            # Compute gradient scaling factor: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x)) * scale
            # This adjusts the gradient w.r.t. the underlying unconstrained variable
            sigmoid_grad = rho * (1.0 - rho) * scale
            
            # Scale the gradients appropriately
            scaled_update = updates[key] * sigmoid_grad
            
            # Create new updates dict with scaled update
            transformed_updates = {**updates, key: scaled_update}
        else:
            transformed_updates = updates
        
        return transformed_updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)