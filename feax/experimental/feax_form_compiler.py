"""
FEAX Form Compiler: Compiles symbolic weak forms to JAX kernels.

This module implements a form compiler that translates mathematical weak form
expressions (similar to UFL/FEniCS) into efficient JAX kernel functions. It
uses runtime evaluation of the symbolic expression tree to generate the finite
element residual and Jacobian calculations.

Architecture:
    Symbolic Expression → Expression Tree → RuntimeEvaluator → JAX Kernel

Key Components:
    - SymbolicProblem: Main problem class that manages compilation
    - RuntimeEvaluator: Runtime evaluator for symbolic expressions
    - extract_forms(): Extracts integrals from expression trees
"""

import jax
import jax.numpy as np
import jax.flatten_util
from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from feax.internal_vars import InternalVars

from feax.problem import Problem
from feax.mesh import Mesh
from .symbolic import (
    Expr, Integral, IntegralSum,
    TrialFunction, TestFunction, Constant, ScalarConstant, Identity,
    Grad, Div, Sym, Transpose, Trace,
    Inner, Dot, Outer,
    Add, Sub, Mul, Division,
    TensorRank
)


def extract_forms(expr: Expr) -> List[Integral]:
    """Extract all integral forms from symbolic expression, tracking signs."""
    integrals = []

    def collect(e, sign=1):
        if isinstance(e, Integral):
            # If sign is negative, negate the integrand
            if sign < 0:
                negated_integrand = Mul(ScalarConstant(-1.0), e.integrand)
                negated_integral = Integral(negated_integrand, e.measure, e.boundary_id)
                integrals.append(negated_integral)
            else:
                integrals.append(e)
        elif isinstance(e, IntegralSum):
            for integral in e.integrals:
                collect(integral, sign)
        elif isinstance(e, Add):
            collect(e.left, sign)
            collect(e.right, sign)
        elif isinstance(e, Sub):
            collect(e.left, sign)
            collect(e.right, -sign)  # Flip sign for subtraction!

    collect(expr)
    return integrals


def collect_symbols(expr: Expr):
    """Collect all symbolic functions from expression."""
    trial_functions = []
    test_functions = []
    constants = {}

    def visit(e):
        if isinstance(e, TrialFunction):
            if e not in trial_functions:
                trial_functions.append(e)
        elif isinstance(e, TestFunction):
            if e not in test_functions:
                test_functions.append(e)
        elif isinstance(e, Constant):
            if e.name not in constants:
                constants[e.name] = e
        elif hasattr(e, '__dict__'):
            for attr_val in e.__dict__.values():
                if isinstance(attr_val, Expr):
                    visit(attr_val)
                elif isinstance(attr_val, list):
                    for item in attr_val:
                        if isinstance(item, Expr):
                            visit(item)

    visit(expr)
    trial_functions.sort(key=lambda x: x.index)
    test_functions.sort(key=lambda x: x.index)

    return trial_functions, test_functions, constants


class RuntimeEvaluator:
    """Evaluates symbolic expressions at runtime on JAX arrays."""

    def __init__(self, problem, u_list, u_grads_list, const_dict):
        """
        Parameters
        ----------
        problem : SymbolicProblem
            The problem instance
        u_list : list of arrays
            Trial function values at quad points [(num_quads, vec), ...]
        u_grads_list : list of arrays
            Trial function gradients at quad points [(num_quads, vec, dim), ...]
        const_dict : dict
            Constants interpolated to quad points {name: array}
        """
        self.problem = problem
        self.u_list = u_list
        self.u_grads_list = u_grads_list
        self.const_dict = const_dict
        self.dim = problem.dim

    def eval(self, expr: Expr) -> np.ndarray:
        """Evaluate expression and return array at quadrature points."""

        # Trial function - return value
        if isinstance(expr, TrialFunction):
            return self.u_list[expr.index]

        # Test function - shouldn't be evaluated directly
        elif isinstance(expr, TestFunction):
            raise ValueError("Cannot evaluate test function directly")

        # Constant
        elif isinstance(expr, Constant):
            return self.const_dict.get(expr.name, 0.0)

        # Scalar constant
        elif isinstance(expr, ScalarConstant):
            return expr.value

        # Identity tensor
        elif isinstance(expr, Identity):
            # Return identity matrix - will be broadcast to quad points as needed
            return np.eye(expr.dim)

        # Gradient
        elif isinstance(expr, Grad):
            if isinstance(expr.operand, TrialFunction):
                return self.u_grads_list[expr.operand.index]
            else:
                raise NotImplementedError(f"Gradient of {type(expr.operand).__name__}")

        # Divergence
        elif isinstance(expr, Div):
            # Check if operand is directly a TrialFunction
            if isinstance(expr.operand, TrialFunction):
                # div(u) where u is trial function - compute from gradient
                grad_u = self.u_grads_list[expr.operand.index]
                # Sum diagonal: grad_u[:, 0, 0] + grad_u[:, 1, 1] + ...
                if self.dim == 2:
                    return grad_u[:, 0, 0] + grad_u[:, 1, 1]
                else:  # 3D
                    return grad_u[:, 0, 0] + grad_u[:, 1, 1] + grad_u[:, 2, 2]
            elif isinstance(expr.operand, Grad) and isinstance(expr.operand.operand, TrialFunction):
                # div(grad(u)) - same as above
                grad_u = self.u_grads_list[expr.operand.operand.index]
                # Sum diagonal: grad_u[:, 0, 0] + grad_u[:, 1, 1] + ...
                if self.dim == 2:
                    return grad_u[:, 0, 0] + grad_u[:, 1, 1]
                else:  # 3D
                    return grad_u[:, 0, 0] + grad_u[:, 1, 1] + grad_u[:, 2, 2]
            else:
                # Generic divergence - evaluate operand first
                operand_val = self.eval(expr.operand)
                if operand_val.ndim == 3:  # (num_quads, vec, dim)
                    if self.dim == 2:
                        return operand_val[:, 0, 0] + operand_val[:, 1, 1]
                    else:
                        return operand_val[:, 0, 0] + operand_val[:, 1, 1] + operand_val[:, 2, 2]
                else:
                    raise ValueError(f"Cannot take divergence of shape {operand_val.shape}")

        # Symmetric part
        elif isinstance(expr, Sym):
            operand_val = self.eval(expr.operand)
            # (T + T^T) / 2
            return 0.5 * (operand_val + operand_val.transpose(0, 2, 1))

        # Transpose
        elif isinstance(expr, Transpose):
            operand_val = self.eval(expr.operand)
            return operand_val.transpose(0, 2, 1)

        # Trace
        elif isinstance(expr, Trace):
            operand_val = self.eval(expr.operand)
            if self.dim == 2:
                return operand_val[:, 0, 0] + operand_val[:, 1, 1]
            else:
                return operand_val[:, 0, 0] + operand_val[:, 1, 1] + operand_val[:, 2, 2]

        # Inner product
        elif isinstance(expr, Inner):
            left_val = self.eval(expr.left)
            right_val = self.eval(expr.right)

            if left_val.ndim == 2 and right_val.ndim == 2:
                # Vector · Vector: (num_quads, vec) · (num_quads, vec) -> (num_quads,)
                return np.sum(left_val * right_val, axis=-1)
            elif left_val.ndim == 3 and right_val.ndim == 3:
                # Tensor : Tensor: (num_quads, dim, dim) : (num_quads, dim, dim) -> (num_quads,)
                return np.sum(left_val * right_val, axis=(-2, -1))
            else:
                raise ValueError(f"Inner product between shapes {left_val.shape} and {right_val.shape}")

        # Dot product
        elif isinstance(expr, Dot):
            left_val = self.eval(expr.left)
            right_val = self.eval(expr.right)

            if left_val.ndim == 2 and right_val.ndim == 3:
                # Vector · Tensor: (u·∇)u for convection
                # (num_quads, vec) · (num_quads, vec, dim) -> (num_quads, vec)
                return np.einsum('qi,qij->qj', left_val, right_val)
            elif left_val.ndim == 2 and right_val.ndim == 2:
                # Vector · Vector
                return np.sum(left_val * right_val, axis=-1)
            else:
                raise ValueError(f"Dot product between shapes {left_val.shape} and {right_val.shape}")

        # Outer product
        elif isinstance(expr, Outer):
            left_val = self.eval(expr.left)
            right_val = self.eval(expr.right)
            return left_val[:, :, None] * right_val[:, None, :]

        # Addition
        elif isinstance(expr, Add):
            return self.eval(expr.left) + self.eval(expr.right)

        # Subtraction
        elif isinstance(expr, Sub):
            return self.eval(expr.left) - self.eval(expr.right)

        # Multiplication
        elif isinstance(expr, Mul):
            left_val = self.eval(expr.left)
            right_val = self.eval(expr.right)

            # Debug
            # if not isinstance(left_val, (int, float)) and not isinstance(right_val, (int, float)):
            #     jax.debug.print("Mul: left.shape={}, right.shape={}", left_val.shape, right_val.shape)

            # Handle scalar broadcasting
            if isinstance(left_val, (int, float)):
                return left_val * right_val
            elif isinstance(right_val, (int, float)):
                return left_val * right_val
            elif left_val.ndim == 1 and right_val.ndim >= 2:
                # Scalar at quad points * tensor
                # Need to broadcast: (num_quads,) * (...) -> (num_quads, ...)
                if right_val.shape[0] != left_val.shape[0]:
                    # right_val is constant tensor (e.g., Identity), broadcast left_val
                    # (num_quads,) * (d, d) -> (num_quads, d, d)
                    return left_val[:, None, None] * right_val
                else:
                    # right_val has batch dimension matching left_val
                    # (num_quads,) * (num_quads, d, d) -> (num_quads, d, d)
                    # Need to add dimensions to match right_val's rank
                    num_extra_dims = right_val.ndim - 1
                    shape_modifier = (slice(None),) + (None,) * num_extra_dims
                    return left_val[shape_modifier] * right_val
            elif right_val.ndim == 1 and left_val.ndim >= 2:
                # Tensor * scalar at quad points
                if left_val.shape[0] != right_val.shape[0]:
                    # left_val is constant tensor, broadcast right_val
                    return left_val * right_val[:, None, None]
                else:
                    # left_val has batch dimension matching right_val
                    # (num_quads, d, d) * (num_quads,) -> (num_quads, d, d)
                    num_extra_dims = left_val.ndim - 1
                    shape_modifier = (slice(None),) + (None,) * num_extra_dims
                    return left_val * right_val[shape_modifier]
            elif left_val.ndim == 2 and right_val.ndim == 2:
                # Both are 2D - check if one is a batch of scalars vs tensor
                # Case 1: (num_quads, num_nodes) * (num_quads, dim, dim) - shouldn't happen
                # Case 2: (num_quads, dim) * (num_quads, dim) - element-wise (shouldn't happen here)
                # Case 3: Matrix multiplication - delegate to JAX
                # For now, try direct multiplication and let JAX handle broadcasting
                return left_val * right_val
            else:
                return left_val * right_val

        # Division
        elif isinstance(expr, Division):
            left_val = self.eval(expr.left)
            right_val = self.eval(expr.right)

            # Debug
            # if not isinstance(left_val, (int, float)) and not isinstance(right_val, (int, float)):
            #     jax.debug.print("Div: left.shape={}, right.shape={}", left_val.shape, right_val.shape)

            # Handle division broadcasting (similar to multiplication)
            if isinstance(right_val, (int, float)):
                return left_val / right_val
            elif isinstance(left_val, (int, float)):
                return left_val / right_val
            elif left_val.ndim == 1 and right_val.ndim == 1:
                # Both 1D: element-wise division
                return left_val / right_val
            elif left_val.ndim >= 2 and right_val.ndim == 1:
                # Tensor / scalar at quad points
                return left_val / right_val[:, None, ...]
            elif left_val.ndim == 1 and right_val.ndim >= 2:
                # Scalar at quad points / tensor (unusual but handle it)
                return left_val[:, None, ...] / right_val
            else:
                return left_val / right_val

        else:
            raise NotImplementedError(f"Evaluation of {type(expr).__name__} not implemented")


def find_test_function(expr: Expr) -> Optional[TestFunction]:
    """Find the test function in an expression."""
    if isinstance(expr, TestFunction):
        return expr

    if hasattr(expr, '__dict__'):
        for attr_val in expr.__dict__.values():
            if isinstance(attr_val, Expr):
                result = find_test_function(attr_val)
                if result is not None:
                    return result
    return None


def has_trial_gradient(expr: Expr) -> bool:
    """Check if expression involves gradients of trial function."""
    if isinstance(expr, (Grad, Div)):
        # Check if operand is or contains a TrialFunction
        def find_trial(e):
            if isinstance(e, TrialFunction):
                return True
            if hasattr(e, '__dict__'):
                for v in e.__dict__.values():
                    if isinstance(v, Expr) and find_trial(v):
                        return True
            return False
        return find_trial(expr.operand)

    if hasattr(expr, '__dict__'):
        for v in expr.__dict__.values():
            if isinstance(v, Expr) and has_trial_gradient(v):
                return True
    return False


def is_gradient_based(expr: Expr) -> bool:
    """Check if expression involves gradients of TEST function.

    This determines whether to use gradient-based assembly (contracting with v_grads_JxW)
    or value-based assembly (contracting with shape_vals).

    Key: Only test function gradients matter for assembly type!
    """
    if isinstance(expr, Grad):
        if isinstance(expr.operand, TestFunction):
            return True
        if find_test_function(expr.operand):
            return True

    # Divergence involves gradient: div(v) = trace(grad(v))
    if isinstance(expr, Div):
        if isinstance(expr.operand, TestFunction):
            return True
        if find_test_function(expr.operand):
            return True

    if isinstance(expr, (Inner, Dot, Outer)):
        return is_gradient_based(expr.left) or is_gradient_based(expr.right)

    if isinstance(expr, (Add, Sub, Mul, Division)):
        left_grad = is_gradient_based(expr.left) if hasattr(expr, 'left') and isinstance(expr.left, Expr) else False
        right_grad = is_gradient_based(expr.right) if hasattr(expr, 'right') and isinstance(expr.right, Expr) else False
        return left_grad or right_grad

    if isinstance(expr, (Sym, Transpose)):
        return is_gradient_based(expr.operand)

    return False


def create_internal_vars_from_dict(problem: 'SymbolicProblem',
                                   volume_dict: Optional[Dict[str, np.ndarray]] = None,
                                   surface_dict: Optional[Dict[str, np.ndarray]] = None) -> 'InternalVars':
    """Create InternalVars from name-based dictionaries (recommended for symbolic problems).

    This helper converts name-based dictionaries to the correct tuple order
    required by InternalVars, matching the order of Constant() declarations
    in the symbolic weak form.

    Parameters
    ----------
    problem : SymbolicProblem
        The symbolic problem containing constant declarations
    volume_dict : dict, optional
        Dictionary mapping constant names to arrays: {'E': E_array, 'nu': nu_array}
    surface_dict : dict, optional
        Dictionary mapping surface constant names to arrays

    Returns
    -------
    InternalVars
        InternalVars with values in correct order

    Examples
    --------
    >>> # Instead of worrying about order:
    >>> internal_vars = InternalVars(volume_vars=(E_array, nu_array), surface_vars=())
    >>>
    >>> # Use name-based approach:
    >>> internal_vars = create_internal_vars_from_dict(
    ...     problem,
    ...     volume_dict={'E': E_array, 'nu': nu_array}
    ... )
    """
    from feax.internal_vars import InternalVars

    # Convert volume dict to ordered tuple
    volume_tuple = ()
    if volume_dict is not None:
        volume_tuple = tuple(
            volume_dict.get(name, np.zeros(1))  # Default to zero if not provided
            for name in problem.constants.keys()
        )

    # Convert surface dict to ordered tuple (if needed in future)
    surface_tuple = ()
    if surface_dict is not None:
        # For now, surface vars don't have named constants
        surface_tuple = tuple(surface_dict.values())

    return InternalVars(volume_vars=volume_tuple, surface_vars=surface_tuple)


class SymbolicProblem(Problem):
    """Finite element problem defined from symbolic weak form.

    Simplified implementation with direct runtime evaluation.

    Examples
    --------
    >>> from feax.symbolic import TrialFunction, TestFunction, Constant
    >>> from feax.symbolic import grad, inner, dx
    >>> from feax.symbolic_problem_simple import SymbolicProblem
    >>>
    >>> u = TrialFunction(vec=1, name='u')
    >>> v = TestFunction(vec=1, name='v')
    >>> f = Constant(name='source')
    >>>
    >>> F = inner(grad(u), grad(v)) * dx - f * v * dx
    >>>
    >>> problem = SymbolicProblem(weak_form=F, mesh=mesh, dim=3)
    """

    def __init__(
        self,
        weak_form: Expr,
        mesh: Union[Mesh, List[Mesh]],
        dim: int,
        ele_type: Union[str, List[str]] = 'HEX8',
        gauss_order: Optional[Union[int, List[int]]] = None,
        location_fns: Optional[List[callable]] = None,
        additional_info: tuple = ()
    ):
        """Initialize SymbolicProblem from weak form."""

        self.weak_form = weak_form
        self.trial_functions, self.test_functions, self.constants = collect_symbols(weak_form)
        self.integrals = extract_forms(weak_form)

        if not self.integrals:
            raise ValueError("Weak form must contain at least one integral")
        if not self.trial_functions:
            raise ValueError("No trial functions found in weak form")

        # Build vec list - but pass single values for single-variable problems
        if len(self.trial_functions) == 1:
            vec_param = self.trial_functions[0].vec
            mesh_param = mesh
            ele_type_param = ele_type
        else:
            vec_param = [tf.vec for tf in self.trial_functions]
            mesh_param = mesh if isinstance(mesh, list) else [mesh] * len(self.trial_functions)
            ele_type_param = ele_type if isinstance(ele_type, list) else [ele_type] * len(self.trial_functions)

        # Initialize Problem base class
        super().__init__(
            mesh=mesh_param,
            vec=vec_param,
            dim=dim,
            ele_type=ele_type_param,
            gauss_order=gauss_order,
            location_fns=location_fns,
            additional_info=additional_info
        )

        # Separate volume and surface integrals
        self.volume_integrals = [itg for itg in self.integrals if itg.measure == 'dx']
        self.surface_integrals = [itg for itg in self.integrals if itg.measure == 'ds']

    def get_universal_kernel(self):
        """Generate universal kernel for volume integrals."""
        if not self.volume_integrals:
            return None

        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW,
                            cell_v_grads_JxW, *cell_internal_vars):

            # Extract solution variables
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)

            # Compute gradients for each trial function
            u_list = []
            u_grads_list = []

            for i, tf in enumerate(self.trial_functions):
                cell_sol = cell_sol_list[i]

                # Get shape functions for this variable
                start_idx = self.num_nodes_cumsum[i]
                end_idx = self.num_nodes_cumsum[i + 1]
                shape_grads = cell_shape_grads[:, start_idx:end_idx, :]
                shape_vals = self.fes[i].shape_vals

                # Compute gradient: (num_quads, vec, dim)
                u_grad = cell_sol[None, :, :, None] * shape_grads[:, :, None, :]
                u_grad = np.sum(u_grad, axis=1)
                u_grads_list.append(u_grad)

                # Compute values at quad points: (num_quads, vec)
                u = np.sum(cell_sol[None, :, :] * shape_vals[:, :, None], axis=1)
                if tf.vec == 1:
                    u = u[:, 0]  # Scalar
                u_list.append(u)

            # Interpolate constants to quad points
            # For multi-var: use first FE's quad count (assumes same quadrature for all)
            num_quads = self.fes[0].num_quads
            shape_vals = self.fes[0].shape_vals
            const_dict = {}

            # Interpolate constants to quad points
            # Note: Use create_internal_vars_from_dict() for name-based approach
            for i, (name, _) in enumerate(self.constants.items()):
                if i < len(cell_internal_vars):
                    var = cell_internal_vars[i]
                    # Interpolate to quad points (same logic as laplace kernel)
                    if var.ndim == 0:
                        # Scalar (cell-based): broadcast to all quad points
                        var_quad = np.full(num_quads, var)
                    elif var.ndim == 1:
                        # Check if node-based by comparing with any FE's num_nodes
                        is_node_based = any(var.shape[0] == fe.num_nodes for fe in self.fes)

                        if is_node_based:
                            # Node-based: interpolate using shape functions
                            # For multi-var, assumes variables share same nodes (common case)
                            var_quad = np.dot(shape_vals, var)  # (num_quads, num_nodes) @ (num_nodes,) -> (num_quads,)
                        elif var.shape[0] == 1:
                            # Cell-based (single element): broadcast to all quad points
                            var_quad = np.full(num_quads, var[0])
                        elif var.shape[0] == num_quads:
                            # Quad-based (legacy): already has quad point values
                            var_quad = var
                        else:
                            # Unknown, assume cell-based and broadcast first element
                            var_quad = np.full(num_quads, var[0])
                    elif var.ndim == 2:
                        # Vector constant: shape (num_nodes_per_elem, vec) or (num_quads, vec)
                        # After assembler slicing, node-based becomes (num_nodes_per_elem, vec)

                        # Check if this matches num_nodes for any FE (node-based)
                        is_node_based = any(var.shape[0] == fe.num_nodes for fe in self.fes)

                        if is_node_based:
                            # Node-based vector: interpolate using shape functions
                            # (num_quads, num_nodes) @ (num_nodes, vec) -> (num_quads, vec)
                            var_quad = np.dot(shape_vals, var)
                        elif var.shape[0] == num_quads:
                            # Quad-based vector: already at quad points
                            var_quad = var
                        else:
                            # Unknown format, pass through
                            var_quad = var
                    else:
                        # Unknown format, pass through
                        var_quad = var
                    const_dict[name] = var_quad
                else:
                    const_dict[name] = np.zeros(num_quads)

            # Create evaluator
            evaluator = RuntimeEvaluator(self, u_list, u_grads_list, const_dict)

            # Initialize weak form values
            weak_form_vals = [np.zeros((self.fes[i].num_nodes, self.fes[i].vec))
                             for i in range(self.num_vars)]

            # Evaluate each integral
            for integral in self.volume_integrals:
                test_func = find_test_function(integral.integrand)
                if test_func is None:
                    continue

                test_idx = test_func.index

                # Get test function data
                start_idx = self.num_nodes_cumsum[test_idx]
                end_idx = self.num_nodes_cumsum[test_idx + 1]
                v_grads_JxW = cell_v_grads_JxW[:, start_idx:end_idx, :, :]
                JxW = cell_JxW[test_idx]
                shape_vals_test = self.fes[test_idx].shape_vals

                # Check if this is gradient-based or value-based
                grad_based = is_gradient_based(integral.integrand)

                if grad_based:
                    # Gradient-based: inner(something, grad(v))
                    # Evaluate the "something" part (without test function)
                    # For inner(grad(u), grad(v)), we need grad(u)
                    integrand_val = self._eval_gradient_integrand(integral.integrand, evaluator, test_func)

                    # Debug output
                    # jax.debug.print("integrand_val shape: {}", integrand_val.shape)
                    # jax.debug.print("v_grads_JxW shape: {}", v_grads_JxW.shape)
                    # jax.debug.print("integrand_val sample: {}", integrand_val[0, :, :])

                    # Assembly: sum(integrand[:, None, :, :] * v_grads_JxW, axis=(0, -1))
                    # integrand_val should be (num_quads, vec, dim) or special tuple for div(test)

                    # Check if this is a divergence of test function case
                    if isinstance(integrand_val, tuple) and integrand_val[0] == 'div_test':
                        # Special case: p * div(v) - contract with trace of grad(v)
                        p_vals = integrand_val[1]  # shape (num_quads,)
                        # Compute trace: sum over diagonal of grad(v)
                        # v_grads_JxW has shape (num_quads, num_nodes, vec, dim)
                        trace_grad_v = sum(v_grads_JxW[:, :, i, i] for i in range(self.dim))  # (num_quads, num_nodes)
                        val = np.sum(p_vals[:, None] * trace_grad_v, axis=0)  # (num_nodes,)
                        val = val[:, None]  # (num_nodes, 1) for scalar field
                    elif integrand_val.ndim == 3:
                        val = np.sum(integrand_val[:, None, :, :] * v_grads_JxW, axis=(0, -1))
                    elif integrand_val.ndim == 2:
                        # Vector case
                        val = np.sum(integrand_val[:, None, :] * v_grads_JxW[:, :, 0, :], axis=(0, -1))
                    else:
                        # Scalar case
                        val = np.sum(integrand_val[:, None, None] * v_grads_JxW[:, :, 0, :], axis=(0, -1))
                        val = val[:, None]

                    weak_form_vals[test_idx] += val
                else:
                    # Value-based: f * v
                    integrand_val = self._eval_value_integrand(integral.integrand, evaluator, test_func)

                    # Handle Python scalars/floats
                    if isinstance(integrand_val, (int, float)):
                        integrand_val = np.array(integrand_val)

                    # Assembly: sum(integrand[:, None, :] * shape_vals[:, :, None] * JxW[:, None, None], axis=0)
                    if hasattr(integrand_val, 'ndim') and integrand_val.ndim == 2:
                        val = np.sum(integrand_val[:, None, :] * shape_vals_test[:, :, None] * JxW[:, None, None], axis=0)
                    elif hasattr(integrand_val, 'ndim') and integrand_val.ndim == 1:
                        # Scalar at quad points
                        val = np.sum(integrand_val[:, None] * shape_vals_test * JxW[:, None], axis=0)
                        val = val[:, None]
                    else:
                        # Scalar constant
                        val = np.sum(integrand_val * shape_vals_test * JxW[:, None], axis=0)
                        val = val[:, None]

                    weak_form_vals[test_idx] += val

            return jax.flatten_util.ravel_pytree(weak_form_vals)[0]

        return universal_kernel

    def _eval_gradient_integrand(self, expr: Expr, evaluator: RuntimeEvaluator, test_func: TestFunction) -> np.ndarray:
        """Evaluate integrand for gradient-based assembly (removes test function grad)."""
        # For inner(A, grad(v)), return A
        # For inner(grad(u), grad(v)), return grad(u)

        if isinstance(expr, Inner):
            left_has_test = find_test_function(expr.left) is not None
            right_has_test = find_test_function(expr.right) is not None

            if right_has_test and not left_has_test:
                return evaluator.eval(expr.left)
            elif left_has_test and not right_has_test:
                return evaluator.eval(expr.right)
            else:
                raise ValueError("Cannot determine which side has test function")

        elif isinstance(expr, Mul):
            left_has_test = find_test_function(expr.left) is not None
            right_has_test = find_test_function(expr.right) is not None

            # Special case 1: p * div(v) where p is scalar, div(v) involves test function gradient
            if isinstance(expr.right, Div) and isinstance(expr.right.operand, TestFunction):
                # Return p with marker that we need divergence contraction
                return ('div_test', evaluator.eval(expr.left))
            elif isinstance(expr.left, Div) and isinstance(expr.left.operand, TestFunction):
                return ('div_test', evaluator.eval(expr.right))

            elif right_has_test and not left_has_test:
                # Check for nested Mul with div(v): scalar * (p * div(v))
                if isinstance(expr.right, Mul):
                    inner_val = self._eval_gradient_integrand(expr.right, evaluator, test_func)
                    if isinstance(inner_val, tuple) and inner_val[0] == 'div_test':
                        # Multiply the pressure by the scalar
                        return ('div_test', evaluator.eval(expr.left) * inner_val[1])
                    else:
                        return evaluator.eval(expr.left) * inner_val
                else:
                    return evaluator.eval(expr.left)
            elif left_has_test and not right_has_test:
                # Symmetric case
                if isinstance(expr.left, Mul):
                    inner_val = self._eval_gradient_integrand(expr.left, evaluator, test_func)
                    if isinstance(inner_val, tuple) and inner_val[0] == 'div_test':
                        return ('div_test', inner_val[1] * evaluator.eval(expr.right))
                    else:
                        return inner_val * evaluator.eval(expr.right)
                else:
                    return evaluator.eval(expr.right)
            else:
                # Both sides might have structure - evaluate recursively
                return evaluator.eval(expr.left)

        elif isinstance(expr, (Add, Sub)):
            # Handle addition/subtraction in trial function part (e.g., sigma = 2*mu*eps(u) - p*I)
            # Just evaluate both sides and combine
            left_val = self._eval_gradient_integrand(expr.left, evaluator, test_func)
            right_val = self._eval_gradient_integrand(expr.right, evaluator, test_func)
            return left_val + right_val if isinstance(expr, Add) else left_val - right_val

        else:
            # Try to evaluate without test function
            return evaluator.eval(expr)

    def _eval_value_integrand(self, expr: Expr, evaluator: RuntimeEvaluator, test_func: TestFunction) -> np.ndarray:
        """Evaluate integrand for value-based assembly (removes test function value)."""
        # For f * v, return f
        # For inner(f, v), return f

        if isinstance(expr, Mul):
            left_has_test = find_test_function(expr.left) is not None
            right_has_test = find_test_function(expr.right) is not None

            if right_has_test and not left_has_test:
                # Left side doesn't have test function - evaluate it
                # But right side might be nested (e.g., -1.0 * (f * v))
                # So we need to recursively extract the value part from right
                if isinstance(expr.right, Mul):
                    # Nested multiplication: scalar * (f * v)
                    # Extract f from inner Mul, then multiply by scalar
                    inner_val = self._eval_value_integrand(expr.right, evaluator, test_func)
                    return evaluator.eval(expr.left) * inner_val
                else:
                    return evaluator.eval(expr.left)
            elif left_has_test and not right_has_test:
                # Symmetric case
                if isinstance(expr.left, Mul):
                    inner_val = self._eval_value_integrand(expr.left, evaluator, test_func)
                    return inner_val * evaluator.eval(expr.right)
                else:
                    return evaluator.eval(expr.right)
            else:
                # Both or neither have test function
                return evaluator.eval(expr.left)

        elif isinstance(expr, Inner):
            # Handle inner(f, v) or inner(v, f)
            left_has_test = find_test_function(expr.left) is not None
            right_has_test = find_test_function(expr.right) is not None

            if right_has_test and not left_has_test:
                return evaluator.eval(expr.left)
            elif left_has_test and not right_has_test:
                return evaluator.eval(expr.right)
            else:
                # Both or neither have test function - shouldn't happen for value-based assembly
                return evaluator.eval(expr.left)

        elif isinstance(expr, (Add, Sub)):
            left_val = self._eval_value_integrand(expr.left, evaluator, test_func) if not is_gradient_based(expr.left) else 0
            right_val = self._eval_value_integrand(expr.right, evaluator, test_func) if not is_gradient_based(expr.right) else 0
            return left_val + right_val if isinstance(expr, Add) else left_val - right_val

        else:
            return evaluator.eval(expr)

    def get_universal_kernels_surface(self):
        """Generate universal kernels for surface integrals."""
        if not self.surface_integrals:
            return []

        # Group integrals by boundary_id
        boundary_integrals = {}
        for integral in self.surface_integrals:
            bid = integral.boundary_id if integral.boundary_id is not None else 0
            if bid not in boundary_integrals:
                boundary_integrals[bid] = []
            boundary_integrals[bid].append(integral)

        # Generate kernel for each boundary
        kernels = []
        for bid in sorted(boundary_integrals.keys()):
            bnd_integrals = boundary_integrals[bid]
            kernel = self._generate_surface_kernel(bnd_integrals)
            kernels.append(kernel)

        return kernels

    def _generate_surface_kernel(self, integrals: List[Integral]):
        """Generate universal kernel for surface integrals on one boundary."""

        def surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                          face_nanson_scale, *cell_internal_vars_surface):
            """Generated surface kernel."""

            # Extract solution variables
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)

            # Interpolate solution to face quadrature points
            u_list = []
            for i, tf in enumerate(self.trial_functions):
                cell_sol = cell_sol_list[i]
                start_idx = self.num_nodes_cumsum[i]
                end_idx = self.num_nodes_cumsum[i + 1]
                face_shape_vals_var = face_shape_vals[:, start_idx:end_idx]

                # Interpolate: (num_face_quads, vec)
                u = np.sum(cell_sol[None, :, :] * face_shape_vals_var[:, :, None], axis=1)
                if tf.vec == 1:
                    u = u[:, 0]  # Scalar
                u_list.append(u)

            # Interpolate constants (surface variables)
            num_face_quads = face_shape_vals.shape[0]
            const_dict = {}
            for i, (name, _) in enumerate(self.constants.items()):
                if i < len(cell_internal_vars_surface):
                    var = cell_internal_vars_surface[i]
                    # For surface vars, typically already at face quad points
                    if var.ndim == 0:
                        # Scalar constant
                        var_quad = np.full(num_face_quads, var)
                    elif var.ndim == 1:
                        if var.shape[0] == num_face_quads:
                            # Scalar at each quad point
                            var_quad = var
                        else:
                            # Assume it's a constant vector, broadcast to all quad points
                            var_quad = np.tile(var, (num_face_quads, 1))
                    elif var.ndim == 2 and var.shape[0] == num_face_quads:
                        # Vector at each quad point (num_face_quads, vec)
                        var_quad = var
                    else:
                        # Fallback
                        var_quad = var
                    const_dict[name] = var_quad
                else:
                    const_dict[name] = np.zeros(num_face_quads)

            # Create evaluator (no gradients on surface for now)
            u_grads_list = [np.zeros((num_face_quads, tf.vec, self.dim)) for tf in self.trial_functions]
            evaluator = RuntimeEvaluator(self, u_list, u_grads_list, const_dict)

            # Initialize weak form values
            weak_form_vals = [np.zeros((self.fes[i].num_nodes, self.fes[i].vec))
                             for i in range(self.num_vars)]

            # Evaluate each surface integral
            for integral in integrals:
                test_func = find_test_function(integral.integrand)
                if test_func is None:
                    continue

                test_idx = test_func.index

                # Get test function data for this variable
                start_idx = self.num_nodes_cumsum[test_idx]
                end_idx = self.num_nodes_cumsum[test_idx + 1]
                face_shape_vals_test = face_shape_vals[:, start_idx:end_idx]
                face_nanson_scale_var = face_nanson_scale[test_idx]

                # Evaluate integrand (without test function)
                integrand_val = self._eval_surface_integrand(integral.integrand, evaluator, test_func)

                # Assembly: ∫ integrand * v * dS
                # integrand_val: (num_face_quads,) or (num_face_quads, vec)
                if integrand_val.ndim == 2:
                    # Vector: (num_face_quads, vec)
                    val = np.sum(integrand_val[:, None, :] * face_shape_vals_test[:, :, None] *
                                face_nanson_scale_var[:, None, None], axis=0)
                else:
                    # Scalar: (num_face_quads,)
                    val = np.sum(integrand_val[:, None] * face_shape_vals_test *
                                face_nanson_scale_var[:, None], axis=0)
                    val = val[:, None]

                weak_form_vals[test_idx] += val

            return jax.flatten_util.ravel_pytree(weak_form_vals)[0]

        return surface_kernel

    def _eval_surface_integrand(self, expr: Expr, evaluator: RuntimeEvaluator, test_func: TestFunction) -> np.ndarray:
        """Evaluate surface integrand (remove test function)."""
        # For f * v on surface, return f
        # For traction · v, return traction

        if isinstance(expr, Mul):
            left_has_test = find_test_function(expr.left) is not None
            right_has_test = find_test_function(expr.right) is not None

            if right_has_test and not left_has_test:
                return evaluator.eval(expr.left)
            elif left_has_test and not right_has_test:
                return evaluator.eval(expr.right)
            else:
                return evaluator.eval(expr.left)

        elif isinstance(expr, Inner):
            # inner(traction, v) -> return traction
            left_has_test = find_test_function(expr.left) is not None
            right_has_test = find_test_function(expr.right) is not None

            if right_has_test and not left_has_test:
                return evaluator.eval(expr.left)
            elif left_has_test and not right_has_test:
                return evaluator.eval(expr.right)
            else:
                return evaluator.eval(expr.left)

        else:
            return evaluator.eval(expr)

    def get_tensor_map(self):
        """Return None to signal we're using universal kernel instead."""
        return None
