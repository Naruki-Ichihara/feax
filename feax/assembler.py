"""
Assembler functions that work with Problem and InternalVars.

This module provides the main assembler API for finite element analysis with
separated internal variables. It handles the assembly of residual vectors and
Jacobian matrices for both volume and surface integrals, supporting various
physics kernels (Laplace, mass, surface, and universal).
"""

import functools
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import jax
import jax.flatten_util
import jax.numpy as np
import numpy as onp
from jax.experimental import sparse

if TYPE_CHECKING:
    from feax.DCboundary import DirichletBC
    from feax.problem import Problem

from feax.internal_vars import InternalVars


# ---------------------------------------------------------------------------
# Internal-variable storage kinds (static type tags)
# ---------------------------------------------------------------------------
# Volume internal variables come in three storage layouts. Historically the
# assembler decided which layout an array used by *sniffing its shape inside*
# the per-element kernel (after vmap had already sliced away the leading axis),
# which is ambiguous when e.g. num_nodes_per_elem == num_quads. We instead
# classify each variable ONCE from its *original* (pre-gather) global shape —
# a purely static, trace-time decision — and thread the resulting tag into the
# kernel so interpolation dispatches without any runtime shape inspection.
_VAR_NODE = 'node'   # global (num_total_nodes,) — interpolate via shape functions
_VAR_CELL = 'cell'   # global (num_cells,) or scalar — broadcast to all quads
_VAR_QUAD = 'quad'   # global (num_cells, num_quads) — already per-quad, pass through


def classify_volume_var(problem: 'Problem', var: np.ndarray):
    """Classify a volume internal variable by its storage kind.

    The decision is made from the *original* global array shape (before
    per-cell gathering), so it is unambiguous and resolved statically at
    trace-build time.

    Returns a ``(kind, fe_idx)`` pair where ``kind`` is one of
    ``_VAR_NODE``/``_VAR_CELL``/``_VAR_QUAD`` (or ``None`` for an unrecognized
    layout → caller falls back to legacy sniffing), and ``fe_idx`` is the index
    of the finite-element variable whose shape functions interpolate a
    node-based variable. ``fe_idx`` matters only for ``_VAR_NODE`` (node-based
    fields are interpolated with that variable's shape functions); it is ``0``
    otherwise. This is what makes mixed element-type multi-variable problems
    interpolate each internal variable on its *own* mesh rather than always on
    variable 0's.
    """
    if var.ndim == 0:
        return _VAR_CELL, 0
    if var.ndim in (1, 2):
        for i, fe in enumerate(problem.fes):
            if var.shape[0] == fe.num_total_nodes:
                return _VAR_NODE, i
    if var.ndim == 1:
        if var.shape[0] == problem.num_cells:
            return _VAR_CELL, 0
        return None, 0
    if var.ndim == 2:
        if var.shape[0] == problem.num_cells:
            # (num_cells, num_quads) is the legacy per-quad layout; anything
            # else with a leading num_cells axis is a per-cell vector.
            if var.shape[1] == problem.fes[0].num_quads:
                return _VAR_QUAD, 0
            return _VAR_CELL, 0
    return None, 0


class Operator:
    """Element-level operator for quadrature-point computations.

    Provides methods for interpolating solutions, computing gradients,
    interpolating internal variables, and integrating over quadrature points.
    Used internally by kernel functions to eliminate code duplication.

    Parameters
    ----------
    problem : Problem
        The finite element problem.
    fe_index : int
        Index of the finite element variable (default 0).
    """

    def __init__(self, problem: 'Problem', fe_index: int = 0):
        self._problem = problem
        self._fe = problem.fes[fe_index]
        self._shape_vals = self._fe.shape_vals    # (num_quads, num_nodes)
        self._num_quads = self._fe.num_quads
        self._num_nodes = self._fe.num_nodes
        self._vec = self._fe.vec

    def eval(self, cell_sol: np.ndarray, shape_vals: np.ndarray = None) -> np.ndarray:
        """Interpolate nodal solution to quadrature points.

        Parameters
        ----------
        cell_sol : np.ndarray
            Nodal solution values, shape (num_nodes, vec).
        shape_vals : np.ndarray, optional
            Shape function values. Uses volume shape functions if None.

        Returns
        -------
        np.ndarray
            Solution at quadrature points, shape (num_quads, vec).
        """
        sv = shape_vals if shape_vals is not None else self._shape_vals
        return np.sum(cell_sol[None, :, :] * sv[:, :, None], axis=1)

    def grad(self, cell_sol: np.ndarray, cell_shape_grads: np.ndarray) -> np.ndarray:
        """Compute solution gradient at quadrature points.

        Parameters
        ----------
        cell_sol : np.ndarray
            Nodal solution values, shape (num_nodes, vec).
        cell_shape_grads : np.ndarray
            Shape function gradients, shape (num_quads, num_nodes, dim).

        Returns
        -------
        np.ndarray
            Gradient at quadrature points, shape (num_quads, vec, dim).
        """
        u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
        return np.sum(u_grads, axis=1)

    def hess(self, cell_sol: np.ndarray, cell_shape_hessians: np.ndarray) -> np.ndarray:
        """Compute solution Hessian (second spatial derivatives) at quadrature points.

        Parameters
        ----------
        cell_sol : np.ndarray
            Nodal solution values, shape (num_nodes, vec).
        cell_shape_hessians : np.ndarray
            Shape function second derivatives in physical coordinates,
            shape (num_quads, num_nodes, dim, dim).

        Returns
        -------
        np.ndarray
            Hessian at quadrature points, shape (num_quads, vec, dim, dim).
            H[q, i, K, L] = sum_a u[a, i] * d²h_a/(dX_K dX_L) at quad point q.
        """
        # cell_sol: (a=num_nodes, v=vec)
        # cell_shape_hessians: (q=num_quads, a=num_nodes, K=dim, L=dim)
        return np.einsum('av,qaKL->qvKL', cell_sol, cell_shape_hessians)

    def interpolate_var(self, var: np.ndarray, kind: str = None,
                        fe_idx: int = 0) -> np.ndarray:
        """Interpolate a single internal variable to quadrature points.

        Handles node-based (shape function interpolation), cell-based
        (broadcast), and quad-based (pass-through) variables.

        Parameters
        ----------
        var : np.ndarray
            Internal variable for a single element (post-gather, post-slice).
        kind : str, optional
            Static storage tag (``_VAR_NODE``/``_VAR_CELL``/``_VAR_QUAD``)
            classified from the original global shape via
            :func:`classify_volume_var`. When provided, interpolation
            dispatches on the tag with no runtime shape inspection. When
            ``None``, falls back to the legacy shape-sniffing path (kept for
            backward compatibility / unclassifiable shapes).
        fe_idx : int, default 0
            For ``_VAR_NODE`` variables, the finite-element variable whose shape
            functions interpolate this field. Lets a node-based internal var
            tied to variable ``i`` be interpolated on variable ``i``'s mesh even
            when this Operator belongs to a different variable (mixed element
            types). Ignored for cell/quad layouts.

        Returns
        -------
        np.ndarray
            Values at quadrature points, shape (num_quads,).
        """
        if kind == _VAR_NODE:
            shape_vals = self._problem.fes[fe_idx].shape_vals
            return np.dot(shape_vals, var)
        elif kind == _VAR_CELL:
            # Per-cell scalar after slicing: either a 0-d scalar or a (1,) array.
            if var.ndim == 0:
                return np.full(self._num_quads, var)
            return np.full(self._num_quads, var.reshape(-1)[0])
        elif kind == _VAR_QUAD:
            return var

        # Legacy fallback: sniff the post-slice shape.
        if var.ndim == 0:
            return np.full(self._num_quads, var)
        elif var.ndim == 1:
            is_node_based = any(
                var.shape[0] == fe.num_nodes for fe in self._problem.fes)
            if is_node_based:
                return np.dot(self._shape_vals, var)
            elif var.shape[0] == 1:
                return np.full(self._num_quads, var[0])
            elif var.shape[0] == self._num_quads:
                return var
            else:
                return np.full(self._num_quads, var[0])
        else:
            return var

    def interpolate_vars(self, cell_internal_vars: Tuple[np.ndarray, ...],
                         kinds: Tuple = None) -> List[np.ndarray]:
        """Interpolate all internal variables to quadrature points.

        Parameters
        ----------
        cell_internal_vars : tuple of np.ndarray
            Internal variables for a single element.
        kinds : tuple of (str, int), optional
            Static ``(kind, fe_idx)`` descriptors, one per variable, from
            :func:`classify_volume_var` (see :meth:`interpolate_var`). When
            ``None``, every variable uses the legacy sniffing path.

        Returns
        -------
        list of np.ndarray
            Interpolated values at quadrature points.
        """
        if kinds is None:
            return [self.interpolate_var(v) for v in cell_internal_vars]
        return [self.interpolate_var(v, kind, fe_idx)
                for v, (kind, fe_idx) in zip(cell_internal_vars, kinds)]

    def integrate_grad(self, quad_values: np.ndarray,
                       cell_v_grads_JxW: np.ndarray) -> np.ndarray:
        """Integrate tensor quad-point values against test function gradients.

        Computes: sum_q sigma(q) : grad_v(q) * JxW(q)

        Parameters
        ----------
        quad_values : np.ndarray
            Physics values at quad points, shape (num_quads, vec, dim).
        cell_v_grads_JxW : np.ndarray
            Pre-multiplied test function gradients × JxW,
            shape (num_quads, num_nodes, vec, dim).

        Returns
        -------
        np.ndarray
            Element contribution, shape (num_nodes, vec).
        """
        return np.sum(
            quad_values[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))

    def integrate_val(self, quad_values: np.ndarray,
                      cell_JxW: np.ndarray,
                      shape_vals: np.ndarray = None) -> np.ndarray:
        """Integrate scalar/vector quad-point values with shape functions.

        Computes: sum_q f(q) * N(q) * JxW(q)

        Parameters
        ----------
        quad_values : np.ndarray
            Physics values at quad points, shape (num_quads, vec).
        cell_JxW : np.ndarray
            Jacobian determinant × quadrature weights, shape (num_quads,).
        shape_vals : np.ndarray, optional
            Shape function values. Uses volume shape functions if None.

        Returns
        -------
        np.ndarray
            Element contribution, shape (num_nodes, vec).
        """
        sv = shape_vals if shape_vals is not None else self._shape_vals
        return np.sum(
            quad_values[:, None, :] * sv[:, :, None] * cell_JxW[:, None, None],
            axis=0)

    @staticmethod
    def gather_internal_vars(problem: 'Problem',
                             internal_vars: Tuple[np.ndarray, ...],
                             sv=None) -> List[np.ndarray]:
        """Gather global internal variables to per-cell format.

        Transforms node-based variables from global (num_nodes,) arrays to
        per-cell (num_cells, num_nodes_per_elem) arrays using element
        connectivity. Cell-based and quad-based variables are passed through.

        Parameters
        ----------
        problem : Problem
            The finite element problem with connectivity information.
        internal_vars : tuple of np.ndarray
            Global internal variables.

        Returns
        -------
        list of np.ndarray
            Per-cell internal variables ready for vmapped kernels.
        """
        result = []
        for var in internal_vars:
            # 1-D (node/cell scalar field) and 2-D (node/cell vector field)
            # share identical gather logic: cell-based arrays pass through,
            # node-based arrays are gathered to per-cell via element connectivity.
            if var.ndim in (1, 2):
                if var.shape[0] == problem.num_cells:
                    result.append(var)
                else:
                    fe_idx = 0
                    for i, fe in enumerate(problem.fes):
                        if var.shape[0] == fe.num_total_nodes:
                            fe_idx = i
                            break
                    cells = (sv.cells_list[fe_idx] if sv is not None
                             else problem.fes[fe_idx].cells)
                    result.append(var[cells])
            else:
                result.append(var)
        return result


def interpolate_to_quad_points(var: np.ndarray, shape_vals: np.ndarray, num_cells: int, num_quads: int) -> np.ndarray:
    """Interpolate node-based or cell-based values to quadrature points.

    .. deprecated::
        Use :meth:`Operator.interpolate_var` instead.

    This function handles three cases:
    1. Node-based: shape (num_nodes,) -> interpolate using shape functions
    2. Cell-based: shape (num_cells,) -> broadcast to all quad points in cell
    3. Quad-based: shape (num_cells, num_quads) -> pass through (legacy)

    Parameters
    ----------
    var : np.ndarray
        Variable to interpolate. Can be:
        - (num_nodes,) for node-based
        - (num_cells,) for cell-based
        - (num_cells, num_quads) for quad-based (legacy)
    shape_vals : np.ndarray
        Shape function values at quadrature points, shape (num_quads, num_nodes)
    num_cells : int
        Number of cells/elements
    num_quads : int
        Number of quadrature points per cell

    Returns
    -------
    np.ndarray
        Values at quadrature points, shape (num_quads,)
    """
    if var.ndim == 1:
        if var.shape[0] == num_cells:
            # Cell-based: broadcast single value to all quad points
            return np.full(num_quads, var[0])  # For single cell, var[0] is the cell value
        else:
            # Node-based: interpolate using shape functions
            # var has shape (num_nodes,), need to extract cell nodes
            # This is handled by the caller passing cell_var_nodal
            return np.dot(shape_vals, var)  # (num_quads, num_nodes) @ (num_nodes,) -> (num_quads,)
    elif var.ndim == 2:
        # Quad-based (legacy): shape (num_cells, num_quads)
        # Return just this cell's quad values
        return var[0]  # Assumes var is already sliced for this cell
    else:
        raise ValueError(f"Variable has unexpected shape: {var.shape}")


def get_laplace_kernel(problem: 'Problem', tensor_map: Callable,
                       var_kinds: Tuple[str, ...] = None) -> Callable:
    """Create Laplace kernel function for gradient-based physics.

    The Laplace kernel handles gradient-based terms in the weak form, such as
    those arising in elasticity, heat conduction, and diffusion problems. It
    implements the integral term: ∫ σ(∇u) : ∇v dΩ where σ is the stress/flux
    tensor computed from the gradient.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and element information.
    tensor_map : Callable
        Function that maps gradient tensor to stress/flux tensor.
        Signature: (u_grad: ndarray, *internal_vars) -> ndarray
        where u_grad has shape (vec, dim) and returns (vec, dim).

    Returns
    -------
    Callable
        Laplace kernel function that computes the contribution to the weak form
        from gradient-based physics.
    """
    op = Operator(problem)

    def laplace_kernel(cell_sol_flat: np.ndarray,
                      cell_shape_grads: np.ndarray,
                      cell_v_grads_JxW: np.ndarray,
                      *cell_internal_vars: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        cell_shape_grads = cell_shape_grads[:, :op._num_nodes, :]
        cell_v_grads_JxW = cell_v_grads_JxW[:, :op._num_nodes, :, :]

        u_grads = op.grad(cell_sol, cell_shape_grads)  # (num_quads, vec, dim)
        vars_quad = op.interpolate_vars(cell_internal_vars, var_kinds)
        u_physics = jax.vmap(tensor_map)(u_grads, *vars_quad)

        val = op.integrate_grad(u_physics, cell_v_grads_JxW)
        return jax.flatten_util.ravel_pytree(val)[0]

    return laplace_kernel


def get_mass_kernel(problem: 'Problem', mass_map: Callable,
                    var_kinds: Tuple[str, ...] = None) -> Callable:
    """Create mass kernel function for non-gradient terms.

    The mass kernel handles terms without derivatives in the weak form, such as
    mass matrices, reaction terms, or body forces. It implements the integral
    term: ∫ m(u, x) · v dΩ where m is a mass-like term.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and element information.
    mass_map : Callable
        Function that computes the mass term.
        Signature: (u: ndarray, x: ndarray, *internal_vars) -> ndarray
        where u has shape (vec,), x has shape (dim,), and returns (vec,).

    Returns
    -------
    Callable
        Mass kernel function that computes the contribution to the weak form
        from non-gradient physics.
    """
    op = Operator(problem)

    def mass_kernel(cell_sol_flat: np.ndarray,
                   x: np.ndarray,
                   cell_JxW: np.ndarray,
                   *cell_internal_vars: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        cell_JxW = cell_JxW[0]

        u = op.eval(cell_sol)  # (num_quads, vec)
        vars_quad = op.interpolate_vars(cell_internal_vars, var_kinds)
        u_physics = jax.vmap(mass_map)(u, x, *vars_quad)  # (num_quads, vec)

        val = op.integrate_val(u_physics, cell_JxW)
        return jax.flatten_util.ravel_pytree(val)[0]

    return mass_kernel


def get_surface_kernel(problem: 'Problem', surface_map: Callable) -> Callable:
    """Create surface kernel function for boundary integrals.

    The surface kernel handles boundary integrals in the weak form, such as
    surface tractions, pressures, or fluxes. It implements the integral term:
    ∫ t(u, x) · v dΓ where t is the surface load/flux.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and element information.
    surface_map : Callable
        Function that computes the surface traction/flux.
        Signature: (u: ndarray, x: ndarray, *internal_vars) -> ndarray
        where u has shape (vec,), x has shape (dim,), and returns (vec,).

    Returns
    -------
    Callable
        Surface kernel function that computes the contribution to the weak form
        from boundary loads/fluxes.
    """
    op = Operator(problem)

    def surface_kernel(cell_sol_flat: np.ndarray,
                      x: np.ndarray,
                      face_shape_vals: np.ndarray,
                      face_shape_grads: np.ndarray,
                      face_nanson_scale: np.ndarray,
                      *cell_internal_vars_surface: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        face_shape_vals = face_shape_vals[:, :op._num_nodes]
        face_nanson_scale = face_nanson_scale[0]

        u = op.eval(cell_sol, shape_vals=face_shape_vals)  # (num_face_quads, vec)
        u_physics = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)

        val = op.integrate_val(u_physics, face_nanson_scale, shape_vals=face_shape_vals)
        return jax.flatten_util.ravel_pytree(val)[0]

    return surface_kernel


def _build_surface_kernel_from_weak_form(problem: 'Problem', surface_weak_form: Callable) -> Callable:
    """Build a surface kernel from a high-level surface weak form function.

    Converts a user-defined surface weak form (operating at a single surface
    quadrature point) into a full element-level surface kernel.

    Parameters
    ----------
    problem : Problem
        The finite element problem.
    surface_weak_form : Callable
        Surface weak form function with signature:
        ``(vals, x, *internal_vars) -> tractions``
        where vals[i] is (vec_i,), tractions[i] is (vec_i,).
        Automatically vmapped over surface quadrature points.

    Returns
    -------
    Callable
        Element-level surface kernel compatible with the assembler.
    """
    ops = [Operator(problem, i) for i in range(problem.num_vars)]
    # Compute static node offsets (Python ints, JIT-safe)
    nc = [0]
    for op in ops:
        nc.append(nc[-1] + op._num_nodes)

    def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
               face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)

        # Interpolate solutions to surface quad points per variable
        vals = [ops[i].eval(cell_sol_list[i],
                            shape_vals=face_shape_vals[:, nc[i]:nc[i+1]])
                for i in range(problem.num_vars)]

        # Call surface weak_form vmapped over surface quad points
        tractions = jax.vmap(surface_weak_form)(
            vals, physical_surface_quad_points, *cell_internal_vars_surface)

        # Integrate per variable: ∫ t_i · v_i dΓ
        residuals = []
        for i in range(problem.num_vars):
            fsv = face_shape_vals[:, nc[i]:nc[i+1]]
            res_i = ops[i].integrate_val(tractions[i], face_nanson_scale[i],
                                         shape_vals=fsv)
            residuals.append(res_i)

        return jax.flatten_util.ravel_pytree(residuals)[0]

    return kernel


def _build_kernel_from_weak_form(problem: 'Problem', weak_form: Callable,
                                 var_kinds: Tuple[str, ...] = None) -> Callable:
    """Build a volume kernel from a high-level weak form function.

    Converts a user-defined weak form (operating at a single quadrature point)
    into a full element-level kernel that handles solution interpolation,
    gradient computation, internal variable interpolation, and integration.

    Parameters
    ----------
    problem : Problem
        The finite element problem.
    weak_form : Callable
        Weak form function with signature:
        ``(vals, grads, x, *internal_vars) -> (mass_terms, grad_terms)``
        where vals[i] is (vec_i,), grads[i] is (vec_i, dim),
        mass_terms[i] is (vec_i,), grad_terms[i] is (vec_i, dim).
        Automatically vmapped over quadrature points.

    Returns
    -------
    Callable
        Element-level kernel compatible with the assembler.
    """
    ops = [Operator(problem, i) for i in range(problem.num_vars)]
    # Compute static node offsets from Operator instances (Python ints, JIT-safe)
    nc = [0]
    for op in ops:
        nc.append(nc[-1] + op._num_nodes)

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
               cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)

        # Interpolate solutions and gradients to quad points
        vals = [ops[i].eval(cell_sol_list[i]) for i in range(problem.num_vars)]
        grads = [ops[i].grad(cell_sol_list[i], cell_shape_grads[:, nc[i]:nc[i+1], :])
                 for i in range(problem.num_vars)]

        # Interpolate internal vars to quad points
        vars_quad = ops[0].interpolate_vars(cell_internal_vars, var_kinds)

        # Call weak_form vmapped over quadrature points
        mass_terms, grad_terms = jax.vmap(weak_form)(
            vals, grads, physical_quad_points, *vars_quad)

        # Integrate per variable
        residuals = []
        for i in range(problem.num_vars):
            vg = cell_v_grads_JxW[:, nc[i]:nc[i+1], :, :]
            res_i = ops[i].integrate_val(mass_terms[i], cell_JxW[i]) + \
                    ops[i].integrate_grad(grad_terms[i], vg)
            residuals.append(res_i)

        return jax.flatten_util.ravel_pytree(residuals)[0]

    return kernel


def create_volume_kernel(problem: 'Problem',
                         var_kinds: Tuple = None) -> Callable:
    """Create the unified volume kernel for residual / Jacobian assembly.

    Composition rules (consistent for single- and multi-variable problems):

    1. **Base element residual** (full physics, chosen exclusively):

       - ``get_universal_kernel()`` — if defined, it is the complete element
         residual (full low-level control); the standard pieces below are *not*
         added. This is the same "escape hatch" meaning in both single- and
         multi-variable problems.
       - otherwise, for multi-variable problems: ``get_weak_form()``
         (raises if neither is defined).
       - otherwise, for single-variable problems: ``get_tensor_map()`` (or
         ``get_energy_density()`` differentiated via ``jax.grad``) **plus**
         ``get_mass_map()`` — these are complementary standard pieces and are
         summed.

    2. **Extra additive kernel** (optional): ``get_extra_kernel()`` is *always
       added on top* of the base, regardless of which base was chosen. Use it
       for complementary low-level terms (e.g. a regularization or stabilization
       term layered on standard physics) — the additive counterpart of the
       full-replacement ``get_universal_kernel()``.

    Parameters
    ----------
    problem : Problem
        Defines some subset of ``get_tensor_map`` / ``get_energy_density`` /
        ``get_mass_map`` / ``get_weak_form`` / ``get_universal_kernel`` /
        ``get_extra_kernel``.

    Returns
    -------
    Callable
        Element kernel ``base(...) + extra(...)``.
    """

    # Resolve the BASE element kernel (full replacement) once, outside the trace.
    _universal = problem.get_universal_kernel() if hasattr(problem, 'get_universal_kernel') else None

    _weak_form_kernel = None
    if _universal is None and problem.num_vars > 1:
        weak_form = problem.get_weak_form()
        if weak_form is None:
            raise ValueError(
                "Multi-variable problems require get_universal_kernel() or get_weak_form()")
        _weak_form_kernel = _build_kernel_from_weak_form(problem, weak_form, var_kinds)

    # Resolve the optional EXTRA (always-additive) kernel.
    _extra = problem.get_extra_kernel() if hasattr(problem, 'get_extra_kernel') else None

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        # --- base element residual (exclusive) ---
        if _universal is not None:
            base = _universal(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW,
                              cell_v_grads_JxW, *cell_internal_vars)
        elif problem.num_vars > 1:
            base = _weak_form_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW,
                                     cell_v_grads_JxW, *cell_internal_vars)
        else:
            base = 0.
            # mass (non-gradient) term
            if hasattr(problem, 'get_mass_map') and problem.get_mass_map() is not None:
                mass_kernel = get_mass_kernel(problem, problem.get_mass_map(), var_kinds)
                base = base + mass_kernel(cell_sol_flat, physical_quad_points, cell_JxW, *cell_internal_vars)
            # laplace (gradient) term: explicit tensor_map, else energy_density via jax.grad
            tensor_map = problem.get_tensor_map() if hasattr(problem, 'get_tensor_map') else None
            if tensor_map is None and hasattr(problem, 'get_energy_density'):
                energy_density = problem.get_energy_density()
                if energy_density is not None:
                    tensor_map = jax.grad(energy_density)
            if tensor_map is not None:
                laplace_kernel = get_laplace_kernel(problem, tensor_map, var_kinds)
                base = base + laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars)

        # --- extra additive kernel (always added on top of the base) ---
        if _extra is not None:
            base = base + _extra(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW,
                                 cell_v_grads_JxW, *cell_internal_vars)
        return base

    return kernel


def create_surface_kernel(problem: 'Problem', surface_index: int) -> Callable:
    """Create unified surface kernel for a specific boundary.

    This function creates a kernel that combines contributions from standard
    surface maps and universal surface kernels for a specific boundary surface
    identified by surface_index.

    Parameters
    ----------
    problem : Problem
        The finite element problem that may define get_surface_maps() and/or
        get_universal_kernels_surface() methods.
    surface_index : int
        Index identifying which boundary surface this kernel is for.
        Corresponds to the index in problem.location_fns.

    Returns
    -------
    Callable
        Combined surface kernel function for the specified boundary.

    Notes
    -----
    Multiple boundaries can have different physics. The surface_index
    parameter selects which surface map and universal kernel to use.

    For multi-variable problems, only universal_kernels_surface should be used,
    as get_surface_maps() only supports single-variable problems.
    """

    # Pre-build multi-variable surface kernel outside traced function
    _multi_var_surface_kernel = None
    if problem.num_vars > 1:
        if hasattr(problem, 'get_universal_kernels_surface') and len(problem.get_universal_kernels_surface()) > surface_index:
            _multi_var_surface_kernel = problem.get_universal_kernels_surface()[surface_index]
        elif hasattr(problem, 'get_surface_weak_forms') and callable(getattr(problem, 'get_surface_weak_forms', None)):
            swf_list = problem.get_surface_weak_forms()
            if len(swf_list) > surface_index:
                _multi_var_surface_kernel = _build_surface_kernel_from_weak_form(problem, swf_list[surface_index])

    def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
        # For multi-variable problems, use pre-built kernel
        if problem.num_vars > 1:
            if _multi_var_surface_kernel is not None:
                return _multi_var_surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                    face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
            elif hasattr(problem, 'get_surface_maps') and len(problem.get_surface_maps()) > surface_index:
                raise ValueError(
                    f"Multi-variable problems (num_vars={problem.num_vars}) cannot use get_surface_maps(). "
                    f"Please implement get_surface_weak_forms() instead."
                )
            else:
                # No surface kernel defined - return zeros with proper shape
                return np.zeros(problem.num_total_dofs_all_vars)

        # Single-variable: use surface_maps and/or universal kernels
        surface_val = 0.
        if hasattr(problem, 'get_surface_maps') and len(problem.get_surface_maps()) > surface_index:
            surface_kernel = get_surface_kernel(problem, problem.get_surface_maps()[surface_index])
            surface_val = surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)

        universal_val = 0.
        if hasattr(problem, 'get_universal_kernels_surface') and len(problem.get_universal_kernels_surface()) > surface_index:
            universal_kernel = problem.get_universal_kernels_surface()[surface_index]
            universal_val = universal_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)

        return surface_val + universal_val

    return kernel


def _value_and_jacfwd(f: Callable, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-mode value-and-Jacobian for a single element kernel.

    Returns ``(y, jac)`` where ``y = f(x)`` and ``jac`` has the differentiated
    axis last (``out_axes=-1``), matching the assembler's element-Jacobian
    layout ``(num_dofs, num_dofs)``.
    """
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
    y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis,))
    return y, jac


# Fixed number of cells/faces per scan batch. Holding the batch size constant
# (rather than the legacy "≤100 cuts", which made the batch grow ∝ N) keeps the
# per-batch working set — and hence peak assembly memory — independent of the
# mesh size.
_ASSEMBLY_BATCH_SIZE = 256


def _batching(n_items: int, batch_size: int = _ASSEMBLY_BATCH_SIZE):
    """Return ``(batch_size, num_cuts, n_padded)`` for a fixed-size scan.

    ``num_cuts = ceil(n_items / batch_size)`` and ``n_padded = num_cuts *
    batch_size`` (≥ n_items). Padding the leading axis up to ``n_padded`` lets
    every scan step share one shape, so there is a single compiled batch path
    (no separate ragged-remainder kernel).
    """
    if n_items <= 0:
        return 0, 0, 0
    bs = min(batch_size, n_items)
    num_cuts = -(-n_items // bs)            # ceil division
    return bs, num_cuts, num_cuts * bs


def _pad_axis0(x: np.ndarray, pad: int) -> np.ndarray:
    """Append ``pad`` zero rows along axis 0."""
    if pad == 0:
        return x
    return np.concatenate(
        [x, np.zeros((pad, *x.shape[1:]), dtype=x.dtype)], axis=0)


def _scan_vmap_batched(vmap_fn: Callable,
                       input_collection: list,
                       n_items: int,
                       jac_flag: bool,
                       batch_size: int = _ASSEMBLY_BATCH_SIZE) -> Any:
    """Apply a vmapped kernel over ``n_items`` rows in fixed-size scan batches.

    The leading axis is padded to a multiple of ``batch_size`` and reshaped to
    ``(num_cuts, batch_size, …)`` for a single ``lax.scan`` — only one batch's
    per-item Jacobians are live at a time, and there is exactly one compiled
    batch shape (the padded tail rows are sliced off the result). Peak memory is
    therefore ``batch_size``-bounded and mesh-size independent.

    Parameters
    ----------
    vmap_fn : Callable
        ``jax.vmap`` of the per-item kernel. Returns an array when
        ``jac_flag`` is False, or a ``(val, jac)`` tuple when True.
    input_collection : list of np.ndarray
        Per-item inputs, each with leading axis ``n_items``.
    n_items : int
        Number of rows (cells or faces).
    jac_flag : bool
        Whether ``vmap_fn`` also returns the Jacobian.
    batch_size : int
        Rows per scan batch (fixed).

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Results over all ``n_items`` rows, matching ``vmap_fn``'s output arity.
    """
    bs, num_cuts, n_padded = _batching(n_items, batch_size)
    if num_cuts <= 1:
        # Empty, or a single batch — just vmap (no scan, no padding).
        return vmap_fn(*input_collection)

    pad = n_padded - n_items
    batched = jax.tree_util.tree_map(
        lambda x: _pad_axis0(x, pad).reshape(num_cuts, bs, *x.shape[1:]),
        input_collection)

    if jac_flag:
        def scan_body(carry, batch):
            return carry, vmap_fn(*batch)
        _, (vals, jacs) = jax.lax.scan(scan_body, None, batched)
        vals = vals.reshape(n_padded, *vals.shape[2:])[:n_items]
        jacs = jacs.reshape(n_padded, *jacs.shape[2:])[:n_items]
        return vals, jacs
    else:
        def scan_body(carry, batch):
            return carry, vmap_fn(*batch)
        _, vals = jax.lax.scan(scan_body, None, batched)
        return vals.reshape(n_padded, *vals.shape[2:])[:n_items]


def split_and_compute_cell(problem: 'Problem',
                           cells_sol_flat: np.ndarray,
                           jac_flag: bool,
                           internal_vars_volume: Tuple[np.ndarray, ...],
                           sv=None) -> Any:
    """Compute volume integrals for residual or Jacobian assembly.
    
    This function evaluates volume integrals over all elements, optionally
    computing the Jacobian via forward-mode automatic differentiation. It
    uses batching to manage memory for large meshes.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and quadrature data.
    cells_sol_flat : np.ndarray
        Flattened solution values at element nodes.
        Shape: (num_cells, num_nodes * vec).
    jac_flag : bool
        If True, compute both values and Jacobian. If False, compute only values.
    internal_vars_volume : tuple of np.ndarray
        Material properties at quadrature points for each variable.
        Each array has shape (num_cells, num_quads).
    
    Returns
    -------
    np.ndarray or tuple of np.ndarray
        If jac_flag is False: weak form values with shape (num_cells, num_dofs).
        If jac_flag is True: tuple of (values, jacobian) where jacobian has
        shape (num_cells, num_dofs, num_dofs).
    
    Notes
    -----
    The function splits computation into batches (default 20) to avoid memory
    issues with large meshes. This is particularly important for 3D problems.
    """

    # Classify each volume variable's storage kind ONCE from its original
    # global shape (static, trace-time). This lets the per-element kernel
    # dispatch interpolation on an explicit tag instead of sniffing the
    # post-slice shape (which is ambiguous when num_nodes == num_quads).
    var_kinds = tuple(classify_volume_var(problem, v) for v in internal_vars_volume)
    kernel = create_volume_kernel(problem, var_kinds)

    src = sv if sv is not None else problem

    if jac_flag:
        def kernel_jac(cell_sol_flat, *args):
            kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
            return _value_and_jacfwd(kernel_partial, cell_sol_flat)
        vmap_fn = jax.vmap(kernel_jac)
    else:
        vmap_fn = jax.vmap(kernel)

    internal_vars_per_cell = Operator.gather_internal_vars(
        problem, internal_vars_volume, sv)

    input_collection = [cells_sol_flat, src.physical_quad_points, src.shape_grads,
                       src.JxW, src.v_grads_JxW, *internal_vars_per_cell]

    # Adaptive batching bounds peak memory from materializing per-element
    # Jacobians (especially for large 3D meshes).
    return _scan_vmap_batched(vmap_fn, input_collection, problem.num_cells,
                              jac_flag)


def compute_face(problem: 'Problem',
                cells_sol_flat: np.ndarray,
                jac_flag: bool,
                internal_vars_surfaces: List[Tuple[np.ndarray, ...]],
                sv=None) -> Any:
    """Compute surface integrals for residual or Jacobian assembly.
    
    This function evaluates surface integrals over all boundary faces,
    optionally computing the Jacobian via forward-mode automatic differentiation.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing boundary information.
    cells_sol_flat : np.ndarray
        Flattened solution values at element nodes.
        Shape: (num_cells, num_nodes * vec).
    jac_flag : bool
        If True, compute both values and Jacobian. If False, compute only values.
    internal_vars_surfaces : list of tuple of np.ndarray
        Surface variables for each boundary. Each entry corresponds to one
        boundary surface and contains arrays with shape
        (num_surface_faces, num_face_quads).
    
    Returns
    -------
    list of np.ndarray or list of tuple
        If jac_flag is False: list of weak form values for each boundary.
        If jac_flag is True: list of (values, jacobian) tuples for each boundary.
    
    Notes
    -----
    Each boundary surface can have different loading conditions or physics,
    handled through separate surface kernels and internal variables.
    """

    src = sv if sv is not None else problem

    if jac_flag:
        values = []
        jacs = []
        for i, boundary_inds in enumerate(src.boundary_inds_list):
            kernel = create_surface_kernel(problem, i)
            def kernel_jac(cell_sol_flat, *args):
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return _value_and_jacfwd(kernel_partial, cell_sol_flat)
            vmap_fn = jax.vmap(kernel_jac)

            selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]

            # Handle case where internal_vars_surfaces might be empty or insufficient
            surface_vars_for_boundary = internal_vars_surfaces[i] if i < len(internal_vars_surfaces) else ()

            input_collection = [selected_cell_sols_flat, src.physical_surface_quad_points[i],
                              src.selected_face_shape_vals[i], src.selected_face_shape_grads[i],
                              src.nanson_scale[i], *surface_vars_for_boundary]

            # Batch the surface Jacobian the same way as the volume path so a
            # large boundary (e.g. a pressure surface) does not materialize all
            # per-face Jacobians at once.
            n_faces = boundary_inds.shape[0]
            val, jac = _scan_vmap_batched(vmap_fn, input_collection, n_faces, True)
            values.append(val)
            jacs.append(jac)
        return values, jacs
    else:
        values = []
        for i, boundary_inds in enumerate(src.boundary_inds_list):
            kernel = create_surface_kernel(problem, i)
            vmap_fn = jax.vmap(kernel)

            selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]

            # Handle case where internal_vars_surfaces might be empty or insufficient
            surface_vars_for_boundary = internal_vars_surfaces[i] if i < len(internal_vars_surfaces) else ()

            input_collection = [selected_cell_sols_flat, src.physical_surface_quad_points[i],
                              src.selected_face_shape_vals[i], src.selected_face_shape_grads[i],
                              src.nanson_scale[i], *surface_vars_for_boundary]
            n_faces = boundary_inds.shape[0]
            val = _scan_vmap_batched(vmap_fn, input_collection, n_faces, False)
            values.append(val)
        return values


def compute_residual_vars_helper(problem: 'Problem',
                                 weak_form_flat: np.ndarray,
                                 weak_form_face_flat: List[np.ndarray],
                                 sv=None) -> List[np.ndarray]:
    """Assemble residual from element and face contributions.
    
    This helper function assembles the global residual vector by accumulating
    contributions from volume and surface integrals at the appropriate nodes.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing connectivity information.
    weak_form_flat : np.ndarray
        Flattened weak form values from volume integrals.
        Shape: (num_cells, num_dofs_per_cell).
    weak_form_face_flat : list of np.ndarray
        Weak form values from surface integrals for each boundary.
        Each array has shape (num_boundary_faces, num_dofs_per_face).
    
    Returns
    -------
    list of np.ndarray
        Global residual for each solution variable.
        Each array has shape (num_total_nodes, vec).
    
    Notes
    -----
    Volume and all boundary contributions are concatenated into one value array
    and reduced with a single sorted ``segment_sum`` over the precomputed DOF
    scatter map (``problem.res_perm`` / ``res_sorted_dofs``). This replaces the
    ``1 + num_boundaries`` scatter-adds with one deterministic reduction (no
    atomics), matching the CSR-direct Jacobian assembly.
    """
    src = sv if sv is not None else problem
    # Values in volume-then-boundary order — exactly the order the residual
    # scatter DOFs were assembled in Problem.__post_init__.
    val_arrays = [weak_form_flat.reshape(-1)]
    for j in range(len(problem.boundary_inds_list)):
        val_arrays.append(weak_form_face_flat[j].reshape(-1))
    V = np.concatenate(val_arrays) if len(val_arrays) > 1 else val_arrays[0]

    res_flat = jax.ops.segment_sum(
        V[src.res_perm], src.res_sorted_dofs,
        num_segments=problem.num_total_dofs_all_vars, indices_are_sorted=True)
    return problem.unflatten_fn_sol_list(res_flat)


def _assemble_jacobian_values(problem: 'Problem',
                              sol_list: List[np.ndarray],
                              internal_vars: InternalVars) -> np.ndarray:
    """Assemble the raw per-entry Jacobian value array ``V``.

    Computes the element-local Jacobians for the volume and every boundary and
    concatenates them in **volume-then-boundary order** — exactly the order in
    which ``problem.I`` / ``problem.J`` were built in ``Problem.__post_init__``.
    ``V`` therefore has length ``len(problem.I)`` (the unfiltered COO length),
    with one value per (row, col) entry including duplicates.

    This is the shared front half of both the BCOO path (:func:`get_jacobian`) and the
    CSR-direct path (:func:`_get_J_csr`): the only difference downstream is how
    ``V`` is turned into a matrix representation.
    """
    cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)

    # Jacobian values from volume integrals.
    _, cells_jac_flat = split_and_compute_cell(problem, cells_sol_flat, True, internal_vars.volume_vars)
    V_arrays = [cells_jac_flat.reshape(-1)]

    # Jacobian values from surface integrals (per boundary, in order).
    _, cells_jac_face_flat = compute_face(problem, cells_sol_flat, True, internal_vars.surface_vars)
    for cells_jac_f_flat in cells_jac_face_flat:
        V_arrays.append(cells_jac_f_flat.reshape(-1))

    V = np.concatenate(V_arrays) if len(V_arrays) > 1 else np.array(V_arrays[0])

    # V-order contract: the concatenation order here MUST mirror the order in
    # which problem.I / problem.J were assembled. Shapes are static under JIT,
    # so this assert is trace-time only and adds no runtime cost.
    assert V.shape[0] == problem.I.shape[0], (
        f"V-order contract violated: assembled {V.shape[0]} jacobian values but "
        f"index array has {problem.I.shape[0]} entries. The volume/surface "
        f"concatenation order in _assemble_jacobian_values must mirror "
        f"Problem.__post_init__."
    )
    return V


def _csr_data_from_values(problem: 'Problem', V: np.ndarray) -> np.ndarray:
    """Turn a full raw value array ``V`` into the deduplicated CSR ``data``.

    Uses the precomputed slot map (``problem.csr_perm`` / ``csr_seg_ids``) to
    perform a single sorted ``segment_sum`` — no sort, no BCOO, and (unlike
    scatter-add) deterministic. The matrix-view filter is applied first so that
    UPPER/LOWER assemble only their retained entries.

    This materializes the full value array; the memory-efficient assembly path
    (:func:`_assemble_volume_csr_data`) avoids that. Kept for reference / testing.
    """
    V_f = V if problem.filter_indices is None else V[problem.filter_indices]
    return jax.ops.segment_sum(
        V_f[problem.csr_perm], problem.csr_seg_ids,
        num_segments=problem.csr_nse, indices_are_sorted=True)


def _vol_slot_sort(problem: 'Problem') -> Tuple[onp.ndarray, onp.ndarray]:
    """Precompute (and cache on ``problem``) the static volume slot-sort.

    Returns ``(perm, sorted_slots)`` as concrete numpy arrays. For the batched
    case (``num_cuts > 1``) both have shape ``(num_cuts, bs * ndof^2)``; for a
    single batch they are flat ``(num_cells * ndof^2,)``. The sort depends only
    on mesh structure, so it is computed once per Problem — either embedded as
    a trace-time constant (closure path) or shipped as :class:`StaticVars`
    leaves (runtime-argument path).
    """
    cached = getattr(problem, '_vol_slot_sort_cache', None)
    if cached is not None:
        return cached

    vslots = onp.asarray(problem.csr_volume_slots)     # (num_cells, ndof^2)
    ndof2 = vslots.shape[1]
    nse = problem.csr_nse
    bs, num_cuts, n_padded = _batching(problem.num_cells)

    if num_cuts <= 1:
        flat = vslots.reshape(-1)
        perm = onp.argsort(flat, kind='stable')
        out = (perm, flat[perm])
    else:
        # Pad cells up to n_padded; padded cells' slots are the discard bucket
        # (nse) so their Jacobians never reach the matrix.
        pad = n_padded - problem.num_cells
        vslots_pad = onp.concatenate(
            [vslots, onp.full((pad, ndof2), nse, dtype=vslots.dtype)], axis=0)
        vslots_batched = vslots_pad.reshape(num_cuts, bs * ndof2)
        perm = onp.argsort(vslots_batched, axis=1, kind='stable')
        out = (perm, onp.take_along_axis(vslots_batched, perm, axis=1))

    problem._vol_slot_sort_cache = out
    return out


def _res_vol_slot_sort(problem: 'Problem') -> Tuple[onp.ndarray, onp.ndarray]:
    """Precompute (and cache on ``problem``) the static residual-DOF sort used
    by the fused residual+Jacobian volume assembly.

    Same contract as :func:`_vol_slot_sort` but for the per-cell residual DOF
    map (``problem.res_volume_dofs``): returns ``(perm, sorted_dofs)``, batched
    ``(num_cuts, bs * ndof)`` or flat.
    """
    cached = getattr(problem, '_res_vol_slot_sort_cache', None)
    if cached is not None:
        return cached

    rdofs = onp.asarray(problem.res_volume_dofs)       # (num_cells, ndof)
    ndof = rdofs.shape[1]
    ndof_total = problem.num_total_dofs_all_vars
    bs, num_cuts, n_padded = _batching(problem.num_cells)

    if num_cuts <= 1:
        flat = rdofs.reshape(-1)
        perm = onp.argsort(flat, kind='stable')
        out = (perm, flat[perm])
    else:
        pad = n_padded - problem.num_cells
        rdofs_pad = onp.concatenate(
            [rdofs, onp.full((pad, ndof), ndof_total, dtype=rdofs.dtype)], axis=0)
        rdofs_b = rdofs_pad.reshape(num_cuts, bs * ndof)
        perm = onp.argsort(rdofs_b, axis=1, kind='stable')
        out = (perm, onp.take_along_axis(rdofs_b, perm, axis=1))

    problem._res_vol_slot_sort_cache = out
    return out


def _assemble_volume_csr_data(problem: 'Problem',
                              cells_sol_flat: np.ndarray,
                              internal_vars_volume: Tuple[np.ndarray, ...],
                              sv=None) -> np.ndarray:
    """Assemble the volume contribution to the CSR ``data`` without holding all
    element Jacobians.

    Each scan batch computes only its own element Jacobians, reduces them into
    the shared length-``nse`` CSR accumulator via ``segment_sum`` over the
    precomputed per-cell slot map, and is then discarded. Peak memory is one
    batch of element Jacobians plus the accumulator — not the full
    ``(num_cells, ndof, ndof)`` stack. This is the core memory win of the
    CSR-direct path.

    When ``sv`` (:class:`feax.static_vars.StaticVars`) is given, all mesh-sized
    arrays are read from it (traced arguments) instead of ``problem`` (closure
    constants) — see the StaticVars module docstring.

    Returns an array of length ``nse + 1`` (the trailing slot is the discard
    bucket for entries dropped by the UPPER/LOWER filter).
    """
    src = sv if sv is not None else problem
    var_kinds = tuple(classify_volume_var(problem, v) for v in internal_vars_volume)
    kernel = create_volume_kernel(problem, var_kinds)

    def kernel_jac(cell_sol_flat, *args):
        return _value_and_jacfwd(lambda s: kernel(s, *args), cell_sol_flat)
    vmap_fn = jax.vmap(kernel_jac)

    internal_vars_per_cell = Operator.gather_internal_vars(problem, internal_vars_volume, sv)
    inputs = [cells_sol_flat, src.physical_quad_points, src.shape_grads,
              src.JxW, src.v_grads_JxW, *internal_vars_per_cell]

    nse = problem.csr_nse
    num_cells = problem.num_cells
    bs, num_cuts, n_padded = _batching(num_cells)
    carry = np.zeros(nse + 1, dtype=cells_sol_flat.dtype)

    # Slot-sort: from StaticVars leaves (traced) or trace-time constants.
    if sv is not None:
        perm_arr, sorted_arr = sv.csr_vol_perm, sv.csr_vol_sorted_slots
    else:
        perm_np, sorted_np = _vol_slot_sort(problem)
        perm_arr, sorted_arr = np.asarray(perm_np), np.asarray(sorted_np)

    if num_cuts <= 1:
        _, jacs = vmap_fn(*inputs)
        return carry + jax.ops.segment_sum(
            jacs.reshape(-1)[perm_arr], sorted_arr,
            num_segments=nse + 1, indices_are_sorted=True)

    pad = n_padded - num_cells
    padded_inputs = jax.tree_util.tree_map(lambda x: _pad_axis0(x, pad), inputs)
    main_inputs = jax.tree_util.tree_map(
        lambda x: x.reshape(num_cuts, bs, *x.shape[1:]), padded_inputs)

    def scan_body(carry, batch):
        binputs, bperm, bsorted = batch
        _, jacs = vmap_fn(*binputs)
        vals = jacs.reshape(-1)
        carry = carry + jax.ops.segment_sum(
            vals[bperm], bsorted, num_segments=nse + 1, indices_are_sorted=True)
        return carry, None

    carry, _ = jax.lax.scan(scan_body, carry, (main_inputs, perm_arr, sorted_arr))
    return carry


def _get_J_csr(problem: 'Problem',
               sol_list: List[np.ndarray],
               internal_vars: InternalVars,
               sv=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the Jacobian directly in CSR form, bypassing BCOO.

    Returns ``(csr_data, csr_indptr, csr_indices)`` — the cuDSS-ready triple.
    The volume Jacobian is reduced into the CSR ``data`` batch-by-batch (see
    :func:`_assemble_volume_csr_data`) so the full element-Jacobian array is
    never materialized; surface (boundary) Jacobians are small and are added on
    top via a single ``segment_sum``. ``csr_indptr`` / ``csr_indices`` are the
    static structure from :meth:`Problem._build_csr_assembly_structure`.

    This is the CSR-direct counterpart of :func:`get_jacobian`. No Dirichlet BCs are
    applied here (see the BC-aware CSR path).
    """
    src = sv if sv is not None else problem
    cells_sol_list = [sol[cells] for cells, sol in zip(src.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)

    # Volume: memory-efficient per-batch accumulation (length nse + 1).
    csr_data = _assemble_volume_csr_data(problem, cells_sol_flat, internal_vars.volume_vars, sv)

    # Surface: per-boundary face Jacobians (small) added via their CSR slots.
    # The slot-sorted permutation (precomputed) keeps the reduction deterministic.
    _, cells_jac_face_flat = compute_face(problem, cells_sol_flat, True, internal_vars.surface_vars, sv)
    if cells_jac_face_flat:
        face_vals = np.concatenate([fj.reshape(-1) for fj in cells_jac_face_flat])
        csr_data = csr_data + jax.ops.segment_sum(
            face_vals[src.csr_face_perm], src.csr_face_sorted_slots,
            num_segments=problem.csr_nse + 1, indices_are_sorted=True)

    return csr_data[:problem.csr_nse], src.csr_indptr, src.csr_indices


def _assemble_volume_res_and_J(problem: 'Problem',
                               cells_sol_flat: np.ndarray,
                               internal_vars_volume: Tuple[np.ndarray, ...],
                               sv=None) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble the volume residual AND CSR Jacobian from a single jacfwd pass.

    ``_value_and_jacfwd`` already returns both the element residual ``y`` and
    its Jacobian; this reduces ``y`` into the global residual accumulator and
    the Jacobian into the CSR accumulator in one scan, so the volume kernel is
    evaluated once per Newton iterate instead of twice (residual-only + jacfwd).

    Returns ``(res_carry, J_carry)`` of length ``num_total_dofs + 1`` and
    ``nse + 1`` respectively (trailing entry = discard bucket for padded cells /
    filtered-out entries).
    """
    var_kinds = tuple(classify_volume_var(problem, v) for v in internal_vars_volume)
    kernel = create_volume_kernel(problem, var_kinds)

    def kernel_res_jac(cell_sol_flat, *args):
        return _value_and_jacfwd(lambda s: kernel(s, *args), cell_sol_flat)
    vmap_fn = jax.vmap(kernel_res_jac)

    src = sv if sv is not None else problem
    internal_vars_per_cell = Operator.gather_internal_vars(problem, internal_vars_volume, sv)
    inputs = [cells_sol_flat, src.physical_quad_points, src.shape_grads,
              src.JxW, src.v_grads_JxW, *internal_vars_per_cell]

    nse = problem.csr_nse
    ndof_total = problem.num_total_dofs_all_vars
    num_cells = problem.num_cells
    bs, num_cuts, n_padded = _batching(num_cells)

    J_carry = np.zeros(nse + 1, dtype=cells_sol_flat.dtype)
    R_carry = np.zeros(ndof_total + 1, dtype=cells_sol_flat.dtype)

    # Slot/DOF sorts: StaticVars leaves (traced) or trace-time constants.
    if sv is not None:
        jperm, jsorted = sv.csr_vol_perm, sv.csr_vol_sorted_slots
        rperm, rsorted = sv.res_vol_perm, sv.res_vol_sorted_dofs
    else:
        _jp, _js = _vol_slot_sort(problem)
        _rp, _rs = _res_vol_slot_sort(problem)
        jperm, jsorted = np.asarray(_jp), np.asarray(_js)
        rperm, rsorted = np.asarray(_rp), np.asarray(_rs)

    if num_cuts <= 1:
        y, jac = vmap_fn(*inputs)
        R_carry = R_carry + jax.ops.segment_sum(
            y.reshape(-1)[rperm], rsorted,
            num_segments=ndof_total + 1, indices_are_sorted=True)
        J_carry = J_carry + jax.ops.segment_sum(
            jac.reshape(-1)[jperm], jsorted,
            num_segments=nse + 1, indices_are_sorted=True)
        return R_carry, J_carry

    # Pad cells; padded entries route to the discard buckets (nse / ndof_total).
    pad = n_padded - num_cells
    main_inputs = jax.tree_util.tree_map(
        lambda x: _pad_axis0(x, pad).reshape(num_cuts, bs, *x.shape[1:]), inputs)

    xs = (main_inputs, jperm, jsorted, rperm, rsorted)

    def scan_body(carry, batch):
        Rc, Jc = carry
        binputs, bjperm, bjsorted, brperm, brsorted = batch
        y, jac = vmap_fn(*binputs)
        Jc = Jc + jax.ops.segment_sum(
            jac.reshape(-1)[bjperm], bjsorted, num_segments=nse + 1, indices_are_sorted=True)
        Rc = Rc + jax.ops.segment_sum(
            y.reshape(-1)[brperm], brsorted, num_segments=ndof_total + 1, indices_are_sorted=True)
        return (Rc, Jc), None

    (R_carry, J_carry), _ = jax.lax.scan(scan_body, (R_carry, J_carry), xs)
    return R_carry, J_carry


def _get_res_J_csr(problem: 'Problem',
                   sol_list: List[np.ndarray],
                   internal_vars: InternalVars,
                   sv=None) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Fused residual + CSR Jacobian (single kernel pass per element).

    Returns ``(res_list, csr_data, csr_indptr, csr_indices)``. No Dirichlet BCs
    applied (see :func:`create_res_J_bc_csr_parametric`).
    """
    src = sv if sv is not None else problem
    cells_sol_list = [sol[cells] for cells, sol in zip(src.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)

    R_carry, J_carry = _assemble_volume_res_and_J(problem, cells_sol_flat, internal_vars.volume_vars, sv)

    # Surface: one jacfwd pass yields both the face residual (values) and face
    # Jacobian; add each to its accumulator.
    face_res, face_jacs = compute_face(problem, cells_sol_flat, True, internal_vars.surface_vars, sv)
    if face_jacs:
        fjv = np.concatenate([fj.reshape(-1) for fj in face_jacs])
        J_carry = J_carry + jax.ops.segment_sum(
            fjv[src.csr_face_perm], src.csr_face_sorted_slots,
            num_segments=problem.csr_nse + 1, indices_are_sorted=True)
        frv = np.concatenate([fr.reshape(-1) for fr in face_res])
        R_carry = R_carry + jax.ops.segment_sum(
            frv[src.res_face_perm], src.res_face_sorted_dofs,
            num_segments=problem.num_total_dofs_all_vars + 1, indices_are_sorted=True)

    res_list = problem.unflatten_fn_sol_list(R_carry[:problem.num_total_dofs_all_vars])
    return res_list, J_carry[:problem.csr_nse], src.csr_indptr, src.csr_indices


def get_jacobian(problem: 'Problem',
                 sol_list: List[np.ndarray],
                 internal_vars: InternalVars) -> sparse.BCOO:
    """Assemble the global tangent (Jacobian) as a sparse ``BCOO`` matrix.

    Companion to :func:`get_res` (which assembles the global residual): this
    assembles the element tangents into the global Jacobian **without applying
    Dirichlet boundary conditions**, returned as a JAX ``BCOO``. It is the entry
    point for callers that need the raw assembled operator — e.g. building the
    material/geometric stiffness pair ``(K, K_g)`` for the linear-buckling
    eigensolver (:func:`feax.solvers.eigen.create_linear_buckling_solver`).

    For the solver stack, prefer the CSR-direct, BC-applied assembly
    (:func:`create_J_bc_csr_function`); for cheap statistics without
    materializing the matrix, use :func:`get_jacobian_info`.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and physics definitions.
    sol_list : list of np.ndarray
        Solution arrays for each variable.
    internal_vars : InternalVars
        Container with material properties and loading parameters.

    Returns
    -------
    sparse.BCOO
        The assembled global Jacobian with no boundary conditions applied,
        shape ``(num_total_dofs_all_vars, num_total_dofs_all_vars)``.

    Notes
    -----
    With a JIT-compiled cuDSS solver, materializing this BCOO outside the JIT
    boundary can contend with the backend's GPU memory; assemble inside the
    traced region (or use the CSR-direct path) in that setting.
    """
    V = _assemble_jacobian_values(problem, sol_list, internal_vars)

    # Use pre-computed filtered indices (JIT-compatible)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    if hasattr(problem, 'filter_indices') and problem.filter_indices is not None:
        # Use pre-computed integer indices for filtering (JIT-compatible)
        filtered_data = V[problem.filter_indices]
        filtered_indices = np.stack([problem.I_filtered, problem.J_filtered], axis=1)
        J = sparse.BCOO((filtered_data, filtered_indices), shape=shape)
    else:
        # FULL - use pre-filtered indices (which are same as I, J for FULL)
        indices = np.stack([problem.I_filtered, problem.J_filtered], axis=1)
        J = sparse.BCOO((V, indices), shape=shape)

    return J


def get_jacobian_info(problem: 'Problem',
                      sol_list: List[np.ndarray],
                      internal_vars: InternalVars) -> dict:
    """Get Jacobian matrix information without full matrix construction.

    This function provides safe access to Jacobian statistics that works
    correctly with JIT-compiled solvers and cuDSS backend. Unlike the
    full assembly in :func:`get_jacobian`, this does not cause GPU memory conflicts.

    Parameters
    ----------
    problem : Problem
        The finite element problem definition.
    sol_list : list of np.ndarray
        Solution arrays for each variable.
    internal_vars : InternalVars
        Internal variables container.

    Returns
    -------
    dict
        Dictionary containing:
        - 'nnz': Number of non-zero entries (int)
        - 'shape': Matrix shape (tuple)
        - 'matrix_view': Matrix storage format (MatrixView enum)

    Examples
    --------
    >>> info = get_jacobian_info(problem, sol_list, internal_vars)
    >>> print(f"Jacobian NNZ: {info['nnz']:,}")
    >>> print(f"Matrix view: {info['matrix_view'].name}")

    Notes
    -----
    This function is safe to call from user code and does not interfere
    with JIT-compiled solvers using cuDSS backend.
    """
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    # Calculate NNZ based on matrix view
    if hasattr(problem, 'filter_indices') and problem.filter_indices is not None:
        nnz = len(problem.filter_indices)
    elif hasattr(problem, 'I_filtered'):
        nnz = len(problem.I_filtered)
    else:
        nnz = len(problem.I)

    return {
        'nnz': nnz,
        'shape': shape,
        'matrix_view': problem.matrix_view if hasattr(problem, 'matrix_view') else None
    }


def get_res(problem: 'Problem',
            sol_list: List[np.ndarray],
            internal_vars: InternalVars,
            sv=None) -> List[np.ndarray]:
    """Compute residual vector with separated internal variables.
    
    Assembles the global residual vector by evaluating the weak form at the
    current solution state. Includes contributions from both volume and
    surface integrals.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and physics definitions.
    sol_list : list of np.ndarray
        Solution arrays for each variable.
        Each array has shape (num_total_nodes, vec).
    internal_vars : InternalVars
        Container with material properties and loading parameters.
    
    Returns
    -------
    list of np.ndarray
        Residual arrays for each solution variable.
        Each array has shape (num_total_nodes, vec).
    
    Examples
    --------
    >>> residual = get_res(problem, [solution], internal_vars)
    >>> res_norm = np.linalg.norm(jax.flatten_util.ravel_pytree(residual)[0])
    >>> print(f"Residual norm: {res_norm}")
    
    Notes
    -----
    The residual represents the imbalance in the weak form equations.
    For converged solutions, the residual should be near zero.
    """
    src = sv if sv is not None else problem
    cells_sol_list = [sol[cells] for cells, sol in zip(src.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)

    # Compute weak form values from volume integrals
    weak_form_flat = split_and_compute_cell(problem, cells_sol_flat, False, internal_vars.volume_vars, sv)

    # Add weak form values from surface integrals
    weak_form_face_flat = compute_face(problem, cells_sol_flat, False, internal_vars.surface_vars, sv)

    return compute_residual_vars_helper(problem, weak_form_flat, weak_form_face_flat, sv)


def create_J_bc_csr_function(
    problem: 'Problem', bc: 'DirichletBC', symmetric: bool = True,
) -> Callable[[np.ndarray, InternalVars], 'CSRMatrix']:
    """Assemble the BC-applied Jacobian directly as a deduplicated CSRMatrix.

    Returns ``(sol_flat, internal_vars) -> CSRMatrix`` that assembles the
    BC-applied Jacobian straight into deduplicated CSR form — no BCOO, no
    per-solve ``sum_duplicates`` sort — using the slot map precomputed in
    :meth:`Problem._build_csr_assembly_structure`, ready for direct backends
    (cuDSS/cholmod/umfpack/spsolve) without conversion.
    """
    from feax.DCboundary import apply_boundary_to_J_csr
    from feax.csr import CSRMatrix

    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    def J_bc_func(sol_flat, internal_vars: InternalVars):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        csr_data, _, _ = _get_J_csr(problem, sol_list, internal_vars)
        csr_data = apply_boundary_to_J_csr(bc, problem, csr_data, symmetric=symmetric)
        return CSRMatrix(csr_data, problem.csr_indptr, problem.csr_indices, shape)

    return J_bc_func


def create_J_bc_csr_parametric(problem: 'Problem', symmetric: bool = True) -> Callable:
    """Parametric form of :func:`create_J_bc_csr_function`.

    Like :func:`create_J_bc_csr_function` but takes ``bc`` as an explicit third
    argument so it traces through the BC pytree (vmap / per-step BC values).
    """
    from feax.DCboundary import apply_boundary_to_J_csr
    from feax.csr import CSRMatrix

    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    def J_bc_func(sol_flat, internal_vars: InternalVars, bc, sv=None):
        src = sv if sv is not None else problem
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        csr_data, _, _ = _get_J_csr(problem, sol_list, internal_vars, sv)
        csr_data = apply_boundary_to_J_csr(bc, problem, csr_data, symmetric=symmetric, sv=sv)
        return CSRMatrix(csr_data, src.csr_indptr, src.csr_indices, shape)

    return J_bc_func


def create_res_J_bc_csr_parametric(problem: 'Problem', symmetric: bool = True) -> Callable:
    """Fused BC-applied residual + CSR Jacobian, ``bc`` as an explicit argument.

    Returns ``(sol_flat, internal_vars, bc) -> (res_bc, J_bc_csr)`` computed from
    a single element-kernel pass (see :func:`_get_res_J_csr`) — used by the
    Newton step so it does not evaluate the volume kernel twice (once for the
    residual, once for the Jacobian).
    """
    from feax.DCboundary import apply_boundary_to_J_csr, apply_boundary_to_res
    from feax.csr import CSRMatrix

    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    def res_J_func(sol_flat, internal_vars: InternalVars, bc, sv=None):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        res_list, csr_data, indptr, indices = _get_res_J_csr(problem, sol_list, internal_vars, sv)
        res_flat = jax.flatten_util.ravel_pytree(res_list)[0]
        res_bc = apply_boundary_to_res(bc, res_flat, sol_flat)
        csr_bc = apply_boundary_to_J_csr(bc, problem, csr_data, symmetric=symmetric, sv=sv)
        return res_bc, CSRMatrix(csr_bc, indptr, indices, shape)

    return res_J_func


def _residual_flat(problem: 'Problem', sol_flat: np.ndarray,
                   internal_vars: InternalVars, sv=None) -> np.ndarray:
    """The un-Dirichlet-applied global residual as a flat vector of ``sol_flat``."""
    return jax.flatten_util.ravel_pytree(
        get_res(problem, problem.unflatten_fn_sol_list(sol_flat), internal_vars, sv))[0]


def create_matfree_res_J_parametric(problem: 'Problem', symmetric: bool = True) -> Callable:
    """Matrix-free counterpart of :func:`create_res_J_bc_csr_parametric` (Krylov).

    Returns ``(sol_flat, internal_vars, bc) -> (res_bc, J_matvec)`` where
    ``J_matvec(v)`` applies the BC-eliminated tangent **without assembling it**:
    the bulk action ``K @ v`` is a forward-mode ``jax.jvp`` of the residual, and
    the Dirichlet rows/columns are handled by masking. It reproduces
    ``apply_boundary_to_J_csr(_get_J_csr(...)) @ v`` exactly. For symmetric BC the
    operator is symmetric (``Jᵀ = J``), so the same matvec serves the adjoint.

    This is the operator the Krylov (cg/bicgstab/gmres) solvers consume — they
    never need the matrix entries, only this matvec, so the CSR assembly is
    skipped entirely on the iterative path.
    """
    from feax.DCboundary import apply_boundary_to_res

    def build(sol_flat, internal_vars: InternalVars, bc, sv=None):
        res_flat = _residual_flat(problem, sol_flat, internal_vars, sv)
        res_bc = apply_boundary_to_res(bc, res_flat, sol_flat)

        def matvec(v):
            # Symmetric elimination zeros BC columns by masking the input.
            v_in = np.where(bc.bc_mask, 0.0, v) if symmetric else v
            _, Kv = jax.jvp(
                lambda s: _residual_flat(problem, s, internal_vars, sv),
                (sol_flat,), (v_in,))
            # BC rows become identity: J_bc @ v has v on BC rows, K@v on free rows.
            return np.where(bc.bc_mask, v, Kv)

        return res_bc, matvec

    return build


def create_matfree_Kt_parametric(problem: 'Problem') -> Callable:
    """Matrix-free ``K_bulk^T`` (un-eliminated residual transpose) for the
    symmetric-BC adjoint correction.

    Returns ``(sol_flat, internal_vars) -> (w -> K_bulk^T @ w)`` via reverse-mode
    ``jax.vjp`` of the residual — used to recover the correct ``bc_vals``
    gradient without assembling the bulk Jacobian.
    """
    def build(sol_flat, internal_vars: InternalVars, sv=None):
        _, vjp_fn = jax.vjp(
            lambda s: _residual_flat(problem, s, internal_vars, sv), sol_flat)
        return lambda w: vjp_fn(w)[0]

    return build


def create_res_bc_function(problem: 'Problem', bc: 'DirichletBC') -> Callable[[np.ndarray, InternalVars], np.ndarray]:
    """Create residual function with Dirichlet BC applied.
    
    Returns a function that computes the residual vector with Dirichlet
    boundary conditions enforced. The BC application zeros out residuals
    at constrained DOFs.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem definition.
    bc : DirichletBC
        Dirichlet boundary condition specifications.
    
    Returns
    -------
    Callable
        Function with signature (sol_flat, internal_vars) -> np.ndarray
        that returns the BC-modified residual vector.
    
    Notes
    -----
    The returned function is used in Newton solvers to find solutions
    that satisfy both the weak form equations and boundary conditions.
    """
    from feax.DCboundary import apply_boundary_to_res

    def res_bc_func(sol_flat, internal_vars: InternalVars):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(problem, sol_list, internal_vars)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)

    return res_bc_func


def create_res_bc_parametric(problem: 'Problem') -> Callable:
    """Create a residual function that takes ``bc`` as an explicit argument.

    Unlike :func:`create_res_bc_function` which captures ``bc`` in a closure,
    this version accepts ``bc`` as a third argument.  This enables a single
    JIT-compiled function to be reused across time steps where only the
    prescribed BC values change (same DOF locations, different values).

    Parameters
    ----------
    problem : Problem
        The finite element problem definition.

    Returns
    -------
    Callable
        Function with signature ``(sol_flat, internal_vars, bc) -> np.ndarray``.
    """
    from feax.DCboundary import apply_boundary_to_res

    def res_bc_func(sol_flat, internal_vars: InternalVars, bc, sv=None):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(problem, sol_list, internal_vars, sv)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)

    return res_bc_func


# ============================================================================
# Energy integration utility
# ============================================================================

def create_energy_fn(problem) -> Callable:
    """Create a total-energy integration function from a feax Problem.

    Builds a pure JAX function that integrates the problem's energy density
    over the domain::

        E(u) = ∫ ψ(∇u, *internal_vars) dΩ

    The energy density is obtained from ``problem.get_energy_density()``. This
    is the same density the residual assembler differentiates (``tensor_map =
    jax.grad(energy_density)``) — :func:`create_energy_fn` exposes the *scalar*
    energy itself, which is useful for objective evaluation (e.g. compliance /
    stored energy) and energy-based post-processing.

    Parameters
    ----------
    problem : feax.Problem
        A problem defining ``get_energy_density()`` (must return non-None).

    Returns
    -------
    energy : callable
        ``energy(u_flat)`` or ``energy(u_flat, internal_vars)`` returning a
        scalar. Without ``internal_vars`` the density receives only ``∇u``;
        with it, each volume variable is interpolated to quadrature points
        (node-based via shape functions, cell-based by broadcast) and passed as
        extra arguments ``ψ(∇u, var0_q, var1_q, …)``.

    Raises
    ------
    ValueError
        If ``problem.get_energy_density()`` returns None.
    """
    psi_fn = problem.get_energy_density()
    if psi_fn is None:
        raise ValueError(
            f"{type(problem).__name__}.get_energy_density() returned None. "
            "Define get_energy_density() to use create_energy_fn()."
        )

    fe0 = problem.fes[0]
    cells = problem.cells_list[0]
    sg = fe0.shape_grads      # (num_cells, num_quads, num_nodes, dim)
    jxw = fe0.JxW              # (num_cells, num_quads)
    sv = fe0.shape_vals        # (num_quads, num_nodes_per_cell)
    vec = fe0.vec
    num_cells = fe0.num_cells
    num_quads = fe0.num_quads

    def _interpolate_volume_vars(internal_vars):
        """Interpolate volume variables to quadrature points.

        Returns a list of arrays each with shape ``(num_cells, num_quads)``.
        """
        result = []
        for var in internal_vars.volume_vars:
            if var.ndim == 1:
                if var.shape[0] == num_cells:
                    result.append(np.tile(var[:, None], (1, num_quads)))
                else:
                    var_cell = var[cells]
                    result.append(np.einsum('qn,cn->cq', sv, var_cell))
            else:
                result.append(var)
        return result

    def energy(u_flat, internal_vars=None):
        u = u_flat.reshape(-1, vec)
        cell_u = u[cells]  # (num_cells, num_nodes_per_cell, vec)

        if internal_vars is None:
            def cell_energy(cell_sol, cell_sg, cell_jxw):
                u_grads = np.sum(
                    cell_sol[None, :, :, None] * cell_sg[:, :, None, :],
                    axis=1,
                )  # (num_quads, vec, dim)
                return np.sum(jax.vmap(lambda ug, w: psi_fn(ug) * w)(u_grads, cell_jxw))

            return np.sum(jax.vmap(cell_energy)(cell_u, sg, jxw))
        else:
            vol_vars_q = _interpolate_volume_vars(internal_vars)

            def cell_energy_iv(cell_sol, cell_sg, cell_jxw, *vars_c):
                u_grads = np.sum(
                    cell_sol[None, :, :, None] * cell_sg[:, :, None, :],
                    axis=1,
                )  # (num_quads, vec, dim)

                def quad_fn(ug, w, *vq):
                    return psi_fn(ug, *vq) * w

                return np.sum(jax.vmap(quad_fn)(u_grads, cell_jxw, *vars_c))

            return np.sum(jax.vmap(cell_energy_iv)(cell_u, sg, jxw, *vol_vars_q))

    return energy
