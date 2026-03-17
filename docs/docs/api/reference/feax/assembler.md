---
sidebar_label: assembler
title: feax.assembler
---

Assembler functions that work with Problem and InternalVars.

This module provides the main assembler API for finite element analysis with
separated internal variables. It handles the assembly of residual vectors and
Jacobian matrices for both volume and surface integrals, supporting various
physics kernels (Laplace, mass, surface, and universal).

## Operator Objects

```python
class Operator()
```

Element-level operator for quadrature-point computations.

Provides methods for interpolating solutions, computing gradients,
interpolating internal variables, and integrating over quadrature points.
Used internally by kernel functions to eliminate code duplication.

Parameters
----------
- **problem** (*Problem*): The finite element problem.
- **fe_index** (*int*): Index of the finite element variable (default 0).


#### eval

```python
def eval(cell_sol: np.ndarray, shape_vals: np.ndarray = None) -> np.ndarray
```

Interpolate nodal solution to quadrature points.

Parameters
----------
- **cell_sol** (*np.ndarray*): Nodal solution values, shape (num_nodes, vec).
- **shape_vals** (*np.ndarray, optional*): Shape function values. Uses volume shape functions if None.


Returns
-------
np.ndarray
    Solution at quadrature points, shape (num_quads, vec).

#### grad

```python
def grad(cell_sol: np.ndarray, cell_shape_grads: np.ndarray) -> np.ndarray
```

Compute solution gradient at quadrature points.

Parameters
----------
- **cell_sol** (*np.ndarray*): Nodal solution values, shape (num_nodes, vec).
- **cell_shape_grads** (*np.ndarray*): Shape function gradients, shape (num_quads, num_nodes, dim).


Returns
-------
np.ndarray
    Gradient at quadrature points, shape (num_quads, vec, dim).

#### hess

```python
def hess(cell_sol: np.ndarray, cell_shape_hessians: np.ndarray) -> np.ndarray
```

Compute solution Hessian (second spatial derivatives) at quadrature points.

Parameters
----------
- **cell_sol** (*np.ndarray*): Nodal solution values, shape (num_nodes, vec).
- **cell_shape_hessians** (*np.ndarray*): Shape function second derivatives in physical coordinates, shape (num_quads, num_nodes, dim, dim).


Returns
-------
np.ndarray
    Hessian at quadrature points, shape (num_quads, vec, dim, dim).
    H[q, i, K, L] = sum_a u[a, i] * d²h_a/(dX_K dX_L) at quad point q.

#### interpolate\_var

```python
def interpolate_var(var: np.ndarray) -> np.ndarray
```

Interpolate a single internal variable to quadrature points.

Handles node-based (shape function interpolation), cell-based
(broadcast), and quad-based (pass-through) variables.

Parameters
----------
- **var** (*np.ndarray*): Internal variable. Shape determines interpolation strategy: - scalar: broadcast to all quad points - (num_nodes,): node-based, interpolate via shape functions - (1,) or cell-based: broadcast to all quad points - (num_quads,): quad-based, pass through


Returns
-------
np.ndarray
    Values at quadrature points, shape (num_quads,).

#### interpolate\_vars

```python
def interpolate_vars(
        cell_internal_vars: Tuple[np.ndarray, ...]) -> List[np.ndarray]
```

Interpolate all internal variables to quadrature points.

Parameters
----------
- **cell_internal_vars** (*tuple of np.ndarray*): Internal variables for a single element.


Returns
-------
list of np.ndarray
    Interpolated values at quadrature points.

#### integrate\_grad

```python
def integrate_grad(quad_values: np.ndarray,
                   cell_v_grads_JxW: np.ndarray) -> np.ndarray
```

Integrate tensor quad-point values against test function gradients.

Computes: sum_q sigma(q) : grad_v(q) * JxW(q)

Parameters
----------
- **quad_values** (*np.ndarray*): Physics values at quad points, shape (num_quads, vec, dim).
- **cell_v_grads_JxW** (*np.ndarray*): Pre-multiplied test function gradients × JxW, shape (num_quads, num_nodes, vec, dim).


Returns
-------
np.ndarray
    Element contribution, shape (num_nodes, vec).

#### integrate\_val

```python
def integrate_val(quad_values: np.ndarray,
                  cell_JxW: np.ndarray,
                  shape_vals: np.ndarray = None) -> np.ndarray
```

Integrate scalar/vector quad-point values with shape functions.

Computes: sum_q f(q) * N(q) * JxW(q)

Parameters
----------
- **quad_values** (*np.ndarray*): Physics values at quad points, shape (num_quads, vec).
- **cell_JxW** (*np.ndarray*): Jacobian determinant × quadrature weights, shape (num_quads,).
- **shape_vals** (*np.ndarray, optional*): Shape function values. Uses volume shape functions if None.


Returns
-------
np.ndarray
    Element contribution, shape (num_nodes, vec).

#### gather\_internal\_vars

```python
@staticmethod
def gather_internal_vars(
        problem: 'Problem', internal_vars: Tuple[np.ndarray,
                                                 ...]) -> List[np.ndarray]
```

Gather global internal variables to per-cell format.

Transforms node-based variables from global (num_nodes,) arrays to
per-cell (num_cells, num_nodes_per_elem) arrays using element
connectivity. Cell-based and quad-based variables are passed through.

Parameters
----------
- **problem** (*Problem*): The finite element problem with connectivity information.
- **internal_vars** (*tuple of np.ndarray*): Global internal variables.


Returns
-------
list of np.ndarray
    Per-cell internal variables ready for vmapped kernels.

#### interpolate\_to\_quad\_points

```python
def interpolate_to_quad_points(var: np.ndarray, shape_vals: np.ndarray,
                               num_cells: int, num_quads: int) -> np.ndarray
```

Interpolate node-based or cell-based values to quadrature points.

.. deprecated::
    Use :meth:`Operator.interpolate_var` instead.

This function handles three cases:
1. Node-based: shape (num_nodes,) -&gt; interpolate using shape functions
2. Cell-based: shape (num_cells,) -&gt; broadcast to all quad points in cell
3. Quad-based: shape (num_cells, num_quads) -&gt; pass through (legacy)

Parameters
----------
- **var** (*np.ndarray*): Variable to interpolate. Can be: - (num_nodes,) for node-based - (num_cells,) for cell-based - (num_cells, num_quads) for quad-based (legacy)
- **shape_vals** (*np.ndarray*): Shape function values at quadrature points, shape (num_quads, num_nodes)
- **num_cells** (*int*): Number of cells/elements
- **num_quads** (*int*): Number of quadrature points per cell


Returns
-------
np.ndarray
    Values at quadrature points, shape (num_quads,)

#### get\_laplace\_kernel

```python
def get_laplace_kernel(problem: 'Problem', tensor_map: Callable) -> Callable
```

Create Laplace kernel function for gradient-based physics.

The Laplace kernel handles gradient-based terms in the weak form, such as
those arising in elasticity, heat conduction, and diffusion problems. It
implements the integral term: ∫ σ(∇u) : ∇v dΩ where σ is the stress/flux
tensor computed from the gradient.

Parameters
----------
- **problem** (*Problem*): The finite element problem containing mesh and element information.
- **tensor_map** (*Callable*): Function that maps gradient tensor to stress/flux tensor. Signature: (u_grad: ndarray, *internal_vars) -&gt; ndarray where u_grad has shape (vec, dim) and returns (vec, dim).


Returns
-------
Callable
    Laplace kernel function that computes the contribution to the weak form
    from gradient-based physics.

#### get\_mass\_kernel

```python
def get_mass_kernel(problem: 'Problem', mass_map: Callable) -> Callable
```

Create mass kernel function for non-gradient terms.

The mass kernel handles terms without derivatives in the weak form, such as
mass matrices, reaction terms, or body forces. It implements the integral
term: ∫ m(u, x) · v dΩ where m is a mass-like term.

Parameters
----------
- **problem** (*Problem*): The finite element problem containing mesh and element information.
- **mass_map** (*Callable*): Function that computes the mass term. Signature: (u: ndarray, x: ndarray, *internal_vars) -&gt; ndarray where u has shape (vec,), x has shape (dim,), and returns (vec,).


Returns
-------
Callable
    Mass kernel function that computes the contribution to the weak form
    from non-gradient physics.

#### get\_surface\_kernel

```python
def get_surface_kernel(problem: 'Problem', surface_map: Callable) -> Callable
```

Create surface kernel function for boundary integrals.

The surface kernel handles boundary integrals in the weak form, such as
surface tractions, pressures, or fluxes. It implements the integral term:
∫ t(u, x) · v dΓ where t is the surface load/flux.

Parameters
----------
- **problem** (*Problem*): The finite element problem containing mesh and element information.
- **surface_map** (*Callable*): Function that computes the surface traction/flux. Signature: (u: ndarray, x: ndarray, *internal_vars) -&gt; ndarray where u has shape (vec,), x has shape (dim,), and returns (vec,).


Returns
-------
Callable
    Surface kernel function that computes the contribution to the weak form
    from boundary loads/fluxes.

#### create\_volume\_kernel

```python
def create_volume_kernel(problem: 'Problem') -> Callable
```

Create unified volume kernel combining all volume physics.

This function creates a kernel that combines contributions from all volume
integral terms: Laplace (gradient-based), mass (non-gradient), universal
(custom), and weak form kernels. The resulting kernel is used for both
residual and Jacobian assembly.

Parameters
----------
- **problem** (*Problem*): The finite element problem that may define get_tensor_map(), get_mass_map(), get_weak_form(), and/or get_universal_kernel() methods.


Returns
-------
Callable
    Combined volume kernel function that sums contributions from all
    applicable physics kernels.

Notes
-----
The kernel checks for the existence of each physics method in the problem
and only includes contributions from those that are defined. This allows
for flexible problem definitions with any combination of physics terms.

For multi-variable problems, the priority order is:
1. ``get_universal_kernel()`` — full low-level control
2. ``get_weak_form()`` — high-level quad-point physics definition

#### create\_surface\_kernel

```python
def create_surface_kernel(problem: 'Problem', surface_index: int) -> Callable
```

Create unified surface kernel for a specific boundary.

This function creates a kernel that combines contributions from standard
surface maps and universal surface kernels for a specific boundary surface
identified by surface_index.

Parameters
----------
- **problem** (*Problem*): The finite element problem that may define get_surface_maps() and/or get_universal_kernels_surface() methods.
- **surface_index** (*int*): Index identifying which boundary surface this kernel is for. Corresponds to the index in problem.location_fns.


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

#### split\_and\_compute\_cell

```python
def split_and_compute_cell(
        problem: 'Problem', cells_sol_flat: np.ndarray, jac_flag: bool,
        internal_vars_volume: Tuple[np.ndarray, ...]) -> Any
```

Compute volume integrals for residual or Jacobian assembly.

This function evaluates volume integrals over all elements, optionally
computing the Jacobian via forward-mode automatic differentiation. It
uses batching to manage memory for large meshes.

Parameters
----------
- **problem** (*Problem*): The finite element problem containing mesh and quadrature data.
- **cells_sol_flat** (*np.ndarray*): Flattened solution values at element nodes. Shape: (num_cells, num_nodes * vec).
- **jac_flag** (*bool*): If True, compute both values and Jacobian. If False, compute only values.
- **internal_vars_volume** (*tuple of np.ndarray*): Material properties at quadrature points for each variable. Each array has shape (num_cells, num_quads).


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

#### compute\_face

```python
def compute_face(problem: 'Problem', cells_sol_flat: np.ndarray,
                 jac_flag: bool,
                 internal_vars_surfaces: List[Tuple[np.ndarray, ...]]) -> Any
```

Compute surface integrals for residual or Jacobian assembly.

This function evaluates surface integrals over all boundary faces,
optionally computing the Jacobian via forward-mode automatic differentiation.

Parameters
----------
- **problem** (*Problem*): The finite element problem containing boundary information.
- **cells_sol_flat** (*np.ndarray*): Flattened solution values at element nodes. Shape: (num_cells, num_nodes * vec).
- **jac_flag** (*bool*): If True, compute both values and Jacobian. If False, compute only values.
- **internal_vars_surfaces** (*list of tuple of np.ndarray*): Surface variables for each boundary. Each entry corresponds to one boundary surface and contains arrays with shape (num_surface_faces, num_face_quads).


Returns
-------
list of np.ndarray or list of tuple
    If jac_flag is False: list of weak form values for each boundary.
    If jac_flag is True: list of (values, jacobian) tuples for each boundary.

Notes
-----
Each boundary surface can have different loading conditions or physics,
handled through separate surface kernels and internal variables.

#### compute\_residual\_vars\_helper

```python
def compute_residual_vars_helper(
        problem: 'Problem', weak_form_flat: np.ndarray,
        weak_form_face_flat: List[np.ndarray]) -> List[np.ndarray]
```

Assemble residual from element and face contributions.

This helper function assembles the global residual vector by accumulating
contributions from volume and surface integrals at the appropriate nodes.

Parameters
----------
- **problem** (*Problem*): The finite element problem containing connectivity information.
- **weak_form_flat** (*np.ndarray*): Flattened weak form values from volume integrals. Shape: (num_cells, num_dofs_per_cell).
- **weak_form_face_flat** (*list of np.ndarray*): Weak form values from surface integrals for each boundary. Each array has shape (num_boundary_faces, num_dofs_per_face).


Returns
-------
list of np.ndarray
    Global residual for each solution variable.
    Each array has shape (num_total_nodes, vec).

Notes
-----
Uses JAX&#x27;s at[].add() for scatter-add operations to accumulate
contributions from multiple elements sharing the same nodes.

#### get\_jacobian\_info

```python
def get_jacobian_info(problem: 'Problem', sol_list: List[np.ndarray],
                      internal_vars: InternalVars) -> dict
```

Get Jacobian matrix information without full matrix construction.

This function provides safe access to Jacobian statistics that works
correctly with JIT-compiled solvers and cuDSS backend. Unlike the
internal _get_J function, this does not cause GPU memory conflicts.

Parameters
----------
- **problem** (*Problem*): The finite element problem definition.
- **sol_list** (*list of np.ndarray*): Solution arrays for each variable.
- **internal_vars** (*InternalVars*): Internal variables container.


Returns
-------
dict
    Dictionary containing:
    - &#x27;nnz&#x27;: Number of non-zero entries (int)
    - &#x27;shape&#x27;: Matrix shape (tuple)
    - &#x27;matrix_view&#x27;: Matrix storage format (MatrixView enum)

Examples
--------
```python
>>> info = get_jacobian_info(problem, sol_list, internal_vars)
>>> print(f&quot;Jacobian NNZ: {`info[&#x27;nnz&#x27;]:,`}&quot;)
>>> print(f&quot;Matrix view: {`info[&#x27;matrix_view&#x27;].name`}&quot;)
```

Notes
-----
This function is safe to call from user code and does not interfere
with JIT-compiled solvers using cuDSS backend.

#### get\_res

```python
def get_res(problem: 'Problem', sol_list: List[np.ndarray],
            internal_vars: InternalVars) -> List[np.ndarray]
```

Compute residual vector with separated internal variables.

Assembles the global residual vector by evaluating the weak form at the
current solution state. Includes contributions from both volume and
surface integrals.

Parameters
----------
- **problem** (*Problem*): The finite element problem containing mesh and physics definitions.
- **sol_list** (*list of np.ndarray*): Solution arrays for each variable. Each array has shape (num_total_nodes, vec).
- **internal_vars** (*InternalVars*): Container with material properties and loading parameters.


Returns
-------
list of np.ndarray
    Residual arrays for each solution variable.
    Each array has shape (num_total_nodes, vec).

Examples
--------
```python
>>> residual = get_res(problem, [solution], internal_vars)
>>> res_norm = np.linalg.norm(jax.flatten_util.ravel_pytree(residual)[0])
>>> print(f&quot;Residual norm: {`res_norm`}&quot;)
```

Notes
-----
The residual represents the imbalance in the weak form equations.
For converged solutions, the residual should be near zero.

#### create\_J\_bc\_function

```python
def create_J_bc_function(
    problem: 'Problem',
    bc: 'DirichletBC',
    symmetric: bool = True
) -> Callable[[np.ndarray, InternalVars], sparse.BCOO]
```

Create Jacobian function with Dirichlet BC applied.

Returns a function that computes the Jacobian matrix with Dirichlet
boundary conditions enforced. The BC application modifies the matrix
to enforce constraints.

Parameters
----------
- **problem** (*Problem*): The finite element problem definition.
- **bc** (*DirichletBC*): Dirichlet boundary condition specifications.
- **symmetric** (*bool, default True*): If True, use symmetric elimination (zero BC rows and columns). If False, use non-symmetric elimination (zero BC rows only, keep columns). Non-symmetric elimination preserves the K_10 coupling, which is needed for the incremental Newton approach without pre-applied BCs.


Returns
-------
Callable
    Function with signature (sol_flat, internal_vars) -&gt; sparse.BCOO
    that returns the BC-modified Jacobian matrix.

Notes
-----
The returned function is suitable for use in Newton solvers and
can be differentiated for sensitivity analysis.

#### create\_res\_bc\_function

```python
def create_res_bc_function(
        problem: 'Problem',
        bc: 'DirichletBC') -> Callable[[np.ndarray, InternalVars], np.ndarray]
```

Create residual function with Dirichlet BC applied.

Returns a function that computes the residual vector with Dirichlet
boundary conditions enforced. The BC application zeros out residuals
at constrained DOFs.

Parameters
----------
- **problem** (*Problem*): The finite element problem definition.
- **bc** (*DirichletBC*): Dirichlet boundary condition specifications.


Returns
-------
Callable
    Function with signature (sol_flat, internal_vars) -&gt; np.ndarray
    that returns the BC-modified residual vector.

Notes
-----
The returned function is used in Newton solvers to find solutions
that satisfy both the weak form equations and boundary conditions.

#### create\_J\_bc\_parametric

```python
def create_J_bc_parametric(problem: 'Problem',
                           symmetric: bool = True) -> Callable
```

Create a Jacobian function that takes ``bc`` as an explicit argument.

Unlike :func:`create_J_bc_function` which captures ``bc`` in a closure,
this version accepts ``bc`` as a third argument so that the function
traces through the BC pytree under ``jax.vmap``.

Parameters
----------
- **problem** (*Problem*): The finite element problem definition.
- **symmetric** (*bool, default True*): If True, use symmetric elimination.


Returns
-------
Callable
    Function with signature ``(sol_flat, internal_vars, bc) -&gt; sparse.BCOO``.

#### create\_res\_bc\_parametric

```python
def create_res_bc_parametric(problem: 'Problem') -> Callable
```

Create a residual function that takes ``bc`` as an explicit argument.

Unlike :func:`create_res_bc_function` which captures ``bc`` in a closure,
this version accepts ``bc`` as a third argument.  This enables a single
JIT-compiled function to be reused across time steps where only the
prescribed BC values change (same DOF locations, different values).

Parameters
----------
- **problem** (*Problem*): The finite element problem definition.


Returns
-------
Callable
    Function with signature ``(sol_flat, internal_vars, bc) -&gt; np.ndarray``.

