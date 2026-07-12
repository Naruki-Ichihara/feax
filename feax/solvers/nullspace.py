"""Nullspace constraints for matrix-free linear FE solves.

Supports the constant null mode of each connected component in a scalar
pure-Neumann Poisson problem. The implementation deliberately assumes that the
caller supplied a compatible load on every component; it never changes the
right-hand side.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as onp
import jax.numpy as jnp

from ..DCboundary import DirichletBC
from ..problem import Problem
from .options import AMGSolverOptions, KrylovSolverOptions


@dataclass(frozen=True)
class NullspaceConstraint:
    """Declaration of a linear-solver nullspace constraint.

    Use :meth:`constant_mean_zero` for the v1 pure-Neumann scalar Poisson
    gauge.  Arbitrary bases and gauges are intentionally deferred.
    """

    _kind: str = "constant_mean_zero"

    @classmethod
    def constant_mean_zero(cls) -> "NullspaceConstraint":
        """Require a zero mass-lumped physical mean on every mesh component."""
        return cls()


def _component_labels(cells: onp.ndarray, num_nodes: int) -> onp.ndarray:
    """Return dense labels for the mesh's cell-connectivity components."""
    parent = list(range(num_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for cell in cells:
        for node in cell[1:]:
            union(int(cell[0]), int(node))

    roots = [find(node) for node in range(num_nodes)]
    root_to_label = {root: label for label, root in enumerate(dict.fromkeys(roots))}
    return onp.asarray([root_to_label[root] for root in roots], dtype=onp.int32)


def component_labels(problem: Problem) -> onp.ndarray:
    """Return static node-to-component labels for the scalar FE mesh."""
    fe = problem.fes[0]
    return _component_labels(onp.asarray(fe.cells), int(fe.num_total_nodes))


def validate_nullspace_constraint(
    constraint: NullspaceConstraint,
    problem: Problem,
    bc: DirichletBC,
    solver_options,
    adjoint_solver_options,
    *,
    symmetric_elimination: bool,
) -> None:
    """Validate the construction-time pure-Neumann contract."""
    if not isinstance(constraint, NullspaceConstraint) or constraint._kind != "constant_mean_zero":
        raise ValueError("v1 only supports NullspaceConstraint.constant_mean_zero().")
    if len(problem.fes) != 1 or len(problem.vec) != 1 or int(problem.vec[0]) != 1:
        raise ValueError("Pure-Neumann v1 requires exactly one scalar field.")
    if len(bc.bc_rows) != 0:
        raise ValueError("Pure-Neumann v1 requires an empty DirichletBC.")
    if not symmetric_elimination:
        raise ValueError("Pure-Neumann v1 requires symmetric_elimination=True.")
    if isinstance(solver_options, AMGSolverOptions) or isinstance(adjoint_solver_options, AMGSolverOptions):
        raise ValueError("AMG nullspace support is deferred; use KrylovSolverOptions(solver='cg').")
    if not isinstance(solver_options, KrylovSolverOptions) or solver_options.solver != "cg":
        raise ValueError(
            "Pure-Neumann v1 requires explicit KrylovSolverOptions(solver='cg')."
        )
    if adjoint_solver_options is not None and (
        not isinstance(adjoint_solver_options, KrylovSolverOptions)
        or adjoint_solver_options.solver != "cg"
    ):
        raise ValueError(
            "Pure-Neumann v1 requires adjoint_solver_options to be CG when provided."
        )



def mass_lumped_weights(problem: Problem) -> jnp.ndarray:
    """Assemble scalar nodal weights ``m_i = integral(phi_i)``.

    This uses FEAX's shape-function table and physical quadrature weights, so
    it applies uniformly to every supported first- and second-order element.
    """
    fe = problem.fes[0]
    local_weights = jnp.einsum("qn,cq->cn", fe.shape_vals, fe.JxW)
    return jnp.zeros(fe.num_total_nodes, dtype=local_weights.dtype).at[fe.cells].add(local_weights)


def create_nullspace_ops(weights: jnp.ndarray, labels: onp.ndarray):
    """Build JAX-only quotient and per-component mass-gauge maps.

    Each connected component contributes one constant null mode.  ``labels`` is
    computed from static mesh topology at solver construction; all operations
    performed by the returned maps remain JAX-native and jit/vmap-safe.
    """
    component_ids = jnp.asarray(labels)
    num_components = int(onp.max(labels)) + 1

    def component_sum(vector):
        return jnp.zeros(num_components, dtype=vector.dtype).at[component_ids].add(vector)

    component_sizes = component_sum(jnp.ones_like(weights))
    component_masses = component_sum(weights)

    def project_quotient(vector):
        component_means = component_sum(vector) / component_sizes
        return vector - component_means[component_ids]

    def mass_zero(vector):
        component_means = component_sum(weights * vector) / component_masses
        return vector - component_means[component_ids]

    def mass_zero_transpose(vector):
        component_scales = component_sum(vector) / component_masses
        return vector - weights * component_scales[component_ids]

    return project_quotient, mass_zero, mass_zero_transpose


def projected_operator(operator: Callable, project_quotient: Callable) -> Callable:
    """Restrict a matrix-free operator to the constant-mode quotient spaces."""
    def operator_on_quotient(vector):
        return project_quotient(operator(project_quotient(vector)))
    return operator_on_quotient
