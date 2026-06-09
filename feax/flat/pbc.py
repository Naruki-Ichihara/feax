"""Periodic boundary condition implementation for finite element analysis.

This module provides functionality for enforcing periodic boundary conditions (PBCs)
in finite element computations, particularly for unit cell-based homogenization and
multiscale analysis. It handles the construction of prolongation matrices that map
between reduced (independent) and full degree-of-freedom spaces.
"""

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, List

import jax
import jax.numpy as np
import numpy as onp
from jax.experimental.sparse import BCOO

from feax.mesh import Mesh

from .unitcell import UnitCell


@dataclass
class PeriodicPairing:
    """Represents a single periodic boundary condition pair between master and slave regions.

    This class encapsulates the geometric and topological relationship between paired
    regions in a periodic boundary condition. It defines which nodes are considered
    'master' (independent) and 'slave' (dependent), along with the mapping function
    that relates their coordinates.

    Attributes:
        location_master (Callable[[onp.ndarray], bool]): Function that identifies points
            on the master side of the periodic boundary. Takes a point coordinate array
            and returns True if the point lies on the master boundary.
        location_slave (Callable[[onp.ndarray], bool]): Function that identifies points
            on the slave side of the periodic boundary. Takes a point coordinate array
            and returns True if the point lies on the slave boundary.
        mapping (Callable[[onp.ndarray], onp.ndarray]): Function that maps a point from
            the master boundary to its corresponding location on the slave boundary.
            Essential for establishing the periodic relationship.
        vec (int): Index of the degree-of-freedom component affected by this pairing.
            For vector problems: 0=x-component, 1=y-component, 2=z-component, etc.

    Example:
        >>> # Create a pairing for x-direction periodicity
        >>> pairing = PeriodicPairing(
        ...     location_master=lambda p: np.isclose(p[0], 0.0),  # Left face
        ...     location_slave=lambda p: np.isclose(p[0], 1.0),   # Right face
        ...     mapping=lambda p: p + np.array([1.0, 0.0, 0.0]),  # Translate by 1 in x
        ...     vec=0  # Apply to x-component of displacement
        ... )
    """

    location_master: Callable[[onp.ndarray], bool]
    location_slave: Callable[[onp.ndarray], bool]
    mapping: Callable[[onp.ndarray], onp.ndarray]
    vec: int


def prolongation_matrix(
    periodic_pairings: Iterable[PeriodicPairing], mesh: Mesh, vec: int, offset: int = 0
) -> BCOO:
    """Constructs the prolongation matrix P for applying periodic boundary conditions.

    The prolongation matrix P maps reduced (independent) degrees of freedom to the full
    set of DoFs, enforcing periodic constraints. This is essential for homogenization
    and multiscale analysis where opposite boundaries must have compatible displacements.

    The matrix enforces: u_slave = u_master for paired nodes, reducing the system size
    while maintaining periodicity. Given reduced DoFs ū, the full DoF vector is u = P @ ū.

    Args:
        periodic_pairings (Iterable[PeriodicPairing]): Collection of periodic constraint
            definitions specifying master/slave boundary pairs, their geometric mapping,
            and affected DoF components.
        mesh (Mesh): Finite element mesh containing node coordinates (points) and
            element connectivity information.
        vec (int): Number of degrees of freedom per node (e.g., 2 for 2D elasticity,
            3 for 3D elasticity).
        offset (int): Global DoF offset for constructing the matrix. Defaults to 0.

    Returns:
        BCOO: Prolongation matrix P with shape (N, M) where:
            - N = total number of DoFs before applying constraints
            - M = number of independent DoFs after applying constraints
            JAX BCOO sparse matrix format for compatibility with FEAX ecosystem.

    Raises:
        AssertionError: If master and slave boundary node counts don't match.
        ValueError: If the DoF reduction is inconsistent with the pairing structure.

    Example:
        >>> # Create prolongation matrix for 3D periodic unit cell
        >>> pairings = periodic_bc_3D(unit_cell, vec=3, dim=3)
        >>> P = prolongation_matrix(pairings, mesh, vec=3)
        >>> print(f"DoF reduction: {P.shape[0]} -> {P.shape[1]}")

    Note:
        The matrix construction process:
        1. Identifies master and slave nodes for each pairing
        2. Establishes point-to-point correspondence using geometric mapping
        3. Builds sparse matrix entries to enforce u_slave = u_master
        4. Results in a rectangular matrix for DoF space transformation
        5. Returns JAX BCOO sparse matrix for JAX transformations compatibility
    """
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []

    for bc in periodic_pairings:
        node_inds_A = onp.argwhere(jax.vmap(bc.location_master)(mesh.points)).reshape(
            -1
        )
        node_inds_B = onp.argwhere(jax.vmap(bc.location_slave)(mesh.points)).reshape(-1)
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-5
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = onp.linalg.norm(bc.mapping(point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
        vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32) * bc.vec

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)
        assert len(node_inds_A) == len(
            node_inds_B_ordered
        ), f"Mismatch in node pairing: {len(node_inds_A)} master nodes vs {len(node_inds_B_ordered)} slave nodes. Check your mapping."

    offset = 0
    inds_A_list = []
    inds_B_list = []
    for i in range(len(p_node_inds_list_A)):
        inds_A_list.append(
            onp.array(
                p_node_inds_list_A[i] * vec + p_vec_inds_list[i] + offset,
                dtype=onp.int32,
            )
        )
        inds_B_list.append(
            onp.array(
                p_node_inds_list_B[i] * vec + p_vec_inds_list[i] + offset,
                dtype=onp.int32,
            )
        )

    inds_A = onp.hstack(inds_A_list)
    inds_B = onp.hstack(inds_B_list)

    num_total_nodes = len(mesh.points)
    num_total_dofs = num_total_nodes * vec
    N = num_total_dofs
    M = num_total_dofs - len(inds_B)

    reduced_inds_map = onp.ones(num_total_dofs, dtype=onp.int32)
    reduced_inds_map[inds_B] = -(inds_A + 1)
    mask = reduced_inds_map == 1
    if onp.count_nonzero(mask) != M:
        raise ValueError(
            f"Inconsistent DoF reduction: expected {M} remaining DoFs "
            f"but found {onp.count_nonzero(mask)} unassigned entries in reduced_inds_map.\n"
            f"Possible cause: some mesh nodes are involved in multiple periodic pairings."
        )
    reduced_inds_map[mask] = onp.arange(M)

    I = []
    J = []
    V = []
    for i in range(num_total_dofs):
        I.append(i)
        V.append(1.0)
        if reduced_inds_map[i] < 0:
            J.append(reduced_inds_map[-reduced_inds_map[i] - 1])
        else:
            J.append(reduced_inds_map[i])

    # Create JAX BCOO sparse matrix for compatibility with FEAX ecosystem
    # BCOO format: (values, indices) where indices shape is (nnz, 2)
    indices = np.stack([np.array(I), np.array(J)], axis=1)
    values = np.array(V)
    P_mat = BCOO((values, indices), shape=(N, M))

    return P_mat


def cyclic_prolongation_matrix(
    mesh: Mesh,
    vec: int,
    location_master: Callable[[onp.ndarray], bool],
    location_slave: Callable[[onp.ndarray], bool],
    rotation: onp.ndarray,
    mapping: Callable[[onp.ndarray], onp.ndarray],
    tol: float = 1e-5,
) -> BCOO:
    """Prolongation matrix for *cyclic* (rotational) symmetry between two cut faces.

    Unlike :func:`prolongation_matrix` — which enforces the translational relation
    ``u_slave = u_master`` component-by-component — this builds the rotation-coupled
    constraint

    .. math:: \\mathbf{u}_{\\text{slave}} = \\mathbf{R}\\,\\mathbf{u}_{\\text{master}},

    where ``R`` is the sector rotation. This is what a 1/N angular sector of a body
    of revolution requires on its two radial cut planes: the slave-plane DoF block of
    each node is expressed as a linear combination (via ``R``) of the matching
    master-plane node's *independent* DoF block, so the whole slave plane is removed
    from the reduced system.

    Args:
        mesh (Mesh): Finite element mesh (provides ``points``).
        vec (int): DoFs per node (3 for 3D elasticity).
        location_master (Callable): Predicate selecting master-plane nodes.
        location_slave (Callable): Predicate selecting slave-plane nodes.
        rotation (array): ``(vec, vec)`` rotation matrix ``R`` relating the master
            DoF block to the slave DoF block (e.g. rotation by the sector angle about
            the symmetry axis).
        mapping (Callable): Maps a master-plane *point* onto its matching slave-plane
            *point* (typically the same physical rotation as ``rotation`` applied to
            coordinates). Used only to pair nodes geometrically.
        tol (float): Geometric matching tolerance.

    Returns:
        BCOO: Prolongation matrix ``P`` of shape ``(N, M)`` with ``u = P @ u_reduced``;
        ``N`` is the full DoF count, ``M = N - vec * (#slave nodes)``.

    Notes:
        Master nodes must themselves be independent (they must not also be slaves of
        another pairing); for a single radial sector with a polar opening the two cut
        planes share no nodes, so this holds. Nodes carrying a Dirichlet BC should be
        disjoint from the slave (eliminated) set to avoid conflicting constraints.
    """
    points = onp.asarray(mesh.points)
    num_nodes = points.shape[0]
    R = onp.asarray(rotation, dtype=onp.float64)
    if R.shape != (vec, vec):
        raise ValueError(f"rotation must be ({vec}, {vec}), got {R.shape}")

    master_nodes = onp.argwhere(jax.vmap(location_master)(mesh.points)).reshape(-1)
    slave_nodes_all = onp.argwhere(jax.vmap(location_slave)(mesh.points)).reshape(-1)
    slave_pts = points[slave_nodes_all]

    # Pair each master node with the slave node sitting at its rotated image.
    slave_ordered = []
    for m in master_nodes:
        target = onp.asarray(mapping(points[m]))
        dist = onp.linalg.norm(target[None, :] - slave_pts, axis=-1)
        hit = onp.argwhere(dist < tol).reshape(-1)
        assert hit.size == 1, (
            f"Master node {m} at {points[m]} matched {hit.size} slave nodes "
            f"(expected exactly 1). Check the mapping/rotation or tolerance."
        )
        slave_ordered.append(int(slave_nodes_all[hit[0]]))
    slave_ordered = onp.array(slave_ordered, dtype=onp.int64)
    assert len(master_nodes) == len(slave_ordered), (
        f"Mismatch in node pairing: {len(master_nodes)} master vs "
        f"{len(slave_ordered)} slave nodes."
    )

    is_slave = onp.zeros(num_nodes, dtype=bool)
    is_slave[slave_ordered] = True
    indep_nodes = onp.where(~is_slave)[0]
    reduced_node_index = -onp.ones(num_nodes, dtype=onp.int64)
    reduced_node_index[indep_nodes] = onp.arange(len(indep_nodes))

    master_of_slave = {s: int(m) for m, s in zip(master_nodes, slave_ordered)}

    N = num_nodes * vec
    M = len(indep_nodes) * vec

    I, J, V = [], [], []
    for node in range(num_nodes):
        if is_slave[node]:
            m = master_of_slave[node]
            rm = reduced_node_index[m]
            assert rm >= 0, "Master node was eliminated; nested pairing not supported."
            for a in range(vec):
                row = node * vec + a
                for b in range(vec):
                    if R[a, b] != 0.0:
                        I.append(row)
                        J.append(int(rm * vec + b))
                        V.append(float(R[a, b]))
        else:
            rn = reduced_node_index[node]
            for a in range(vec):
                I.append(node * vec + a)
                J.append(int(rn * vec + a))
                V.append(1.0)

    indices = np.stack([np.array(I), np.array(J)], axis=1)
    values = np.array(V)
    return BCOO((values, indices), shape=(N, M))


def periodic_bc_3D(
    unitcell: UnitCell, vec: int = 1, dim: int = 3
) -> List[PeriodicPairing]:
    """Generate periodic boundary condition pairings for a 3D unit cell.

    Creates a complete set of periodic pairings for all faces, edges, and corners of a
    3D unit cell. This ensures full periodicity where opposite boundaries are constrained
    to have compatible displacements, essential for RVE-based homogenization.

    The function systematically pairs:
    - Opposite faces (6 face pairs for 3D)
    - Corresponding edges (12 edge pairs for 3D)
    - Corresponding corners (7 corner pairs for 3D, excluding origin)

    Args:
        unitcell (UnitCell): The unit cell object providing boundary identification
            functions and geometric mapping capabilities.
        vec (int): Number of degrees of freedom per node. Defaults to 1.
        dim (int): Spatial dimension of the problem. Defaults to 3.

    Returns:
        List[PeriodicPairing]: Complete list of periodic pairings ordered as:
            1. Corner pairings (excluding origin as master)
            2. Edge pairings (excluding corners)
            3. Face pairings (excluding edges and corners)

    Example:
        >>> unit_cell = UnitCell()
        >>> ...
        >>> pairings = periodic_bc_3D(unit_cell, vec=3)  # 3D elasticity
        >>> print(f"Total pairings: {len(pairings)}")
        >>> # Should be: 7 corners + 12 edges + 6 faces = 25 per DoF component

    Note:
        - The origin corner serves as the master for all other corners
        - Edge and face exclusions prevent double-counting of constraints
        - Each geometric pairing is replicated for each DoF component
    """

    L = unitcell.ub - unitcell.lb

    face_pairs = []
    for axis in range(dim):
        master_fn = unitcell.face_function(axis, 0, excluding_edge=True)
        slave_fn = unitcell.face_function(axis, L[axis], excluding_edge=True)
        for i in range(vec):
            face_pairs.append(
                PeriodicPairing(
                    master_fn, slave_fn, unitcell.mapping(master_fn, slave_fn), i
                )
            )

    edge_pairs = []
    for axes in [[1, 2], [0, 2], [0, 1]]:
        for values in [[L[axes[0]], 0], [L[axes[0]], L[axes[1]]], [0, L[axes[1]]]]:
            master_fn = unitcell.edge_function(axes, [0, 0], excluding_corner=True)
            slave_fn = unitcell.edge_function(axes, values, excluding_corner=True)
            for i in range(vec):
                edge_pairs.append(
                    PeriodicPairing(
                        master_fn, slave_fn, unitcell.mapping(master_fn, slave_fn), i
                    )
                )

    corner_origin = unitcell.lb
    corner_pairs = []
    for corner in itertools.product(
        *[[corner_origin[i], corner_origin[i] + L[i]] for i in range(dim)]
    ):
        if np.allclose(np.array(corner), corner_origin):
            continue
        master_fn = unitcell.corner_function(corner_origin)
        slave_fn = unitcell.corner_function(corner)
        for i in range(vec):
            corner_pairs.append(
                PeriodicPairing(
                    master_fn, slave_fn, unitcell.mapping(master_fn, slave_fn), i
                )
            )
    return corner_pairs + edge_pairs + face_pairs


def periodic_bc_2D(
    unitcell: UnitCell, vec: int = 1, dim: int = 2
) -> List[PeriodicPairing]:
    """Generate periodic boundary condition pairings for a 2D unit cell.

    Creates a complete set of periodic pairings for all edges and corners of a
    2D unit cell. This ensures full periodicity where opposite boundaries are
    constrained to have compatible displacements.

    The function systematically pairs:
    - Opposite edges (2 edge pairs, excluding corners)
    - Corresponding corners (3 corner pairs, excluding origin)

    Args:
        unitcell (UnitCell): The unit cell object providing boundary identification
            functions and geometric mapping capabilities.
        vec (int): Number of degrees of freedom per node. Defaults to 1.
        dim (int): Spatial dimension of the problem. Defaults to 2.

    Returns:
        List[PeriodicPairing]: Complete list of periodic pairings ordered as:
            1. Corner pairings (excluding origin as master)
            2. Edge pairings (excluding corners)
    """

    L = unitcell.ub - unitcell.lb

    # Edge pairs: pair opposite edges, excluding corners
    edge_pairs = []
    for axis in range(dim):
        master_fn = unitcell.face_function(axis, unitcell.lb[axis],
                                           excluding_corner=True)
        slave_fn = unitcell.face_function(axis, unitcell.lb[axis] + L[axis],
                                          excluding_corner=True)
        for i in range(vec):
            edge_pairs.append(
                PeriodicPairing(
                    master_fn, slave_fn,
                    unitcell.mapping(master_fn, slave_fn), i
                )
            )

    # Corner pairs: origin to all other corners
    corner_origin = unitcell.lb
    corner_pairs = []
    for corner in itertools.product(
        *[[corner_origin[i], corner_origin[i] + L[i]] for i in range(dim)]
    ):
        if np.allclose(np.array(corner), corner_origin):
            continue
        master_fn = unitcell.corner_function(corner_origin)
        slave_fn = unitcell.corner_function(corner)
        for i in range(vec):
            corner_pairs.append(
                PeriodicPairing(
                    master_fn, slave_fn,
                    unitcell.mapping(master_fn, slave_fn), i
                )
            )

    return corner_pairs + edge_pairs
