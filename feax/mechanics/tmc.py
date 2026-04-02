"""Third Medium Contact for frictionless self-contact.

Implements the third-medium method with HuHu-LuLu Hessian-based
regularization [1, 2].  Two material regions share a single mesh:

- **Body** (solid): Neo-Hookean compressible with (μ, λ)
- **Medium** (void): Neo-Hookean with scaled-down (γ₀·μ, γ₀·λ)
  plus biharmonic regularization

The regularization energy is:

    E_reg = kr · λ · L² ∫_medium (H:::H − (1/d) L·L) dΩ

where H is the displacement Hessian, L is the Laplacian, and d is the
spatial dimension.

Usage
-----
::

    from feax.mechanics.tmc import ThirdMediumContact, classify_medium_cells

    is_medium = classify_medium_cells(mesh, lambda cx, cy: ...)

    problem, iv = ThirdMediumContact.create(
        mesh, is_medium=is_medium,
        mu=0.357, lmbda=1.667,
        gamma0=5e-7, kr=5e-7,
        ele_type='QUAD9',
    )

References
----------
[1] G. L. Bluhm et al., "Internal contact modeling for finite strain
    topology optimization", Comput. Mech. 67, 1099–1114 (2021).
[2] A. H. Frederiksen et al., "Topology optimization of self-contacting
    structures", Comput. Mech. 73, 967–981 (2023).
"""

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp

import feax as fe
from feax.internal_vars import InternalVars


# ============================================================================
# Cell classification
# ============================================================================

def classify_medium_cells(
    mesh: fe.Mesh,
    is_medium_fn: Callable,
    n_corner_nodes: int = 4,
) -> np.ndarray:
    """Classify cells as body (0) or medium (1) based on centroid coordinates.

    The centroid is computed from the first ``n_corner_nodes`` nodes of each
    cell (e.g. 4 for QUAD9/HEX20, 3 for TRI6).

    Parameters
    ----------
    mesh : feax.Mesh
        Finite element mesh.
    is_medium_fn : callable
        Function ``(centroid_x, centroid_y, ...) -> bool`` that returns
        True for medium cells.  Receives unpacked centroid coordinates
        as positional arguments (2 args for 2D, 3 for 3D).
    n_corner_nodes : int, optional
        Number of corner nodes used to compute centroids (default 4).

    Returns
    -------
    is_medium : jax.Array
        Per-cell indicator, shape ``(num_cells,)``.  1.0 for medium, 0.0
        for body.

    Examples
    --------
    >>> # L-shaped body with interior void and right strip
    >>> is_medium = classify_medium_cells(mesh, lambda cx, cy:
    ...     (t < cx < L and t < cy < H - t) or cx > L)
    """
    points = onp.asarray(mesh.points)
    cells = onp.asarray(mesh.cells)
    centroids = onp.mean(points[cells[:, :n_corner_nodes]], axis=1)

    result = onp.zeros(len(cells), dtype=onp.float64)
    for c in range(len(cells)):
        if is_medium_fn(*centroids[c]):
            result[c] = 1.0
    return np.array(result)


# ============================================================================
# Problem subclass
# ============================================================================

class ThirdMediumContact(fe.Problem):
    """Neo-Hookean + HuHu-LuLu regularization Problem for third-medium contact.

    Do not instantiate directly — use :meth:`create` which also builds the
    matching :class:`~feax.InternalVars`.

    Parameters stored via ``additional_info``:
        ``(kr_coeff, plane_strain)``

    Internal variable ordering (passed to kernels):
        ``(mu_cell, lmbda_cell, shape_hessians, is_medium)``
    """

    def custom_init(self, kr_coeff: float, plane_strain: bool) -> None:
        self._kr_coeff = kr_coeff
        self._plane_strain = plane_strain

    # ── Neo-Hookean energy density ──────────────────────────────────

    def get_energy_density(self):
        """Compressible Neo-Hookean energy with safe log extension.

        2D plane strain:  ψ = μ/2 (tr C + 1) − μ ln J + λ/2 (ln J)²
        3D:               ψ = μ/2 tr C       − μ ln J + λ/2 (ln J)²
        """
        dim = self.dim
        plane_strain = self._plane_strain
        J_min = 1e-4

        def safe_lnJ(J):
            lnJ_min = np.log(J_min)
            s = (J - J_min) / J_min
            ext = lnJ_min + s - 0.5 * s ** 2
            return np.where(J > J_min, np.log(J), ext)

        def psi(u_grad, mu, lmbda, *_unused):
            F = u_grad + np.eye(dim)
            C = F.T @ F
            J = np.linalg.det(F)
            lnJ = safe_lnJ(J)
            trC = np.trace(C)
            if plane_strain:
                trC = trC + 1.0  # F33 = 1 contribution
            return mu / 2.0 * trC - mu * lnJ + lmbda / 2.0 * lnJ ** 2

        return psi

    # ── HuHu-LuLu biharmonic regularization ────────────────────────

    def get_universal_kernel(self):
        """Regularization kernel applied only on medium cells.

        E_reg = kr_coeff ∫_medium (H:::H − (1/d) L·L) dΩ
        """
        dim = self.dim
        kr_coeff = self._kr_coeff

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                   cell_JxW, cell_v_grads_JxW,
                   mu, lmbda, cell_shape_hess, cell_is_medium):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_JxW_1d = cell_JxW[0]

            # Displacement Hessian: H_{q,v,K,L}
            u_hess = np.einsum('av,qaKL->qvKL', cell_sol, cell_shape_hess)

            # Laplacian: L_{q,v} = tr(H_{q,v,:,:})
            lapl_u = np.trace(u_hess, axis1=2, axis2=3)

            # Shape function Laplacian
            shape_lapl = np.trace(cell_shape_hess, axis1=-2, axis2=-1)

            # H:::∇²v
            term1 = np.einsum('qvKL,qaKL->qav', u_hess, cell_shape_hess)
            # (1/d) L · ∇²v
            term2 = np.einsum('qv,qa->qav', lapl_u, shape_lapl) / dim

            integrand = (term1 - term2) * cell_JxW_1d[:, None, None]
            result = kr_coeff * cell_is_medium * np.sum(integrand, axis=0)

            return jax.flatten_util.ravel_pytree(result)[0]

        return kernel

    # ── Factory ─────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        mesh: fe.Mesh,
        is_medium: np.ndarray,
        mu: float,
        lmbda: float,
        gamma0: float = 5e-7,
        kr: float = 5e-7,
        ele_type: str = "QUAD9",
        dim: Optional[int] = None,
        ref_length: float = 1.0,
        plane_strain: bool = True,
    ) -> Tuple["ThirdMediumContact", InternalVars]:
        """Create a TMC problem with matching internal variables.

        Parameters
        ----------
        mesh : feax.Mesh
            Single mesh covering body + medium regions.
        is_medium : jax.Array
            Per-cell indicator (1.0 = medium, 0.0 = body), shape
            ``(num_cells,)``.  Obtain via :func:`classify_medium_cells`.
        mu : float
            Shear modulus of the body (G).
        lmbda : float
            Lamé parameter of the body (K / λ).
        gamma0 : float, optional
            Stiffness scaling for the medium (default 5e-7).
        kr : float, optional
            Regularization prefactor (default 5e-7).
        ele_type : str, optional
            Element type (default ``'QUAD9'``).
        dim : int, optional
            Spatial dimension.  Inferred from ``mesh.points`` if omitted.
        ref_length : float, optional
            Reference length for regularization coefficient
            ``kr_coeff = kr * lmbda * ref_length²`` (default 1.0).
        plane_strain : bool, optional
            If True (default) and ``dim == 2``, add +1 to tr(C) for the
            out-of-plane stretch.  Ignored when ``dim == 3``.

        Returns
        -------
        problem : ThirdMediumContact
            Configured feax Problem (with ``hess=True``).
        iv : feax.InternalVars
            Internal variables ready for ``create_solver`` / ``newton_solve``.

        Examples
        --------
        >>> problem, iv = ThirdMediumContact.create(
        ...     mesh, is_medium, mu=G, lmbda=K,
        ...     gamma0=5e-7, kr=5e-7, ele_type='QUAD9',
        ... )
        >>> solver = fe.create_solver(problem, bc, internal_vars=iv, ...)
        """
        if dim is None:
            dim = mesh.points.shape[1]
        if dim == 3:
            plane_strain = False

        kr_coeff = kr * lmbda * ref_length ** 2

        vec = dim
        problem = cls(
            mesh, vec=vec, dim=dim, ele_type=ele_type, hess=True,
            additional_info=(kr_coeff, plane_strain),
        )

        # Cell-based material properties
        mu_cell = np.where(is_medium, mu * gamma0, mu)
        lmbda_cell = np.where(is_medium, lmbda * gamma0, lmbda)

        shape_hess = problem.fes[0].shape_hessians
        iv = InternalVars(
            volume_vars=(mu_cell, lmbda_cell, shape_hess, is_medium),
        )

        return problem, iv
