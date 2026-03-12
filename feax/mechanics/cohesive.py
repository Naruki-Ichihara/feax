"""Cohesive zone interface for matrix-free fracture simulations.

This module separates two concerns:

1. **Cohesive potential** (material model):
   Pure functions ``potential(delta, delta_max, **params) -> phi_per_node``
   that define the traction-separation law.

2. **CohesiveInterface** (geometry):
   Holds node pairs, integration weights, and interface normals.
   Composes with a potential to produce an energy function
   ``(u_flat, delta_max) -> scalar`` for use with ``newton_solve``.

Supports mixed-mode fracture by decomposing the displacement jump
into normal and tangential components:

```python
δ_n = jump · n          (normal opening, scalar)
δ_t = |jump - δ_n · n|  (tangential sliding, scalar)
δ   = sqrt(⟨δ_n⟩₊² + β² δ_t²)  (effective opening)
```

where β is the mode-mixity ratio (default 1.0).

Usage:

```python
from feax.mechanics.cohesive import CohesiveInterface, exponential_potential

# Axis-aligned interface (simple)
interface = CohesiveInterface.from_axis(
    top_nodes, bottom_nodes, weights, normal_axis=1, vec=2)

# Arbitrary interface with per-node normals
interface = CohesiveInterface(
    top_nodes, bottom_nodes, weights, normals=normals, vec=2)

cohesive_energy = interface.create_energy_fn(
    exponential_potential, Gamma=15.0, sigma_c=20000.0,
)

def total_energy(u, delta_max):
    return elastic_energy(u) + cohesive_energy(u, delta_max)
```
"""

import numpy as onp

import jax.numpy as np


# ============================================================================
# Cohesive potentials (pure functions)
# ============================================================================

def exponential_potential(delta, delta_max, *, Gamma, sigma_c):
    """Xu-Needleman exponential cohesive potential with irreversibility.

    Loading:

    ```python
    φ(δ) = Γ [1 - (1 + δ/δc) exp(-δ/δc)]
    ```

    where ``δc = Γ / (e · σc)``.

    Unloading follows a secant path back to the origin:

    ```python
    φ_unload(δ) = φ(δ_max) / δ_max² · δ²
    ```

    Parameters
    ----------
    delta : jax.Array
        Current effective opening, shape ``(n_nodes,)``.
    delta_max : jax.Array
        Historical maximum opening, shape ``(n_nodes,)``.
    Gamma : float
        Fracture energy [J/m²].
    sigma_c : float
        Critical cohesive traction [Pa].

    Returns
    -------
    phi : jax.Array
        Cohesive energy density per node, shape ``(n_nodes,)``.
    """
    delta_c = Gamma / (np.e * sigma_c)

    phi_load = Gamma * (1 - (1 + delta / delta_c) * np.exp(-delta / delta_c))

    phi_max = Gamma * (1 - (1 + delta_max / delta_c) * np.exp(-delta_max / delta_c))
    k_secant = np.where(delta_max > 1e-15, phi_max / (delta_max**2 + 1e-30), 0.0)
    phi_unload = k_secant * delta**2

    is_loading = delta >= delta_max - 1e-15
    return np.where(is_loading, phi_load, phi_unload)


def bilinear_potential(delta, delta_max, *, Gamma, sigma_c):
    """Bilinear cohesive potential with irreversibility.

    Loading:

    ```python
    φ(δ) = 0.5 k₀ δ²                                  for δ < δ₀
    φ(δ) = Γ - 0.5 σc (δ_f - δ)² / (δ_f - δ₀)        for δ₀ ≤ δ < δ_f
    φ(δ) = Γ                                            for δ ≥ δ_f
    ```

    where ``δ_f = 2Γ / σc``, ``δ₀ = σc / k₀``, ``k₀ = σc² / (2Γ)``.

    Unloading follows a secant path back to the origin.

    Parameters
    ----------
    delta : jax.Array
        Current effective opening, shape ``(n_nodes,)``.
    delta_max : jax.Array
        Historical maximum opening, shape ``(n_nodes,)``.
    Gamma : float
        Fracture energy [J/m²].
    sigma_c : float
        Critical cohesive traction [Pa].

    Returns
    -------
    phi : jax.Array
        Cohesive energy density per node, shape ``(n_nodes,)``.
    """
    delta_f = 2.0 * Gamma / sigma_c
    delta_0 = 2.0 * Gamma / (sigma_c * delta_f)

    k0 = sigma_c / delta_0
    phi_ascending = 0.5 * k0 * delta**2
    phi_descending = Gamma - 0.5 * sigma_c * (delta_f - delta)**2 / (delta_f - delta_0 + 1e-30)
    phi_full = Gamma * np.ones_like(delta)

    phi_load = np.where(delta < delta_0, phi_ascending,
               np.where(delta < delta_f, phi_descending, phi_full))

    phi_at_max = np.where(delta_max < delta_0, 0.5 * k0 * delta_max**2,
                 np.where(delta_max < delta_f,
                          Gamma - 0.5 * sigma_c * (delta_f - delta_max)**2 / (delta_f - delta_0 + 1e-30),
                          Gamma))
    k_secant = np.where(delta_max > 1e-15, phi_at_max / (delta_max**2 + 1e-30), 0.0)
    phi_unload = k_secant * delta**2

    is_loading = delta >= delta_max - 1e-15
    return np.where(is_loading, phi_load, phi_unload)


# ============================================================================
# Interface geometry
# ============================================================================

def compute_trapezoidal_weights(coords_1d):
    """Compute trapezoidal integration weights from sorted 1D coordinates.

    For 2D cohesive interfaces where nodes are arranged along a line.

    Parameters
    ----------
    coords_1d : array_like
        Sorted 1D coordinates of interface nodes.

    Returns
    -------
    weights : jax.Array
        Integration weights, shape ``(n,)``.
    """
    x = onp.asarray(coords_1d)
    n = len(x)
    dx = onp.zeros(n)
    for i in range(n):
        left = (x[i] - x[i - 1]) / 2 if i > 0 else 0.0
        right = (x[i + 1] - x[i]) / 2 if i < n - 1 else 0.0
        dx[i] = left + right
    return np.array(dx)


def compute_lumped_area_weights(node_ids, coords, quads):
    """Compute lumped area weights from quad elements on a planar interface.

    For 3D cohesive interfaces where quad elements define the surface.
    Each quad contributes 1/4 of its area to each of its 4 nodes.

    Parameters
    ----------
    node_ids : array_like
        Node indices for which to compute weights, shape ``(n,)``.
    coords : array_like
        Full coordinate array, shape ``(n_total, 3)``.
    quads : array_like
        Quad element connectivity (indices into ``coords``), shape ``(n_quads, 4)``.

    Returns
    -------
    weights : jax.Array
        Area weights per node, shape ``(n,)``.
    """
    node_ids = onp.asarray(node_ids)
    coords = onp.asarray(coords)
    quads = onp.asarray(quads)

    node_to_local = {int(n): i for i, n in enumerate(node_ids)}
    w = onp.zeros(len(node_ids))

    for quad in quads:
        pts = coords[quad]
        d1 = pts[2] - pts[0]
        d2 = pts[3] - pts[1]
        area = 0.5 * onp.linalg.norm(onp.cross(d1, d2))
        for nid in quad:
            idx = node_to_local.get(int(nid))
            if idx is not None:
                w[idx] += area / 4.0

    return np.array(w)


class CohesiveInterface:
    """Cohesive zone interface with mixed-mode support.

    Decomposes the displacement jump into normal and tangential components
    and computes an effective opening for the cohesive potential:

    ```python
    δ_n = jump · n                     (normal)
    δ_t = |jump - δ_n · n|             (tangential)
    δ   = sqrt(⟨δ_n⟩₊² + β² δ_t²)    (effective)
    ```

    where ``⟨·⟩₊ = max(·, 0)`` is the Macaulay bracket (no energy in
    compression) and β is the mode-mixity ratio.

    Parameters
    ----------
    top_nodes : array_like
        Node indices on the ``+`` side of the interface, shape ``(n,)``.
    bottom_nodes : array_like
        Node indices on the ``-`` side of the interface, shape ``(n,)``.
    weights : array_like
        Integration weights per node pair, shape ``(n,)``.
    normals : array_like
        Unit normal vectors pointing from bottom to top, shape ``(n, vec)``.
    vec : int
        Number of displacement components per node (2 or 3).
    beta : float, optional
        Mode-mixity ratio for tangential contribution (default 1.0).
        β=0 gives pure Mode I; β=1 gives equal weight to Mode I and II.

    Examples
    --------
    >>> # Axis-aligned interface (convenience)
    >>> interface = CohesiveInterface.from_axis(
    ...     top_nodes, bottom_nodes, weights, normal_axis=1, vec=2)
    >>>
    >>> # Arbitrary normals
    >>> normals = compute_normals(...)  # (n, 2)
    >>> interface = CohesiveInterface(
    ...     top_nodes, bottom_nodes, weights, normals, vec=2, beta=0.5)
    """

    def __init__(self, top_nodes, bottom_nodes, weights, normals, vec,
                 beta=1.0):
        self.top_nodes = np.asarray(top_nodes)
        self.bottom_nodes = np.asarray(bottom_nodes)
        self.weights = np.asarray(weights)
        self.normals = np.asarray(normals)
        self.vec = vec
        self.beta = beta
        self.n_nodes = len(self.top_nodes)

        if self.normals.shape != (self.n_nodes, vec):
            raise ValueError(
                f"normals shape {self.normals.shape} does not match "
                f"expected ({self.n_nodes}, {vec})"
            )

    @classmethod
    def from_axis(cls, top_nodes, bottom_nodes, weights, normal_axis, vec, beta=1.0):
        """Create interface with axis-aligned normal.

        Convenience constructor for interfaces aligned with a coordinate
        axis (e.g., y=0 plane with normal in y-direction).

        Parameters
        ----------
        normal_axis : int
            Coordinate axis index for the normal direction
            (0=x, 1=y, 2=z).
        """
        n = len(top_nodes)
        normals = onp.zeros((n, vec))
        normals[:, normal_axis] = 1.0
        return cls(top_nodes, bottom_nodes, weights, normals, vec, beta)

    def get_jump(self, u_flat):
        """Compute displacement jump at interface nodes.

        Parameters
        ----------
        u_flat : jax.Array
            Flat displacement vector.

        Returns
        -------
        jump : jax.Array
            Displacement jump (top - bottom), shape ``(n_nodes, vec)``.
        """
        u = u_flat.reshape(-1, self.vec)
        return u[self.top_nodes] - u[self.bottom_nodes]

    def get_opening(self, u_flat):
        """Compute effective opening at interface nodes.

        Returns the mixed-mode effective opening:

        ```python
        δ = sqrt(⟨δ_n⟩₊² + β² δ_t²)
        ```

        Parameters
        ----------
        u_flat : jax.Array
            Flat displacement vector.

        Returns
        -------
        delta : jax.Array
            Effective opening, shape ``(n_nodes,)``.
        """
        jump = self.get_jump(u_flat)
        return self._effective_opening(jump)

    def get_opening_components(self, u_flat):
        """Compute normal and tangential opening components.

        Parameters
        ----------
        u_flat : jax.Array
            Flat displacement vector.

        Returns
        -------
        delta_n : jax.Array
            Normal opening (signed), shape ``(n_nodes,)``.
        delta_t : jax.Array
            Tangential opening (unsigned), shape ``(n_nodes,)``.
        """
        jump = self.get_jump(u_flat)
        delta_n = np.sum(jump * self.normals, axis=1)
        jump_t = jump - delta_n[:, None] * self.normals
        delta_t = np.linalg.norm(jump_t, axis=1)
        return delta_n, delta_t

    def _effective_opening(self, jump):
        """Compute effective opening from jump vectors."""
        delta_n = np.sum(jump * self.normals, axis=1)
        jump_t = jump - delta_n[:, None] * self.normals
        delta_t_sq = np.sum(jump_t**2, axis=1)
        delta_n_pos = np.maximum(delta_n, 0.0)

        return np.sqrt(delta_n_pos**2 + self.beta**2 * delta_t_sq + 1e-30)

    def create_energy_fn(self, potential, **params):
        """Create a cohesive energy function.

        Parameters
        ----------
        potential : callable
            Cohesive potential function with signature
            ``potential(delta, delta_max, **params) -> phi_per_node``.
        **params
            Keyword arguments forwarded to ``potential``
            (e.g., ``Gamma=15.0, sigma_c=20000.0``).

        Returns
        -------
        energy_fn : callable
            Function ``energy_fn(u_flat, delta_max) -> scalar``
            suitable for use with ``newton_solve``.
        """
        top = self.top_nodes
        bottom = self.bottom_nodes
        normals = self.normals
        beta = self.beta
        vec = self.vec
        w = self.weights

        def energy_fn(u_flat, delta_max):
            u = u_flat.reshape(-1, vec)
            jump = u[top] - u[bottom]

            delta_n = np.sum(jump * normals, axis=1)
            jump_t = jump - delta_n[:, None] * normals
            delta_t_sq = np.sum(jump_t**2, axis=1)
            delta_n_pos = np.maximum(delta_n, 0.0)
            delta = np.sqrt(delta_n_pos**2 + beta**2 * delta_t_sq + 1e-30)

            phi = potential(delta, delta_max, **params)
            return np.sum(phi * w)

        return energy_fn
