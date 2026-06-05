---
sidebar_label: tmc
title: feax.mechanics.tmc
---

Third Medium Contact for frictionless self-contact.

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
        ele_type=&#x27;QUAD9&#x27;,
    )

References
----------
[1] G. L. Bluhm et al., &quot;Internal contact modeling for finite strain
    topology optimization&quot;, Comput. Mech. 67, 1099–1114 (2021).
[2] A. H. Frederiksen et al., &quot;Topology optimization of self-contacting
    structures&quot;, Comput. Mech. 73, 967–981 (2023).

#### classify\_medium\_cells

```python
def classify_medium_cells(mesh: fe.Mesh,
                          is_medium_fn: Callable,
                          n_corner_nodes: int = 4) -> np.ndarray
```

Classify cells as body (0) or medium (1) based on centroid coordinates.

The centroid is computed from the first ``n_corner_nodes`` nodes of each
cell (e.g. 4 for QUAD9/HEX20, 3 for TRI6).

Parameters
----------
- **mesh** (*feax.Mesh*): Finite element mesh.
- **is_medium_fn** (*callable*): Function ``(centroid_x, centroid_y, ...) -&gt; bool`` that returns True for medium cells.  Receives unpacked centroid coordinates as positional arguments (2 args for 2D, 3 for 3D).
- **n_corner_nodes** (*int, optional*): Number of corner nodes used to compute centroids (default 4).


Returns
-------
- **is_medium** (*jax.Array*): Per-cell indicator, shape ``(num_cells,)``.  1.0 for medium, 0.0 for body.


Examples
--------
```python
>>> # L-shaped body with interior void and right strip
>>> is_medium = classify_medium_cells(mesh, lambda cx, cy:
...     (t < cx < L and t < cy < H - t) or cx > L)
```

## ThirdMediumContact Objects

```python
class ThirdMediumContact(fe.Problem)
```

Neo-Hookean + HuHu-LuLu regularization Problem for third-medium contact.

Do not instantiate directly — use :meth:`create` which also builds the
matching :class:`~feax.InternalVars`.

Parameters stored via ``additional_info``:
    ``(kr_coeff, plane_strain)``

Internal variable ordering (passed to kernels):
    ``(mu_cell, lmbda_cell, shape_hessians, is_medium)``

#### get\_energy\_density

```python
def get_energy_density()
```

Compressible Neo-Hookean energy with safe log extension.

2D plane strain:  ψ = μ/2 (tr C + 1) − μ ln J + λ/2 (ln J)²
3D:               ψ = μ/2 tr C       − μ ln J + λ/2 (ln J)²

#### get\_universal\_kernel

```python
def get_universal_kernel()
```

Regularization kernel applied only on medium cells.

E_reg = kr_coeff ∫_medium (H:::H − (1/d) L·L) dΩ

#### create

```python
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
        plane_strain: bool = True
) -> Tuple["ThirdMediumContact", InternalVars]
```

Create a TMC problem with matching internal variables.

Parameters
----------
- **mesh** (*feax.Mesh*): Single mesh covering body + medium regions.
- **is_medium** (*jax.Array*): Per-cell indicator (1.0 = medium, 0.0 = body), shape ``(num_cells,)``.  Obtain via :func:`classify_medium_cells`.
- **mu** (*float*): Shear modulus of the body (G).
- **lmbda** (*float*): Lamé parameter of the body (K / λ).
- **gamma0** (*float, optional*): Stiffness scaling for the medium (default 5e-7).
- **kr** (*float, optional*): Regularization prefactor (default 5e-7).
- **ele_type** (*str, optional*): Element type (default ``&#x27;QUAD9&#x27;``).
- **dim** (*int, optional*): Spatial dimension.  Inferred from ``mesh.points`` if omitted.
- **ref_length** (*float, optional*): Reference length for regularization coefficient ``kr_coeff = kr * lmbda * ref_length²`` (default 1.0).
- **plane_strain** (*bool, optional*): If True (default) and ``dim == 2``, add +1 to tr(C) for the out-of-plane stretch.  Ignored when ``dim == 3``.


Returns
-------
- **problem** (*ThirdMediumContact*): Configured feax Problem (with ``hess=True``).
- **iv** (*feax.InternalVars*): Internal variables ready for ``create_solver`` / ``newton_solve``.


Examples
--------
```python
>>> problem, iv = ThirdMediumContact.create(
...     mesh, is_medium, mu=G, lmbda=K,
...     gamma0=5e-7, kr=5e-7, ele_type=&#x27;QUAD9&#x27;,
... )
>>> solver = fe.create_solver(problem, bc, internal_vars=iv, ...)
```

