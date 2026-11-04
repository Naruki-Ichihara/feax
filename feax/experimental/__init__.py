"""FEAX Experimental - staging area for APIs that are not yet stable.

Modules here are usable but their interfaces may change between releases
without a deprecation cycle.

Currently provides:

* :mod:`feax.experimental.mma` — Svanberg's MMA and GCMMA implemented in pure
  JAX, with every loop traced (``lax.while_loop`` / ``lax.scan``) so a whole
  optimisation compiles into a single XLA program and composes with ``jit``,
  ``vmap`` and ``grad``.
* :mod:`feax.experimental.cmg` — the matrix-free O(band) geometric multigrid
  solver :class:`NarrowBandCMG` for structured narrow bands, with pluggable
  constitutive laws (:class:`ScalarMaterial`, :class:`KernelMaterial`) and any
  number of DOFs per node.
* :mod:`feax.experimental.narrowband` — the moving-band topology
  optimization loop (OC, or MMA with full-grid state) and multires bootstrap
  built on cmg, plus the O(band) sensitivity filter.
* :mod:`feax.experimental.stripe` — stripe patterns aligned with a nodal line
  field on planar TRI3 meshes (2D port of Knöppel et al. 2015): print paths /
  filament layouts as a phase ``α`` (:class:`StripePattern`), with ``cos α``
  and the ``sin α`` interface level set for XFEM, plus line-field smoothing;
  :class:`DifferentiableStripes` gives gradients through the eigenproblem.
* :mod:`feax.experimental.xfem_crack` — LEFM crack growth with XFEM: the
  crack polyline drives traced (fixed-shape) level sets, enrichment masks and
  sub-cell quadrature, Heaviside + branch-function enrichment, and the
  J-vector by autodiff of a virtual crack extension (:class:`CrackSpace`).
* :mod:`feax.experimental.xfem` — XFEM on a fixed mesh: a nodal level set
  selects the cut cells, their nodes get enriched DOFs on a DOF-only second
  space (:class:`XFEMSpace`), and :class:`XFEMProblem` integrates a user
  constitutive law on the interface-conforming sub-cell quadrature with a
  pluggable enrichment (:class:`ShiftedHeaviside`, :class:`Heaviside`,
  :class:`ShiftedAbs`, :class:`ModifiedAbs`).
"""

from . import cmg
from . import mma
from . import narrowband
from . import stripe
from . import xfem
from . import xfem_crack
from .cmg import (
    KernelMaterial,
    Material,
    NarrowBandCMG,
    ScalarMaterial,
    auto_levels,
    build_geometry,
    make_KE_3d,
    make_KE_conduction,
    ramp,
    simp,
)
from .mma import GCMMA, MMA, MMAState
from .narrowband import (
    create_sensitivity_filter,
    oc_update,
    run_narrowband_mma,
    run_narrowband_multires,
    run_narrowband_oc,
)
from .stripe import (
    DifferentiableStripes,
    StripePattern,
    field_index,
    line_field_angle,
    smooth_line_field,
)
from .xfem_crack import (
    CrackProblem,
    CrackSpace,
    crack_problem,
    plane_strain_energy,
)
from .xfem import (
    CohesiveZone,
    Enrichment,
    Heaviside,
    ModifiedAbs,
    ShiftedAbs,
    ShiftedHeaviside,
    XFEMProblem,
    XFEMSpace,
    bilinear_tsl,
    exponential_tsl,
    xfem_problem,
)

__all__ = [
    'cmg',
    'mma',
    'narrowband',
    'stripe',
    'xfem',
    'xfem_crack',
    # cmg
    'NarrowBandCMG',
    'Material',
    'ScalarMaterial',
    'KernelMaterial',
    'build_geometry',
    'auto_levels',
    'make_KE_3d',
    'make_KE_conduction',
    'simp',
    'ramp',
    # mma
    'MMA',
    'GCMMA',
    'MMAState',
    # narrowband
    'oc_update',
    'run_narrowband_oc',
    'run_narrowband_mma',
    'run_narrowband_multires',
    'create_sensitivity_filter',
    # xfem
    'Enrichment',
    'ShiftedHeaviside',
    'Heaviside',
    'ShiftedAbs',
    'ModifiedAbs',
    'XFEMSpace',
    'XFEMProblem',
    'xfem_problem',
    'CohesiveZone',
    'bilinear_tsl',
    'exponential_tsl',
    # stripe
    'StripePattern',
    'DifferentiableStripes',
    'line_field_angle',
    'field_index',
    'smooth_line_field',
    # xfem_crack
    'CrackSpace',
    'CrackProblem',
    'crack_problem',
    'plane_strain_energy',
]
