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
"""

from . import cmg
from . import mma
from . import narrowband
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

__all__ = [
    'cmg',
    'mma',
    'narrowband',
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
]
