"""Tests for feax.solvers.cmg helpers: make_KE_3d (element stiffness),
auto_levels (MG depth), compact_nodes / load_vector (band DOF addressing).
The solver itself (create_solver / compliance_and_dc / TracedParams acceptance)
is covered in test_params_bridge.py and test_solution.py."""

import numpy as onp
import pytest

import feax as fe
from feax.solvers.cmg import auto_levels, make_KE_3d


# ---------------------------------------------------------------------------
# make_KE_3d
# ---------------------------------------------------------------------------

def test_make_ke_3d_symmetric_psd_with_rigid_modes():
    KE = make_KE_3d(nu=0.3, E=1.0)
    assert KE.shape == (24, 24)
    assert onp.allclose(KE, KE.T, atol=1e-12)
    w = onp.linalg.eigvalsh(KE)
    assert w.min() > -1e-10                       # positive semi-definite
    assert (onp.abs(w) < 1e-10).sum() == 6        # exactly the 6 rigid-body modes
    # rigid translation in each direction is in the null space
    for comp in range(3):
        u = onp.zeros(24)
        u[comp::3] = 1.0
        assert onp.abs(KE @ u).max() < 1e-12


def test_make_ke_3d_scales_linearly_with_e():
    K1 = make_KE_3d(nu=0.3, E=1.0)
    K2 = make_KE_3d(nu=0.3, E=7.5)
    assert onp.allclose(K2, 7.5 * K1)


# ---------------------------------------------------------------------------
# auto_levels
# ---------------------------------------------------------------------------

def test_auto_levels_halves_until_floor():
    assert auto_levels((32, 32, 32)) == 4     # 32 -> 16 -> 8 -> 4
    assert auto_levels((8, 4, 4)) == 1        # min dim already at floor
    assert auto_levels((9, 5, 5)) == 2        # ceil-halving handles odd dims
    assert auto_levels((64, 32, 16), floor=8) == 2


# ---------------------------------------------------------------------------
# compact_nodes / load_vector
# ---------------------------------------------------------------------------

def test_compact_nodes_and_load_vector():
    grid = fe.StructuredGrid((8, 4, 4))
    cmg = fe.NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: ni == 0,
                           nu=0.3, penal=3.0, bucket=64)
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    levels = cmg.build(active)

    tip = grid.nodes_where(lambda I, J, K: (I == 8) & (K == 0))
    cn = cmg.compact_nodes(levels, tip)
    # compact index looks the global id back up in the level-0 node list
    assert onp.array_equal(onp.asarray(levels[0]["nodes"])[cn], onp.asarray(tip))

    b = cmg.load_vector(levels, tip, comp=2, value=-1.0)
    assert b.shape == (levels[0]["ndof"],)
    assert onp.allclose(b[3 * cn + 2], -1.0)
    assert onp.count_nonzero(b) == tip.size
    # a different component addresses disjoint DOFs
    bx = cmg.load_vector(levels, tip, comp=0, value=2.0)
    assert onp.allclose(bx[3 * cn], 2.0)
    assert onp.dot(b, bx) == 0.0
