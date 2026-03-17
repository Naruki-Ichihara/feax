"""Tests for jax.grad through solvers w.r.t. bc_vals.

Validates that analytic gradients (via custom VJP / implicit function theorem)
match central finite differences for boundary condition values across:
- Linear solver (iter_num=1)
- Newton solver with fori_loop (make_jittable=True)
- Newton solver with Python loop (make_jittable=False)
"""

import jax
import jax.numpy as np
import pytest

import feax as fe
from feax.solvers.options import NewtonOptions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elastic_setup():
    """Small 2D linear elastic problem with non-trivial BC coupling."""
    mesh = fe.mesh.rectangle_mesh(
        Nx=2, Ny=2, domain_x=1.0, domain_y=1.0, ele_type='QUAD4',
    )

    class Elastic(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad):
                eps = 0.5 * (u_grad + u_grad.T)
                return 2.0 * eps + np.trace(eps) * np.eye(2)
            return stress

    problem = Elastic(mesh, vec=2, dim=2, ele_type='QUAD4')
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(
            location=lambda p: np.isclose(p[0], 0.0, atol=1e-6),
            component='all', value=0.0,
        ),
        fe.DirichletBCSpec(
            location=lambda p: np.isclose(p[0], 1.0, atol=1e-6),
            component='x', value=0.0,
        ),
    ]).create_bc(problem)
    iv = fe.InternalVars(volume_vars=())
    bc1 = bc.replace_vals(bc.bc_vals.at[-1].set(0.1))
    initial = fe.zero_like_initial_guess(problem, bc)
    return problem, bc, iv, bc1, initial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fd_grad(loss_fn, bc_vals, eps=1e-5):
    """Central finite-difference gradient."""
    grad = np.zeros_like(bc_vals)
    for i in range(len(bc_vals)):
        p1 = bc_vals.at[i].add(eps)
        p2 = bc_vals.at[i].add(-eps)
        grad = grad.at[i].set((loss_fn(p1) - loss_fn(p2)) / (2 * eps))
    return grad


def _check_grad(loss_fn, bc_vals, rtol=1e-4):
    """Compare analytic grad vs central FD, return relative error."""
    analytic = jax.grad(loss_fn)(bc_vals)
    fd = _fd_grad(loss_fn, bc_vals)
    rel_err = float(np.linalg.norm(analytic - fd) / (np.linalg.norm(fd) + 1e-30))
    return analytic, fd, rel_err


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
def test_bc_grad_linear(elastic_setup):
    """Linear solver (iter_num=1): grad w.r.t. bc_vals matches FD."""
    problem, bc, iv, bc1, initial = elastic_setup
    solver = fe.create_solver(
        problem, bc,
        solver_options=fe.IterativeSolverOptions(solver='cg'),
        iter_num=1, internal_vars=iv,
    )

    def loss(bc_vals_arg):
        return np.sum(solver(iv, initial, bc=bc.replace_vals(bc_vals_arg)) ** 2)

    analytic, fd, rel_err = _check_grad(loss, bc1.bc_vals)
    assert rel_err < 1e-4, (
        f"Linear solver bc_vals grad mismatch: rel_err={rel_err:.2e}\n"
        f"  analytic={analytic}\n  fd={fd}"
    )


@pytest.mark.cpu
def test_bc_grad_newton_jittable(elastic_setup):
    """Newton fori_loop (make_jittable=True): grad w.r.t. bc_vals matches FD."""
    problem, bc, iv, bc1, initial = elastic_setup
    solver = fe.create_solver(
        problem, bc,
        solver_options=fe.IterativeSolverOptions(solver='cg'),
        iter_num=3, internal_vars=iv,
        newton_options=NewtonOptions(make_jittable=True),
    )

    def loss(bc_vals_arg):
        return np.sum(solver(iv, initial, bc=bc.replace_vals(bc_vals_arg)) ** 2)

    analytic, fd, rel_err = _check_grad(loss, bc1.bc_vals)
    assert rel_err < 1e-4, (
        f"Newton jittable bc_vals grad mismatch: rel_err={rel_err:.2e}\n"
        f"  analytic={analytic}\n  fd={fd}"
    )


@pytest.mark.cpu
def test_bc_grad_newton_python_loop(elastic_setup):
    """Newton Python loop (make_jittable=False): grad w.r.t. bc_vals matches FD."""
    problem, bc, iv, bc1, initial = elastic_setup
    solver = fe.create_solver(
        problem, bc,
        solver_options=fe.IterativeSolverOptions(solver='cg'),
        iter_num=3, internal_vars=iv,
        newton_options=NewtonOptions(make_jittable=False),
    )

    def loss(bc_vals_arg):
        return np.sum(solver(iv, initial, bc=bc.replace_vals(bc_vals_arg)) ** 2)

    analytic, fd, rel_err = _check_grad(loss, bc1.bc_vals)
    assert rel_err < 1e-4, (
        f"Newton python-loop bc_vals grad mismatch: rel_err={rel_err:.2e}\n"
        f"  analytic={analytic}\n  fd={fd}"
    )


@pytest.mark.cpu
def test_bc_grad_newton_nonsymmetric(elastic_setup):
    """Newton with symmetric_bc=False: grad w.r.t. bc_vals matches FD."""
    problem, bc, iv, bc1, initial = elastic_setup
    solver = fe.create_solver(
        problem, bc,
        solver_options=fe.IterativeSolverOptions(solver='gmres'),
        iter_num=3, internal_vars=iv,
        newton_options=NewtonOptions(make_jittable=True),
        symmetric_bc=False,
    )

    def loss(bc_vals_arg):
        return np.sum(solver(iv, initial, bc=bc.replace_vals(bc_vals_arg)) ** 2)

    analytic, fd, rel_err = _check_grad(loss, bc1.bc_vals)
    assert rel_err < 1e-4, (
        f"Newton non-symmetric bc_vals grad mismatch: rel_err={rel_err:.2e}\n"
        f"  analytic={analytic}\n  fd={fd}"
    )
