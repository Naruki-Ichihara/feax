"""Pure-Neumann regression tests."""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import feax as fe
from feax.solvers.nullspace import mass_lumped_weights


_TWO_D_ELEMENTS = ("TRI3", "TRI6", "QUAD4", "QUAD8", "QUAD9")
_THREE_D_ELEMENTS = ("TET4", "TET10", "HEX8", "HEX20")


class _Poisson(fe.Problem):
    def get_tensor_map(self):
        return lambda u_grad, *args: u_grad


class _FourFluxPoisson(_Poisson):
    """Poisson with left/right flux maps on two disconnected components."""

    def get_surface_maps(self):
        def flux_map(u, x, residual_flux):
            return residual_flux * jnp.ones_like(u)
        return [flux_map] * 4


class _EightFluxPoisson(_Poisson):
    """Poisson with all faces loaded on two disconnected tetrahedra."""

    def get_surface_maps(self):
        def flux_map(u, x, residual_flux):
            return residual_flux * jnp.ones_like(u)
        return [flux_map] * 8


class _ConductivityFluxPoisson(_Poisson):
    """EIT conductivity equation ``div(sigma * grad(u)) = 0``."""

    def get_tensor_map(self):
        return lambda u_grad, conductivity: conductivity * u_grad

    def get_surface_maps(self):
        def flux_map(u, x, residual_flux):
            return residual_flux * jnp.ones_like(u)
        return [flux_map, flux_map]


def _solver(problem, bc):
    return fe.create_solver(
        problem,
        bc,
        linear=True,
        solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, maxiter=200),
        nullspace=fe.NullspaceConstraint.constant_mean_zero(),
        return_solution=False,
    )


def _combine_meshes(first, second, element_type):
    return fe.Mesh(
        onp.concatenate((first.points, second.points)),
        onp.concatenate((first.cells, second.cells + len(first.points))),
        ele_type=element_type,
    )


def _triangle_square_mesh(origin, element_type):
    points = [(origin, 0.0), (origin + 1.0, 0.0),
              (origin + 1.0, 1.0), (origin, 1.0)]
    triangles = ((0, 1, 2), (0, 2, 3))
    if element_type == "TRI3":
        return fe.Mesh(onp.asarray(points), onp.asarray(triangles, dtype=onp.int32), ele_type=element_type)

    edge_nodes, cells = {}, []
    for triangle in triangles:
        midsides = []
        for left, right in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((left, right)))
            if edge not in edge_nodes:
                edge_nodes[edge] = len(points)
                points.append(tuple((onp.asarray(points[left]) + onp.asarray(points[right])) / 2.0))
            midsides.append(edge_nodes[edge])
        cells.append((*triangle, *midsides))
    return fe.Mesh(onp.asarray(points), onp.asarray(cells, dtype=onp.int32), ele_type=element_type)


def _two_component_2d_problem(element_type):
    if element_type.startswith("QUAD"):
        first = fe.mesh.rectangle_mesh(1, 1, ele_type=element_type)
        second = fe.mesh.rectangle_mesh(1, 1, origin=(3.0, 0.0), ele_type=element_type)
    else:
        first = _triangle_square_mesh(0.0, element_type)
        second = _triangle_square_mesh(3.0, element_type)
    mesh = _combine_meshes(first, second, element_type)
    locations = [
        lambda point: jnp.isclose(point[0], 0.0, atol=1e-12),
        lambda point: jnp.isclose(point[0], 1.0, atol=1e-12),
        lambda point: jnp.isclose(point[0], 3.0, atol=1e-12),
        lambda point: jnp.isclose(point[0], 4.0, atol=1e-12),
    ]
    problem = _FourFluxPoisson(mesh, vec=1, dim=2, ele_type=element_type, location_fns=locations)
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    surface_vars = [
        (fe.TracedParams.create_uniform_surface_var(problem, flux, surface_index=index),)
        for index, flux in enumerate((1.0, -1.0, 1.0, -1.0))
    ]
    return problem, bc, fe.TracedParams(surface_vars=surface_vars), (0.5, 3.5)


def _hex20_component(origin):
    corners = onp.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
    ])
    midsides = onp.array([
        [.5, 0., 0.], [1., .5, 0.], [.5, 1., 0.], [0., .5, 0.],
        [.5, 0., 1.], [1., .5, 1.], [.5, 1., 1.], [0., .5, 1.],
        [0., 0., .5], [1., 0., .5], [1., 1., .5], [0., 1., .5],
    ])
    return fe.Mesh(onp.concatenate((corners, midsides)) + onp.asarray(origin),
                   onp.arange(20, dtype=onp.int32)[None, :], ele_type="HEX20")


def _tetra_component(origin, element_type):
    points = [(origin, 0.0, 0.0), (origin + 1.0, 0.0, 0.0),
              (origin, 1.0, 0.0), (origin, 0.0, 1.0)]
    if element_type == "TET4":
        return fe.Mesh(onp.asarray(points), onp.asarray([[0, 1, 2, 3]], dtype=onp.int32), ele_type=element_type)
    edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    points.extend(tuple((onp.asarray(points[a]) + onp.asarray(points[b])) / 2.0) for a, b in edges)
    return fe.Mesh(onp.asarray(points), onp.arange(10, dtype=onp.int32)[None, :], ele_type=element_type)


def _two_component_3d_problem(element_type):
    if element_type.startswith("HEX"):
        if element_type == "HEX20":
            first, second = _hex20_component((0.0, 0.0, 0.0)), _hex20_component((3.0, 0.0, 0.0))
        else:
            points = onp.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                                [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
            first = fe.Mesh(points, onp.arange(8, dtype=onp.int32)[None, :], ele_type="HEX8")
            second = fe.Mesh(points + onp.array([3., 0., 0.]), first.cells, ele_type="HEX8")
        locations = [
            lambda point: jnp.isclose(point[0], 0.0, atol=1e-12),
            lambda point: jnp.isclose(point[0], 1.0, atol=1e-12),
            lambda point: jnp.isclose(point[0], 3.0, atol=1e-12),
            lambda point: jnp.isclose(point[0], 4.0, atol=1e-12),
        ]
        problem_class, fluxes, offsets = _FourFluxPoisson, (1., -1., 1., -1.), (0.5, 3.5)
    else:
        first, second = _tetra_component(0.0, element_type), _tetra_component(3.0, element_type)
        locations = [
            lambda point: jnp.isclose(jnp.sum(point), 1.0, atol=1e-12),
            lambda point: jnp.isclose(point[0], 0.0, atol=1e-12),
            lambda point: jnp.isclose(point[1], 0.0, atol=1e-12),
            lambda point: jnp.isclose(point[2], 0.0, atol=1e-12),
            lambda point: jnp.isclose(jnp.sum(point), 4.0, atol=1e-12),
            lambda point: jnp.isclose(point[0], 3.0, atol=1e-12),
            lambda point: jnp.isclose(point[1], 0.0, atol=1e-12),
            lambda point: jnp.isclose(point[2], 0.0, atol=1e-12),
        ]
        problem_class = _EightFluxPoisson
        fluxes, offsets = (-1.0 / onp.sqrt(3.0), 1., 0., 0.) * 2, (0.25, 3.25)
    mesh = _combine_meshes(first, second, element_type)
    problem = problem_class(mesh, vec=1, dim=3, ele_type=element_type, location_fns=locations)
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    surface_vars = [
        (fe.TracedParams.create_uniform_surface_var(problem, flux, surface_index=index),)
        for index, flux in enumerate(fluxes)
    ]
    return problem, bc, fe.TracedParams(surface_vars=surface_vars), offsets


def _eit_problem():
    mesh = fe.mesh.rectangle_mesh(12, 12, domain_x=1.0, domain_y=1.0,
                                  origin=(-0.5, -0.5), ele_type="QUAD4")
    locations = [
        lambda point: jnp.isclose(point[0], -0.5, atol=1e-12),
        lambda point: jnp.isclose(point[0], 0.5, atol=1e-12),
    ]
    problem = _ConductivityFluxPoisson(mesh, vec=1, dim=2, ele_type="QUAD4", location_fns=locations)
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    centers = onp.asarray(mesh.points)[onp.asarray(mesh.cells)].mean(axis=1)
    conductivity = jnp.where(jnp.sum(jnp.asarray(centers) ** 2, axis=1) <= 0.20 ** 2, 2.0, 0.5)
    surface_vars = [
        (fe.TracedParams.create_uniform_surface_var(problem, flux, surface_index=index),)
        for index, flux in enumerate((1.0, -1.0))
    ]
    return problem, bc, fe.TracedParams(volume_vars=(conductivity,), surface_vars=surface_vars), conductivity


@pytest.mark.parametrize("element_type", _TWO_D_ELEMENTS + _THREE_D_ELEMENTS)
def test_pure_neumann_manufactured_solution_for_every_element_family(element_type):
    if element_type in _TWO_D_ELEMENTS:
        problem, bc, traced_params, offsets = _two_component_2d_problem(element_type)
    else:
        problem, bc, traced_params, offsets = _two_component_3d_problem(element_type)
    solver = _solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)
    residual = fe.create_res_bc_function(problem, bc)
    points = jnp.asarray(problem.fes[0].points)
    first = points[:, 0] < 2.0
    weights = mass_lumped_weights(problem)

    load = -residual(initial, traced_params)
    assert abs(float(jnp.sum(load[first]))) < 1e-12
    assert abs(float(jnp.sum(load[~first]))) < 1e-12

    sol = solver(traced_params, initial)
    expected = jnp.where(first, points[:, 0] - offsets[0], points[:, 0] - offsets[1])
    assert onp.allclose(sol, expected, atol=1e-9)
    assert abs(float(jnp.vdot(weights[first], sol[first]))) < 1e-12
    assert abs(float(jnp.vdot(weights[~first], sol[~first]))) < 1e-12
    assert float(jnp.max(jnp.abs(residual(sol, traced_params)))) < 1e-9


def test_eit_flux_loads_and_conductivity_are_differentiable():
    problem, bc, traced_params, conductivity = _eit_problem()
    solver = _solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)
    residual = fe.create_res_bc_function(problem, bc)
    points = jnp.asarray(problem.fes[0].points)
    left_nodes = jnp.isclose(points[:, 0], -0.5, atol=1e-12)
    right_nodes = jnp.isclose(points[:, 0], 0.5, atol=1e-12)

    load = -residual(initial, traced_params)
    assert abs(float(jnp.sum(load))) < 1e-12
    sol = solver(traced_params, initial)
    weights = mass_lumped_weights(problem)
    assert abs(float(jnp.vdot(weights, sol))) < 1e-12
    assert float(jnp.max(jnp.abs(residual(sol, traced_params)))) < 1e-9

    uniform = fe.TracedParams(volume_vars=(jnp.full_like(conductivity, 0.5),),
                              surface_vars=traced_params.surface_vars)
    voltage_drop = jnp.mean(sol[right_nodes]) - jnp.mean(sol[left_nodes])
    uniform_sol = solver(uniform, initial)
    uniform_drop = jnp.mean(uniform_sol[right_nodes]) - jnp.mean(uniform_sol[left_nodes])
    assert 0.0 < float(voltage_drop) < float(uniform_drop)

    def solve_at_scale(scale):
        params = fe.TracedParams(volume_vars=(conductivity,),
                                 surface_vars=[(surface[0] * scale,) for surface in traced_params.surface_vars])
        return solver(params, initial)

    batched = jax.jit(jax.vmap(solve_at_scale))(jnp.array([0.5, 1.0]))
    assert onp.allclose(batched[0] * 2.0, batched[1], atol=1e-8)

    inclusion = conductivity == 2.0

    def voltage_at(inclusion_conductivity):
        params = fe.TracedParams(
            volume_vars=(jnp.where(inclusion, inclusion_conductivity, 0.5),),
            surface_vars=traced_params.surface_vars,
        )
        solution = solver(params, initial)
        return jnp.mean(solution[right_nodes]) - jnp.mean(solution[left_nodes])

    step = 1e-4
    finite_difference = (voltage_at(2.0 + step) - voltage_at(2.0 - step)) / (2.0 * step)
    assert abs(float(jax.grad(voltage_at)(2.0) - finite_difference)) < 2e-4


def test_pure_neumann_requires_explicit_cg():
    problem, bc, _, _ = _two_component_2d_problem("QUAD4")
    with pytest.raises(ValueError, match="explicit KrylovSolverOptions"):
        fe.create_solver(problem, bc, linear=True,
                         nullspace=fe.NullspaceConstraint.constant_mean_zero())


def test_pure_neumann_public_entry_points_support_traced_structure():
    problem, bc, traced_params, _ = _two_component_2d_problem("QUAD4")
    initial = fe.zero_like_initial_guess(problem, bc)
    weights = mass_lumped_weights(problem)
    ts = fe.TracedStructure.from_problem(problem)
    options = fe.KrylovSolverOptions(solver="cg", tol=1e-10, maxiter=200)
    constraint = fe.NullspaceConstraint.constant_mean_zero()

    wrapped = fe.create_solver(problem, bc, linear=True, solver_options=options,
                               nullspace=constraint, traced_structure=ts)
    solution = wrapped(traced_params, initial, traced_structure=ts)
    assert abs(float(jnp.vdot(weights, solution.dofs))) < 1e-12

    linear = fe.create_linear_solver(problem, bc, solver_options=options,
                                     nullspace=constraint, traced_structure=ts)
    raw_solution = linear(traced_params, initial, traced_structure=ts)
    assert onp.allclose(raw_solution, solution.dofs, atol=1e-9)
