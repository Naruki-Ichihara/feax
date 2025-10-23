"""
Examples demonstrating the symbolic DSL for defining physics in FEAX.

This file shows how to use the symbolic interface to define various PDEs
without writing low-level universal kernels.
"""

import sys
sys.path.insert(0, '/workspace')

from feax.experimental.symbolic import (
    TrialFunction, TestFunction, Constant, Identity,
    grad, div, inner, dot, sym, epsilon, sigma, advect,
    dx, ds
)


def example_poisson():
    """Poisson equation: -∇²u = f

    Weak form: ∫ ∇u·∇v dx = ∫ f·v dx
    """
    print("=" * 60)
    print("Example 1: Poisson Equation")
    print("=" * 60)

    # Define function spaces
    u = TrialFunction(vec=1, name='u')  # Scalar unknown
    v = TestFunction(vec=1, name='v')   # Scalar test function
    f = Constant(name='source', vec=1)  # Source term

    # Weak form
    F = inner(grad(u), grad(v)) * dx - f * v * dx

    print(f"Trial function: {u}")
    print(f"Test function: {v}")
    print(f"Weak form: {F}")
    print(f"Form type: {type(F).__name__}")
    print()


def example_linear_elasticity():
    """Linear elasticity: -∇·σ(u) = f

    where σ(u) = 2με(u) + λtr(ε)I
    and ε(u) = (∇u + ∇u^T)/2

    Weak form: ∫ σ(u):ε(v) dx = ∫ f·v dx
    """
    print("=" * 60)
    print("Example 2: Linear Elasticity")
    print("=" * 60)

    # Define function spaces
    u = TrialFunction(vec=3, name='displacement')
    v = TestFunction(vec=3, name='v')

    # Material properties
    mu = Constant(name='shear_modulus')
    lmbda = Constant(name='lame_lambda')
    f = Constant(name='body_force', vec=3)

    # Strain tensor
    eps_u = epsilon(u)  # or: sym(grad(u))
    eps_v = epsilon(v)

    # Stress tensor (isotropic linear elasticity)
    from feax.experimental.symbolic import tr
    sigma_u = 2 * mu * eps_u + lmbda * tr(eps_u) * Identity(3)

    # Weak form
    F = inner(sigma_u, eps_v) * dx - inner(f, v) * dx

    print(f"Displacement: {u}")
    print(f"Strain: {eps_u}")
    print(f"Stress: {sigma_u}")
    print(f"Weak form: {F}")
    print()


def example_stokes():
    """Stokes flow (steady incompressible viscous flow)

    Momentum: -∇·σ(u,p) = f  where σ = 2με(u) - pI
    Continuity: ∇·u = 0

    Weak form:
        ∫ σ(u,p):ε(v) dx + ∫ q·∇·u dx = ∫ f·v dx
    """
    print("=" * 60)
    print("Example 3: Stokes Flow")
    print("=" * 60)

    # Define function spaces (mixed formulation)
    u = TrialFunction(vec=3, name='velocity', index=0)
    p = TrialFunction(vec=1, name='pressure', index=1)
    v = TestFunction(vec=3, name='v', index=0)
    q = TestFunction(vec=1, name='q', index=1)

    # Material properties
    mu = Constant(name='viscosity')
    f = Constant(name='body_force', vec=3)

    # Stress tensor (using helper function)
    stress = sigma(u, p, mu)  # 2με(u) - pI

    # Weak form
    F = (
        inner(stress, epsilon(v)) * dx +  # Momentum equation
        q * div(u) * dx -                  # Continuity equation
        inner(f, v) * dx                   # Body force
    )

    print(f"Velocity: {u}")
    print(f"Pressure: {p}")
    print(f"Stress: {stress}")
    print(f"Weak form: {F}")
    print()


def example_navier_stokes():
    """Navier-Stokes (incompressible flow with convection)

    Momentum: ρ(u·∇)u - ∇·σ(u,p) = f
    Continuity: ∇·u = 0

    Weak form:
        ∫ ρ(u·∇)u·v dx + ∫ σ(u,p):ε(v) dx + ∫ q·∇·u dx = ∫ f·v dx
    """
    print("=" * 60)
    print("Example 4: Navier-Stokes (with convection)")
    print("=" * 60)

    # Define function spaces
    u = TrialFunction(vec=3, name='velocity', index=0)
    p = TrialFunction(vec=1, name='pressure', index=1)
    v = TestFunction(vec=3, name='v', index=0)
    q = TestFunction(vec=1, name='q', index=1)

    # Material properties
    rho = Constant(name='density')
    mu = Constant(name='viscosity')
    f = Constant(name='body_force', vec=3)

    # Weak form with convection term
    F = (
        rho * inner(advect(u), v) * dx +   # Convection (nonlinear!)
        inner(sigma(u, p, mu), epsilon(v)) * dx +  # Viscous stress
        q * div(u) * dx -                          # Incompressibility
        inner(f, v) * dx                           # Body force
    )

    print(f"Convection term: {advect(u)}")
    print(f"Full form: {F}")
    print()


def example_with_boundary_conditions():
    """Example with surface integrals (Neumann BC)

    Poisson with surface flux:
        -∇²u = f  in Ω
        ∇u·n = g  on Γ

    Weak form:
        ∫ ∇u·∇v dx = ∫ f·v dx + ∫ g·v ds
    """
    print("=" * 60)
    print("Example 5: With Surface Integrals")
    print("=" * 60)

    # Define function spaces
    u = TrialFunction(vec=1, name='u')
    v = TestFunction(vec=1, name='v')
    f = Constant(name='source')
    g = Constant(name='flux')  # Surface flux

    # Weak form with surface integral
    F = (
        inner(grad(u), grad(v)) * dx -
        f * v * dx -
        g * v * ds  # Surface integral
    )

    print(f"Volume integral: ∫ ∇u·∇v dx")
    print(f"Surface integral: ∫ g·v ds")
    print(f"Full form: {F}")
    print()


def example_expression_tree():
    """Show the expression tree structure"""
    print("=" * 60)
    print("Example 6: Expression Tree Structure")
    print("=" * 60)

    u = TrialFunction(vec=3, name='u')
    v = TestFunction(vec=3, name='v')

    # Build expression step by step
    grad_u = grad(u)
    grad_v = grad(v)
    eps_u = sym(grad_u)
    eps_v = sym(grad_v)
    integrand = inner(eps_u, eps_v)
    form = integrand * dx

    print(f"u: {u}")
    print(f"  rank={u.rank}, shape={u.shape}")
    print()
    print(f"grad(u): {grad_u}")
    print(f"  rank={grad_u.rank}, shape={grad_u.shape}")
    print()
    print(f"sym(grad(u)): {eps_u}")
    print(f"  rank={eps_u.rank}, shape={eps_u.shape}")
    print()
    print(f"inner(eps_u, eps_v): {integrand}")
    print(f"  rank={integrand.rank}, shape={integrand.shape}")
    print()
    print(f"∫...dx: {form}")
    print(f"  type={type(form).__name__}")
    print()


def example_arithmetic():
    """Show arithmetic operations"""
    print("=" * 60)
    print("Example 7: Arithmetic Operations")
    print("=" * 60)

    u = TrialFunction(vec=3, name='u')
    p = TrialFunction(vec=1, name='pressure')

    mu = Constant(name='mu')
    lmbda = Constant(name='lambda')

    # Various arithmetic
    expr1 = 2 * mu  # Python scalar * Constant
    expr2 = mu + lmbda  # Constant + Constant
    expr3 = 2 * mu * epsilon(u)  # Scalar * tensor
    expr4 = -p * Identity(3)  # Negative * tensor
    expr5 = expr3 + expr4  # Add tensors

    print(f"2*mu: {expr1}")
    print(f"mu + lambda: {expr2}")
    print(f"2*mu*epsilon(u): {expr3}")
    print(f"-p*I: {expr4}")
    print(f"2*mu*eps(u) - p*I: {expr5}")
    print()


def example_multiple_integrals():
    """Show sum of multiple integrals"""
    print("=" * 60)
    print("Example 8: Multiple Integrals")
    print("=" * 60)

    u = TrialFunction(vec=3, name='u')
    v = TestFunction(vec=3, name='v')

    # Multiple volume integrals
    term1 = inner(grad(u), grad(v)) * dx
    term2 = inner(u, v) * dx

    # Multiple surface integrals
    traction = Constant(name='traction', vec=3)
    term3 = inner(traction, v) * ds(0)  # Boundary 0
    term4 = inner(traction, v) * ds(1)  # Boundary 1

    # Combine
    F = term1 + term2 + term3 + term4

    print(f"Form with multiple integrals: {F}")
    print(f"Type: {type(F).__name__}")

    if hasattr(F, 'integrals'):
        print(f"Number of integrals: {len(F.integrals)}")
        for i, integral in enumerate(F.integrals):
            print(f"  Integral {i}: measure={integral.measure}, boundary_id={integral.boundary_id}")
    print()


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  FEAX Symbolic DSL Examples                            ║")
    print("╚" + "=" * 58 + "╝")
    print()

    example_poisson()
    example_linear_elasticity()
    example_stokes()
    example_navier_stokes()
    example_with_boundary_conditions()
    example_expression_tree()
    example_arithmetic()
    example_multiple_integrals()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Implement SymbolicProblem class to compile these forms")
    print("2. Create code generator to convert symbolic → universal kernel")
    print("3. Add support for nonlinear problems (Newton iteration)")
    print()
