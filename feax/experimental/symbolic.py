"""
Symbolic DSL for defining finite element weak forms.

This module provides a lightweight symbolic interface for defining physics
in FEAX, similar to UFL in FEniCS but built purely on JAX. The symbolic
expressions are compiled to efficient JAX kernels via the universal kernel
mechanism.

Example
-------
>>> from feax.symbolic import TrialFunction, TestFunction, grad, div, inner, dx
>>>
>>> # Stokes flow
>>> u = TrialFunction(vec=3, name='velocity')
>>> p = TrialFunction(vec=1, name='pressure')
>>> v = TestFunction(vec=3, name='v')
>>> q = TestFunction(vec=1, name='q')
>>>
>>> mu = Constant(name='viscosity')
>>>
>>> # Weak form
>>> F = (
>>>     inner(grad(u), grad(v)) * dx +
>>>     div(u) * q * dx +
>>>     p * div(v) * dx
>>> )
"""

import jax.numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional, Any
from enum import Enum


# ============================================================================
# Expression Type System
# ============================================================================

class TensorRank(Enum):
    """Tensor rank enumeration for type checking."""
    SCALAR = 0
    VECTOR = 1
    TENSOR = 2


# ============================================================================
# Base Symbolic Expression Classes
# ============================================================================

@dataclass
class Expr:
    """Base class for all symbolic expressions.

    Attributes
    ----------
    rank : TensorRank
        Tensor rank of the expression (scalar=0, vector=1, tensor=2)
    shape : tuple
        Shape of the expression (e.g., (3,) for vector, (3,3) for tensor)
    """
    rank: TensorRank = field(default=TensorRank.SCALAR, init=False)
    shape: tuple = field(default=(), init=False)

    def __add__(self, other):
        return Add(self, _ensure_expr(other))

    def __radd__(self, other):
        return Add(_ensure_expr(other), self)

    def __sub__(self, other):
        return Sub(self, _ensure_expr(other))

    def __rsub__(self, other):
        return Sub(_ensure_expr(other), self)

    def __mul__(self, other):
        # Special handling for integration measures
        if isinstance(other, (_DX, _DS, _DSBoundary)):
            return other.__rmul__(self)
        return Mul(self, _ensure_expr(other))

    def __rmul__(self, other):
        return Mul(_ensure_expr(other), self)

    def __truediv__(self, other):
        return Division(self, _ensure_expr(other))

    def __neg__(self):
        return Mul(ScalarConstant(-1.0), self)


def _ensure_expr(obj) -> Expr:
    """Convert Python scalars to Expr objects."""
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, (int, float)):
        return ScalarConstant(float(obj))
    else:
        raise TypeError(f"Cannot convert {type(obj)} to Expr")


# ============================================================================
# Function Space Symbols
# ============================================================================

@dataclass
class TrialFunction(Expr):
    """Trial function (unknown) in the weak form.

    Parameters
    ----------
    vec : int
        Vector dimension (1 for scalar, 3 for 3D vector)
    name : str
        Name of the function (for code generation)
    index : int, optional
        Index in multi-variable problems (default: 0)

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='velocity')  # Vector field
    >>> p = TrialFunction(vec=1, name='pressure')  # Scalar field
    """
    vec: int
    name: str
    index: int = 0

    def __post_init__(self):
        if self.vec == 1:
            self.rank = TensorRank.SCALAR
            self.shape = ()
        else:
            self.rank = TensorRank.VECTOR
            self.shape = (self.vec,)


@dataclass
class TestFunction(Expr):
    """Test function in the weak form.

    Parameters
    ----------
    vec : int
        Vector dimension (1 for scalar, 3 for 3D vector)
    name : str
        Name of the function (for code generation)
    index : int, optional
        Index in multi-variable problems (default: 0)

    Examples
    --------
    >>> v = TestFunction(vec=3, name='v')  # Vector test function
    >>> q = TestFunction(vec=1, name='q')  # Scalar test function
    """
    vec: int
    name: str
    index: int = 0

    def __post_init__(self):
        if self.vec == 1:
            self.rank = TensorRank.SCALAR
            self.shape = ()
        else:
            self.rank = TensorRank.VECTOR
            self.shape = (self.vec,)


@dataclass
class Constant(Expr):
    """Constant value or internal variable.

    Parameters
    ----------
    name : str
        Name of the constant (for internal variable lookup)
    vec : int, optional
        Vector dimension (default: 1 for scalar)
    value : float or array, optional
        Constant value (if not using internal variables)

    Examples
    --------
    >>> mu = Constant(name='viscosity')  # Scalar from internal vars
    >>> f = Constant(name='force', vec=3)  # Vector from internal vars
    >>> rho = Constant(name='density', value=1000.0)  # Fixed value
    """
    name: str
    vec: int = 1
    value: Optional[Union[float, np.ndarray]] = None

    def __post_init__(self):
        if self.vec == 1:
            self.rank = TensorRank.SCALAR
            self.shape = ()
        else:
            self.rank = TensorRank.VECTOR
            self.shape = (self.vec,)


@dataclass
class ScalarConstant(Expr):
    """Literal scalar constant (e.g., 2.0, -1.0).

    Used internally for arithmetic with Python numbers.
    """
    value: float

    def __post_init__(self):
        self.rank = TensorRank.SCALAR
        self.shape = ()


# ============================================================================
# Differential Operators
# ============================================================================

@dataclass
class Grad(Expr):
    """Gradient operator: ∇u

    Parameters
    ----------
    operand : Expr
        Expression to take gradient of

    Notes
    -----
    - Scalar → Vector: grad(p) has shape (dim,)
    - Vector → Tensor: grad(u) has shape (vec, dim)

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> grad_u = grad(u)  # Shape (3, 3) - velocity gradient tensor
    """
    operand: Expr

    def __post_init__(self):
        # Infer dimension from vec (2D or 3D)
        dim = getattr(self.operand, 'vec', 3) if self.operand.rank == TensorRank.VECTOR else 3

        if self.operand.rank == TensorRank.SCALAR:
            self.rank = TensorRank.VECTOR
            self.shape = (dim,)
        elif self.operand.rank == TensorRank.VECTOR:
            self.rank = TensorRank.TENSOR
            self.shape = (self.operand.vec, dim)
        else:
            raise ValueError("Cannot take gradient of tensor")


@dataclass
class Div(Expr):
    """Divergence operator: ∇·u

    Parameters
    ----------
    operand : Expr
        Vector expression to take divergence of

    Notes
    -----
    - Vector → Scalar: div(u) is scalar
    - Tensor → Vector: div(T) has shape (vec,)

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> div_u = div(u)  # Scalar - incompressibility constraint
    """
    operand: Expr

    def __post_init__(self):
        if self.operand.rank == TensorRank.VECTOR:
            self.rank = TensorRank.SCALAR
            self.shape = ()
        elif self.operand.rank == TensorRank.TENSOR:
            self.rank = TensorRank.VECTOR
            # Shape is first dimension of tensor
            self.shape = (self.operand.shape[0],) if self.operand.shape else (3,)
        else:
            raise ValueError("Can only take divergence of vector or tensor")


@dataclass
class Sym(Expr):
    """Symmetric part of tensor: (T + T^T) / 2

    Parameters
    ----------
    operand : Expr
        Tensor expression to symmetrize

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> epsilon = sym(grad(u))  # Strain tensor
    """
    operand: Expr

    def __post_init__(self):
        if self.operand.rank != TensorRank.TENSOR:
            raise ValueError("Can only symmetrize tensors")
        self.rank = TensorRank.TENSOR
        self.shape = self.operand.shape


@dataclass
class Transpose(Expr):
    """Transpose operator: T^T

    Parameters
    ----------
    operand : Expr
        Tensor expression to transpose
    """
    operand: Expr

    def __post_init__(self):
        if self.operand.rank != TensorRank.TENSOR:
            raise ValueError("Can only transpose tensors")
        self.rank = TensorRank.TENSOR
        # Swap dimensions
        if len(self.operand.shape) == 2:
            self.shape = (self.operand.shape[1], self.operand.shape[0])
        else:
            self.shape = self.operand.shape


@dataclass
class Trace(Expr):
    """Trace of tensor: tr(T)

    Parameters
    ----------
    operand : Expr
        Tensor expression to take trace of

    Examples
    --------
    >>> sigma = ...  # Stress tensor
    >>> pressure = tr(sigma) / 3  # Mean stress
    """
    operand: Expr

    def __post_init__(self):
        if self.operand.rank != TensorRank.TENSOR:
            raise ValueError("Can only take trace of tensors")
        self.rank = TensorRank.SCALAR
        self.shape = ()


# ============================================================================
# Algebraic Operations
# ============================================================================

@dataclass
class Inner(Expr):
    """Inner product: a·b or A:B

    Parameters
    ----------
    left : Expr
        Left operand
    right : Expr
        Right operand

    Notes
    -----
    - Vector·Vector → Scalar: u·v
    - Tensor:Tensor → Scalar: σ:ε (double contraction)

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> v = TestFunction(vec=3, name='v')
    >>> inner(u, v)  # Scalar product
    >>>
    >>> sigma = ...
    >>> epsilon = ...
    >>> inner(sigma, epsilon)  # Stress-strain work
    """
    left: Expr
    right: Expr

    def __post_init__(self):
        if self.left.rank != self.right.rank:
            raise ValueError(f"Inner product requires same rank: {self.left.rank} vs {self.right.rank}")
        self.rank = TensorRank.SCALAR
        self.shape = ()


@dataclass
class Dot(Expr):
    """Dot product: a·b or a·T (vector with rows of tensor)

    Parameters
    ----------
    left : Expr
        Vector operand
    right : Expr
        Vector or tensor operand

    Notes
    -----
    - Vector·Vector → Scalar: u·v
    - Vector·Tensor → Vector: u·∇u (convection term)

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> grad_u = grad(u)
    >>> dot(u, grad_u)  # (u·∇)u for Navier-Stokes convection
    """
    left: Expr
    right: Expr

    def __post_init__(self):
        if self.left.rank != TensorRank.VECTOR:
            raise ValueError("Left operand of dot must be vector")

        if self.right.rank == TensorRank.VECTOR:
            # Vector · Vector → Scalar
            self.rank = TensorRank.SCALAR
            self.shape = ()
        elif self.right.rank == TensorRank.TENSOR:
            # Vector · Tensor → Vector (dot with each row)
            self.rank = TensorRank.VECTOR
            self.shape = (self.right.shape[0],) if self.right.shape else (3,)
        else:
            raise ValueError("Right operand of dot must be vector or tensor")


@dataclass
class Outer(Expr):
    """Outer product: a ⊗ b

    Parameters
    ----------
    left : Expr
        Left vector
    right : Expr
        Right vector

    Notes
    -----
    Vector ⊗ Vector → Tensor

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> v = TestFunction(vec=3, name='v')
    >>> outer(u, v)  # Tensor (3, 3)
    """
    left: Expr
    right: Expr

    def __post_init__(self):
        if self.left.rank != TensorRank.VECTOR or self.right.rank != TensorRank.VECTOR:
            raise ValueError("Outer product requires two vectors")
        self.rank = TensorRank.TENSOR
        self.shape = (self.left.shape[0] if self.left.shape else 3,
                      self.right.shape[0] if self.right.shape else 3)


# ============================================================================
# Arithmetic Operations
# ============================================================================

@dataclass
class Add(Expr):
    """Addition: a + b"""
    left: Expr
    right: Expr

    def __post_init__(self):
        # Result has same rank/shape as operands
        self.rank = self.left.rank
        self.shape = self.left.shape


@dataclass
class Sub(Expr):
    """Subtraction: a - b"""
    left: Expr
    right: Expr

    def __post_init__(self):
        self.rank = self.left.rank
        self.shape = self.left.shape


@dataclass
class Mul(Expr):
    """Multiplication: a * b (scalar multiplication or element-wise)"""
    left: Expr
    right: Expr

    def __post_init__(self):
        # At least one must be scalar for now
        if self.left.rank == TensorRank.SCALAR:
            self.rank = self.right.rank
            self.shape = self.right.shape
        elif self.right.rank == TensorRank.SCALAR:
            self.rank = self.left.rank
            self.shape = self.left.shape
        else:
            # Element-wise multiplication (same rank)
            if self.left.rank != self.right.rank:
                raise ValueError("Element-wise multiplication requires same rank")
            self.rank = self.left.rank
            self.shape = self.left.shape


@dataclass
class Division(Expr):
    """Division: a / b (scalar division)"""
    left: Expr
    right: Expr

    def __post_init__(self):
        if self.right.rank != TensorRank.SCALAR:
            raise ValueError("Can only divide by scalar")
        self.rank = self.left.rank
        self.shape = self.left.shape


# ============================================================================
# Special Tensors
# ============================================================================

@dataclass
class Identity(Expr):
    """Identity tensor: I

    Parameters
    ----------
    dim : int
        Dimension (e.g., 3 for 3D)

    Examples
    --------
    >>> I = Identity(3)
    >>> sigma = 2*mu*epsilon - p*I  # Stress tensor
    """
    dim: int

    def __post_init__(self):
        self.rank = TensorRank.TENSOR
        self.shape = (self.dim, self.dim)


# ============================================================================
# Integration Measures
# ============================================================================

@dataclass
class Integral(Expr):
    """Integral expression: ∫ expr dΩ or ∫ expr dΓ

    Parameters
    ----------
    integrand : Expr
        Expression to integrate
    measure : str
        Integration measure ('dx' for volume, 'ds' for surface)
    boundary_id : int, optional
        Boundary identifier for surface integrals

    Notes
    -----
    This is the top-level expression returned when using dx or ds.
    """
    integrand: Expr
    measure: str  # 'dx' or 'ds'
    boundary_id: Optional[int] = None

    def __post_init__(self):
        self.rank = self.integrand.rank
        self.shape = self.integrand.shape

    def __add__(self, other):
        """Allow adding integrals: ∫ a dx + ∫ b dx"""
        if isinstance(other, Integral):
            return IntegralSum([self, other])
        return super().__add__(other)


@dataclass
class IntegralSum(Expr):
    """Sum of multiple integrals.

    Used to represent forms like: F = ∫ a dx + ∫ b dx + ∫ c ds
    """
    integrals: List[Integral]

    def __add__(self, other):
        if isinstance(other, Integral):
            return IntegralSum(self.integrals + [other])
        elif isinstance(other, IntegralSum):
            return IntegralSum(self.integrals + other.integrals)
        return super().__add__(other)


class _DX:
    """Volume integral measure: dx

    Examples
    --------
    >>> F = inner(grad(u), grad(v)) * dx
    """
    def __rmul__(self, integrand: Expr) -> Integral:
        return Integral(integrand, 'dx')


class _DS:
    """Surface integral measure: ds

    Parameters
    ----------
    boundary_id : int, optional
        Boundary marker (corresponds to location_fns index)

    Examples
    --------
    >>> F = inner(traction, v) * ds
    >>> F = inner(traction, v) * ds(0)  # Specific boundary
    """
    def __call__(self, boundary_id: int = 0):
        return _DSBoundary(boundary_id)

    def __rmul__(self, integrand: Expr) -> Integral:
        return Integral(integrand, 'ds', boundary_id=0)


class _DSBoundary:
    """Surface measure for specific boundary."""
    def __init__(self, boundary_id: int):
        self.boundary_id = boundary_id

    def __rmul__(self, integrand: Expr) -> Integral:
        return Integral(integrand, 'ds', boundary_id=self.boundary_id)


# Singleton instances
dx = _DX()
ds = _DS()


# ============================================================================
# Operator Functions (User API)
# ============================================================================

def grad(expr: Expr) -> Grad:
    """Gradient operator: ∇expr

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> grad(u)  # Velocity gradient tensor
    """
    return Grad(expr)


def div(expr: Expr) -> Div:
    """Divergence operator: ∇·expr

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> div(u)  # Scalar divergence (incompressibility)
    """
    return Div(expr)


def sym(expr: Expr) -> Sym:
    """Symmetric part: (T + T^T) / 2

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> epsilon = sym(grad(u))  # Strain tensor
    """
    return Sym(expr)


def transpose(expr: Expr) -> Transpose:
    """Transpose: T^T

    Examples
    --------
    >>> T = grad(u)
    >>> transpose(T)
    """
    return Transpose(expr)


def tr(expr: Expr) -> Trace:
    """Trace: tr(T)

    Examples
    --------
    >>> epsilon = sym(grad(u))
    >>> vol_strain = tr(epsilon)  # Volumetric strain
    """
    return Trace(expr)


def inner(left: Expr, right: Expr) -> Inner:
    """Inner product: a·b or A:B

    Examples
    --------
    >>> inner(u, v)  # Vector dot product
    >>> inner(sigma, epsilon)  # Tensor double contraction
    """
    return Inner(left, right)


def dot(left: Expr, right: Expr) -> Dot:
    """Dot product: a·b or a·T

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='u')
    >>> dot(u, grad(u))  # Convection term (u·∇)u
    """
    return Dot(left, right)


def outer(left: Expr, right: Expr) -> Outer:
    """Outer product: a ⊗ b

    Examples
    --------
    >>> outer(u, v)  # Tensor from two vectors
    """
    return Outer(left, right)


# ============================================================================
# Helper Functions
# ============================================================================

def epsilon(u: Expr) -> Expr:
    """Strain tensor: ε(u) = (∇u + ∇u^T) / 2

    Convenience function for linear elasticity and fluid mechanics.

    Parameters
    ----------
    u : Expr
        Displacement or velocity field

    Returns
    -------
    Expr
        Symmetric strain tensor

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='displacement')
    >>> eps = epsilon(u)
    """
    return sym(grad(u))


def sigma(u: Expr, p: Expr, mu: Union[Expr, float]) -> Expr:
    """Stokes stress tensor: σ = 2με(u) - pI

    Convenience function for Stokes/Navier-Stokes flows.

    Parameters
    ----------
    u : Expr
        Velocity field
    p : Expr
        Pressure field
    mu : Expr or float
        Dynamic viscosity

    Returns
    -------
    Expr
        Stress tensor

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='velocity')
    >>> p = TrialFunction(vec=1, name='pressure')
    >>> mu = Constant(name='viscosity')
    >>> stress = sigma(u, p, mu)
    """
    dim = u.vec if hasattr(u, 'vec') else 3
    return 2 * mu * epsilon(u) - p * Identity(dim)


def advect(u: Expr) -> Expr:
    """Advection operator: (u·∇)u

    Convenience function for convection term in Navier-Stokes.

    Parameters
    ----------
    u : Expr
        Velocity field

    Returns
    -------
    Expr
        Convection term

    Examples
    --------
    >>> u = TrialFunction(vec=3, name='velocity')
    >>> v = TestFunction(vec=3, name='v')
    >>> F = rho * inner(advect(u), v) * dx  # Navier-Stokes convection
    """
    return dot(u, grad(u))
