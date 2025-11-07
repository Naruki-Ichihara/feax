---
sidebar_label: symbolic
title: feax.experimental.symbolic
---

Symbolic DSL for defining finite element weak forms.

This module provides a lightweight symbolic interface for defining physics
in FEAX, similar to UFL in FEniCS but built purely on JAX. The symbolic
expressions are compiled to efficient JAX kernels via the universal kernel
mechanism.

Example
-------
```python
>>> from feax.symbolic import TrialFunction, TestFunction, grad, div, inner, dx
>>>
>>> # Stokes flow
>>> u = TrialFunction(vec=3, name=&#x27;velocity&#x27;)
>>> p = TrialFunction(vec=1, name=&#x27;pressure&#x27;)
>>> v = TestFunction(vec=3, name=&#x27;v&#x27;)
>>> q = TestFunction(vec=1, name=&#x27;q&#x27;)
>>>
>>> mu = Constant(name=&#x27;viscosity&#x27;)
>>>
>>> # Weak form
>>> F = (
>>>     inner(grad(u), grad(v)) * dx +
>>>     div(u) * q * dx +
>>>     p * div(v) * dx
>>> )
```

## TensorRank Objects

```python
class TensorRank(Enum)
```

Tensor rank enumeration for type checking.

## Expr Objects

```python
@dataclass
class Expr()
```

Base class for all symbolic expressions.

Attributes
----------
- **rank** (*TensorRank*): Tensor rank of the expression (scalar=0, vector=1, tensor=2)
- **shape** (*tuple*): Shape of the expression (e.g., (3,) for vector, (3,3) for tensor)


## TrialFunction Objects

```python
@dataclass
class TrialFunction(Expr)
```

Trial function (unknown) in the weak form.

Parameters
----------
- **vec** (*int*): Vector dimension (1 for scalar, 3 for 3D vector)
- **name** (*str*): Name of the function (for code generation)
- **index** (*int, optional*): Index in multi-variable problems (default: 0)


Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;velocity&#x27;)  # Vector field
>>> p = TrialFunction(vec=1, name=&#x27;pressure&#x27;)  # Scalar field
```

## TestFunction Objects

```python
@dataclass
class TestFunction(Expr)
```

Test function in the weak form.

Parameters
----------
- **vec** (*int*): Vector dimension (1 for scalar, 3 for 3D vector)
- **name** (*str*): Name of the function (for code generation)
- **index** (*int, optional*): Index in multi-variable problems (default: 0)


Examples
--------
```python
>>> v = TestFunction(vec=3, name=&#x27;v&#x27;)  # Vector test function
>>> q = TestFunction(vec=1, name=&#x27;q&#x27;)  # Scalar test function
```

## Constant Objects

```python
@dataclass
class Constant(Expr)
```

Constant value or internal variable.

Parameters
----------
- **name** (*str*): Name of the constant (for internal variable lookup)
- **vec** (*int, optional*): Vector dimension (default: 1 for scalar)
- **value** (*float or array, optional*): Constant value (if not using internal variables)


Examples
--------
```python
>>> mu = Constant(name=&#x27;viscosity&#x27;)  # Scalar from internal vars
>>> f = Constant(name=&#x27;force&#x27;, vec=3)  # Vector from internal vars
>>> rho = Constant(name=&#x27;density&#x27;, value=1000.0)  # Fixed value
```

## ScalarConstant Objects

```python
@dataclass
class ScalarConstant(Expr)
```

Literal scalar constant (e.g., 2.0, -1.0).

Used internally for arithmetic with Python numbers.

## Grad Objects

```python
@dataclass
class Grad(Expr)
```

Gradient operator: ∇u

Parameters
----------
- **operand** (*Expr*): Expression to take gradient of


Notes
-----
- Scalar → Vector: grad(p) has shape (dim,)
- Vector → Tensor: grad(u) has shape (vec, dim)

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> grad_u = grad(u)  # Shape (3, 3) - velocity gradient tensor
```

## Div Objects

```python
@dataclass
class Div(Expr)
```

Divergence operator: ∇·u

Parameters
----------
- **operand** (*Expr*): Vector expression to take divergence of


Notes
-----
- Vector → Scalar: div(u) is scalar
- Tensor → Vector: div(T) has shape (vec,)

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> div_u = div(u)  # Scalar - incompressibility constraint
```

## Sym Objects

```python
@dataclass
class Sym(Expr)
```

Symmetric part of tensor: (T + T^T) / 2

Parameters
----------
- **operand** (*Expr*): Tensor expression to symmetrize


Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> epsilon = sym(grad(u))  # Strain tensor
```

## Transpose Objects

```python
@dataclass
class Transpose(Expr)
```

Transpose operator: T^T

Parameters
----------
- **operand** (*Expr*): Tensor expression to transpose


## Trace Objects

```python
@dataclass
class Trace(Expr)
```

Trace of tensor: tr(T)

Parameters
----------
- **operand** (*Expr*): Tensor expression to take trace of


Examples
--------
```python
>>> sigma = ...  # Stress tensor
>>> pressure = tr(sigma) / 3  # Mean stress
```

## Inner Objects

```python
@dataclass
class Inner(Expr)
```

Inner product: a·b or A:B

Parameters
----------
- **left** (*Expr*): Left operand
- **right** (*Expr*): Right operand


Notes
-----
- Vector·Vector → Scalar: u·v
- Tensor:Tensor → Scalar: σ:ε (double contraction)

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> v = TestFunction(vec=3, name=&#x27;v&#x27;)
>>> inner(u, v)  # Scalar product
>>>
>>> sigma = ...
>>> epsilon = ...
>>> inner(sigma, epsilon)  # Stress-strain work
```

## Dot Objects

```python
@dataclass
class Dot(Expr)
```

Dot product: a·b or a·T (vector with rows of tensor)

Parameters
----------
- **left** (*Expr*): Vector operand
- **right** (*Expr*): Vector or tensor operand


Notes
-----
- Vector·Vector → Scalar: u·v
- Vector·Tensor → Vector: u·∇u (convection term)

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> grad_u = grad(u)
>>> dot(u, grad_u)  # (u·∇)u for Navier-Stokes convection
```

## Outer Objects

```python
@dataclass
class Outer(Expr)
```

Outer product: a ⊗ b

Parameters
----------
- **left** (*Expr*): Left vector
- **right** (*Expr*): Right vector


Notes
-----
Vector ⊗ Vector → Tensor

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> v = TestFunction(vec=3, name=&#x27;v&#x27;)
>>> outer(u, v)  # Tensor (3, 3)
```

## Add Objects

```python
@dataclass
class Add(Expr)
```

Addition: a + b

## Sub Objects

```python
@dataclass
class Sub(Expr)
```

Subtraction: a - b

## Mul Objects

```python
@dataclass
class Mul(Expr)
```

Multiplication: a * b (scalar multiplication or element-wise)

## Division Objects

```python
@dataclass
class Division(Expr)
```

Division: a / b (scalar division)

## Identity Objects

```python
@dataclass
class Identity(Expr)
```

Identity tensor: I

Parameters
----------
- **dim** (*int*): Dimension (e.g., 3 for 3D)


Examples
--------
```python
>>> I = Identity(3)
>>> sigma = 2*mu*epsilon - p*I  # Stress tensor
```

## Integral Objects

```python
@dataclass
class Integral(Expr)
```

Integral expression: ∫ expr dΩ or ∫ expr dΓ

Parameters
----------
- **integrand** (*Expr*): Expression to integrate
- **measure** (*str*): Integration measure (&#x27;dx&#x27; for volume, &#x27;ds&#x27; for surface)
- **boundary_id** (*int, optional*): Boundary identifier for surface integrals


Notes
-----
This is the top-level expression returned when using dx or ds.

#### measure

&#x27;dx&#x27; or &#x27;ds&#x27;

#### \_\_add\_\_

```python
def __add__(other)
```

Allow adding integrals: ∫ a dx + ∫ b dx

## IntegralSum Objects

```python
@dataclass
class IntegralSum(Expr)
```

Sum of multiple integrals.

Used to represent forms like: F = ∫ a dx + ∫ b dx + ∫ c ds

#### grad

```python
def grad(expr: Expr) -> Grad
```

Gradient operator: ∇expr

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> grad(u)  # Velocity gradient tensor
```

#### div

```python
def div(expr: Expr) -> Div
```

Divergence operator: ∇·expr

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> div(u)  # Scalar divergence (incompressibility)
```

#### sym

```python
def sym(expr: Expr) -> Sym
```

Symmetric part: (T + T^T) / 2

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> epsilon = sym(grad(u))  # Strain tensor
```

#### transpose

```python
def transpose(expr: Expr) -> Transpose
```

Transpose: T^T

Examples
--------
```python
>>> T = grad(u)
>>> transpose(T)
```

#### tr

```python
def tr(expr: Expr) -> Trace
```

Trace: tr(T)

Examples
--------
```python
>>> epsilon = sym(grad(u))
>>> vol_strain = tr(epsilon)  # Volumetric strain
```

#### inner

```python
def inner(left: Expr, right: Expr) -> Inner
```

Inner product: a·b or A:B

Examples
--------
```python
>>> inner(u, v)  # Vector dot product
>>> inner(sigma, epsilon)  # Tensor double contraction
```

#### dot

```python
def dot(left: Expr, right: Expr) -> Dot
```

Dot product: a·b or a·T

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;u&#x27;)
>>> dot(u, grad(u))  # Convection term (u·∇)u
```

#### outer

```python
def outer(left: Expr, right: Expr) -> Outer
```

Outer product: a ⊗ b

Examples
--------
```python
>>> outer(u, v)  # Tensor from two vectors
```

#### epsilon

```python
def epsilon(u: Expr) -> Expr
```

Strain tensor: ε(u) = (∇u + ∇u^T) / 2

Convenience function for linear elasticity and fluid mechanics.

Parameters
----------
- **u** (*Expr*): Displacement or velocity field


Returns
-------
Expr
    Symmetric strain tensor

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;displacement&#x27;)
>>> eps = epsilon(u)
```

#### sigma

```python
def sigma(u: Expr, p: Expr, mu: Union[Expr, float]) -> Expr
```

Stokes stress tensor: σ = 2με(u) - pI

Convenience function for Stokes/Navier-Stokes flows.

Parameters
----------
- **u** (*Expr*): Velocity field
- **p** (*Expr*): Pressure field
- **mu** (*Expr or float*): Dynamic viscosity


Returns
-------
Expr
    Stress tensor

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;velocity&#x27;)
>>> p = TrialFunction(vec=1, name=&#x27;pressure&#x27;)
>>> mu = Constant(name=&#x27;viscosity&#x27;)
>>> stress = sigma(u, p, mu)
```

#### advect

```python
def advect(u: Expr) -> Expr
```

Advection operator: (u·∇)u

Convenience function for convection term in Navier-Stokes.

Parameters
----------
- **u** (*Expr*): Velocity field


Returns
-------
Expr
    Convection term

Examples
--------
```python
>>> u = TrialFunction(vec=3, name=&#x27;velocity&#x27;)
>>> v = TestFunction(vec=3, name=&#x27;v&#x27;)
>>> F = rho * inner(advect(u), v) * dx  # Navier-Stokes convection
```

