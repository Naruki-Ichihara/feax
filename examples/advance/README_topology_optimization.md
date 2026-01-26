# Topology Optimization Examples

This directory contains two implementations of 3D topology optimization for a cantilever beam problem.

## Files

### 1. `topology_optimization.py` - NLopt MMA Implementation
**Optimizer**: NLopt's Method of Moving Asymptotes (MMA)

**Features**:
- Uses external NLopt library for optimization
- MMA algorithm specifically designed for topology optimization
- Direct constraint handling via NLopt's constraint API
- Mature, well-tested algorithm

**Pros**:
- Very efficient for topology optimization problems
- Proven convergence properties
- Built-in constraint handling

**Cons**:
- Requires external NLopt library installation
- Less flexible for custom optimization strategies
- Not pure JAX (harder to extend/customize)

### 2. `topology_optimization_mdmm.py` - Optax + MDMM Implementation
**Optimizer**: Optax (JAX) with MDMM constraint handling

**Features**:
- Pure JAX/Optax implementation
- Modified Differential Multiplier Method (MDMM) for constraints
- Fully differentiable with automatic differentiation
- Flexible optimizer choice (Adam used by default)

**Pros**:
- Pure Python/JAX (no external optimizer dependencies)
- Fully differentiable and composable
- Easy to customize and extend
- Integrates seamlessly with FEAX Gene toolkit
- Visualizes Lagrange multiplier evolution

**Cons**:
- May require more iterations than MMA
- Needs careful tuning of damping and weight parameters
- Less established for topology optimization (newer approach)

## Problem Setup

Both implementations solve the same problem:
- **Geometry**: 3D cantilever beam (100 × 4 × 20)
- **Objective**: Minimize compliance (maximize stiffness)
- **Constraint**: Volume fraction ≤ 40%
- **Material**: SIMP interpolation with penalty = 3
- **Filter**: Density filter with radius = 3

## Usage

### NLopt MMA version:
```bash
# Requires: pip install nlopt
python topology_optimization.py
```

### Optax + MDMM version:
```bash
# Only requires JAX and optax (already in FEAX dependencies)
python topology_optimization_mdmm.py
```

## Key Differences in Implementation

| Aspect | NLopt MMA | Optax + MDMM |
|--------|-----------|--------------|
| Optimizer | `nlopt.LD_MMA` | `optax.adam()` + `mdmm.optax_prepare_update()` |
| Constraints | `add_inequality_constraint()` | `mdmm.ineq()` |
| Parameters | Flat array | Dictionary with Lagrange multipliers |
| Gradient | Callback function | JAX automatic differentiation |
| Box constraints | `set_lower/upper_bounds()` | Manual projection after update |
| Output | Compliance & Volume | + Constraint violation + λ evolution |

## MDMM-Specific Parameters

The MDMM implementation has these tunable parameters:

```python
# Constraint definition
constraint = mdmm.ineq(
    volume_constraint_fn,
    damping=10.0,   # Controls oscillation reduction (higher = more damping)
    weight=100.0,   # Weight relative to objective (higher = stricter constraint)
)

# Optimizer settings
learning_rate = 0.05  # Step size for gradient descent/ascent
```

**Tuning guidelines**:
- Increase `damping` if constraint oscillates too much
- Increase `weight` if constraint is violated consistently
- Decrease `learning_rate` if optimization is unstable
- Increase `learning_rate` for faster convergence (if stable)

## Output

Both implementations generate:
- VTU files with density field for visualization
- Optimization history plots (MDMM includes additional constraint metrics)
- CSV files with iteration data

**NLopt output directory**: `output/`
**MDMM output directory**: `output_mdmm/`

## Visualization

Use ParaView or similar tools to visualize the `.vtu` files:
```bash
paraview output/topology_opt_final.vtu
paraview output_mdmm/topology_opt_final.vtu
```

## References

- **MMA**: Svanberg, K. (1987). "The method of moving asymptotes—a new method for structural optimization"
- **MDMM**: Platt, J.C. & Barr, A.H. (1988). "Constrained differential optimization for neural networks"
- **SIMP**: Bendsøe, M.P. (1989). "Optimal shape design as a material distribution problem"
