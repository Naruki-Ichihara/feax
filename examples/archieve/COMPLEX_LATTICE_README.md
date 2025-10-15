# Complex Lattice Homogenization Example

## Overview

The `periodic_homogenization_complex.py` example demonstrates advanced lattice structures with complex node connectivity patterns for computational homogenization. This builds upon the basic periodic homogenization example with more sophisticated lattice topologies.

## Features

### Three Advanced Lattice Structures

#### 1. BCC with Face Centers (27 nodes, 62 edges)
- **Nodes:**
  - 8 corner nodes (cube vertices)
  - 1 body center node
  - 6 face center nodes
  - 12 edge center nodes
- **Connectivity:**
  - BCC pattern: body center connected to all corners
  - Face centers connected to their 4 corner nodes
  - Edge centers connected to their 2 endpoints
  - Body center connected to all face centers
- **Properties:**
  - Effective Young's modulus: ~8700 MPa
  - Relative density: ~12.4%
  - Highly connected structure for good stiffness

#### 2. Octet Truss (14 nodes, 36 edges)
- **Nodes:**
  - 8 corner nodes
  - 6 face center nodes
- **Connectivity:**
  - Each face center connected to its 4 corners
  - Face centers connected to adjacent face centers (12 connections)
  - Creates tetrahedral-octahedral pattern
- **Properties:**
  - Effective Young's modulus: ~2556 MPa
  - Relative density: ~3.7%
  - Efficient strength-to-weight ratio
  - Commonly used in aerospace applications

#### 3. Kelvin Foam (24 nodes, 61 edges)
- **Nodes:**
  - 8 corner nodes
  - 6 face center nodes
  - 8 truncation nodes (offset from corners)
  - 2 internal nodes
- **Connectivity:**
  - Corners to truncation nodes
  - Truncation nodes to nearby face centers
  - Truncation nodes form hexagonal patterns
  - Internal nodes connected to truncation nodes
  - Creates tetrakaidecahedron (space-filling foam)
- **Properties:**
  - Effective Young's modulus: ~1406 MPa
  - Relative density: ~2.0%
  - Space-filling foam structure
  - Good for energy absorption

## Implementation Details

### Graph-Based Lattice Definition

All lattices are defined using:
1. **Node coordinates** - Spatial positions in unit cell [0,1]³
2. **Adjacency matrix** - Symmetric matrix defining connectivity

Example structure:
```python
def create_custom_lattice():
    # Define node positions
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        # ... more nodes
    ], dtype=np.float32)

    # Build adjacency matrix
    n = nodes.shape[0]
    adj_mat = np.zeros((n, n), dtype=np.int32)

    # Define connections
    adj_mat = adj_mat.at[i, j].set(1)
    adj_mat = adj_mat.at[j, i].set(1)  # Symmetric

    return nodes, adj_mat
```

### Key Functions Used

- `create_lattice_function_from_adjmat()` - Creates lattice evaluation function from adjacency matrix
- `create_lattice_density_field()` - Generates element-based density field for FEA
- `create_homogenization_solver()` - Computes effective stiffness tensor

### JAX Compatibility

All operations are fully JAX-compatible:
- `@jax.jit` decorator for performance
- Nested `vmap` for parallel edge checking
- No dynamic boolean indexing

## Results Interpretation

### Homogenized Stiffness Matrix (Voigt Notation)

The 6×6 matrix represents effective elastic properties:
```
[C11 C12 C13   0   0   0 ]
[C12 C22 C23   0   0   0 ]
[C13 C23 C33   0   0   0 ]
[ 0   0   0  C44  0   0 ]
[ 0   0   0   0  C55  0 ]
[ 0   0   0   0   0  C66]
```

For isotropic materials (like BCC and Octet):
- C11 = C22 = C33 (same stiffness in all directions)
- C44 = C55 = C66 (same shear modulus)
- Off-diagonal terms show coupling

### Effective Properties

Computed from stiffness matrix:
- **Young's modulus** (E): Resistance to uniaxial tension
- **Shear modulus** (G): Resistance to shear deformation
- **Bulk modulus** (K): Resistance to volumetric compression
- **Relative density**: Effective stiffness relative to bulk material

## Usage

```bash
# Run the example
python examples/periodic_homogenization_complex.py

# Computation takes ~17-20 seconds total
# Output shows detailed results for all three lattices
```

## Customization

### Adding New Lattice Structures

1. Define node positions in unit cell [0,1]³
2. Build symmetric adjacency matrix
3. Add to `lattice_configs` list:

```python
def create_my_lattice():
    nodes = np.array([...], dtype=np.float32)
    adj_mat = np.zeros((n, n), dtype=np.int32)
    # Build connectivity...
    return nodes, adj_mat

lattice_configs.append(
    ("My Lattice", create_my_lattice())
)
```

### Adjusting Parameters

- `radius = 0.08` - Strut thickness (affects density)
- `mesh_size=0.1` - FE mesh resolution
- `density_void=1e-5` - Minimum density for numerical stability

## Performance Notes

- First call slower due to JIT compilation
- BCC: ~4.5s (most connections, most computation)
- Octet: ~6.1s (moderate complexity)
- Kelvin: ~6.7s (complex internal structure)
- Memory usage scales with number of nodes and edges

## References

- **Octet Truss**: Deshpande et al., Journal of the Mechanics and Physics of Solids (2001)
- **Kelvin Foam**: Weaire & Phelan, Philosophical Magazine Letters (1994)
- **BCC Lattices**: Ashby, Philosophical Transactions of the Royal Society A (2006)

## Comparison with Basic Example

| Feature | Basic (`periodic_homogenization.py`) | Complex |
|---------|-------------------------------------|---------|
| Nodes | 9 (simple star) | 14-27 (multiple topologies) |
| Edges | 8 | 36-62 |
| Lattice types | 1 | 3 |
| Complexity | Simple connectivity | Advanced patterns |
| Applications | Learning/testing | Research/design |
