# Basic Tutorials

This section contains introductory tutorials for using FEAX. Each tutorial covers fundamental concepts and provides complete working examples.

## Available Tutorials

### [Linear Elasticity](./linear_elasticity.md)

Solve a cantilever beam problem with surface traction loading. This tutorial covers:

- Mesh generation with `box_mesh`
- Defining constitutive laws via `get_tensor_map`
- Applying Dirichlet boundary conditions
- Surface traction implementation
- Linear solver configuration (BiCGSTAB, CG, GMRES)
- Internal variables for parameter management
- Solution extraction and VTK output

**Key concepts:** Problem definition, boundary conditions, iterative solvers, internal variables
