# Basic Tutorials

This section contains introductory tutorials for using FEAX. Each tutorial covers fundamental concepts and provides complete working examples.

## Available Tutorials

### [Linear Elasticity](./linear_elasticity.md)

Solve a cantilever beam problem with surface traction loading. This tutorial covers:

- Mesh generation with `box_mesh`
- Defining constitutive laws via `get_tensor_map`
- Applying Dirichlet boundary conditions
- Surface traction implementation
- Linear solver configuration (BiCGSTAB, CG, GMRES, SPSOLVE)
- Internal variables for parameter management
- Solution extraction and VTK output

**Key concepts:** Problem definition, boundary conditions, iterative solvers, internal variables

### [Just-in-Time Compilation Transform](./jit_transform.md)

Accelerate solver performance using JAX's JIT compilation. This tutorial demonstrates:

- Creating JIT-compiled solvers with `jax.jit`
- Benchmarking JIT vs non-JIT performance
- Understanding compilation overhead
- Achieving 10-20× speedup for repeated solver calls
- Practical parameter studies with JIT
- Combining JIT with `jax.vmap` for vectorized operations

**Key concepts:** JIT compilation, performance optimization, parameter studies, GPU acceleration

### [Vectorization Transform](./vmap_transform.md)

Parallelize parameter studies using JAX's automatic vectorization. This tutorial demonstrates:

- Using `jax.vmap` to solve multiple load cases in parallel
- Benchmarking for-loop vs vmap performance
- Achieving 3-44× CPU speedup with superlinear scaling (50-100× on GPU)
- Understanding memory optimization and batch size effects
- Combining JIT and vmap for multiplicative performance gains
- Gradient computation through vectorized solvers
- Multi-dimensional parameter space exploration

**Key concepts:** Vectorization, batch processing, parameter studies, automatic parallelization, superlinear scaling

### [Nonlinear Hyperelasticity](./hyperelasticity.md)

Solve large-deformation problems using Newton's method and automatic differentiation. This tutorial demonstrates:

- Energy-based constitutive modeling (Neo-Hookean material)
- Automatic stress computation via `jax.grad`
- Newton's method for nonlinear systems
- Large deformation kinematics (deformation gradient)
- Load stepping for convergence
- Implementing alternative hyperelastic models

**Key concepts:** Nonlinear FEA, hyperelasticity, automatic differentiation, Newton solver, large deformations
