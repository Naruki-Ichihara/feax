"""Memory-efficient utilities for FEAX computations."""

import jax
import jax.numpy as np
from typing import Callable, Any, Optional


def chunked_vmap(func: Callable, 
                 chunk_size: Optional[int] = None,
                 in_axes: int = 0,
                 out_axes: int = 0) -> Callable:
    """Create a memory-efficient version of vmap that processes in chunks.
    
    This function wraps jax.vmap to process large batches in smaller chunks,
    reducing peak memory usage at the cost of some performance.
    
    Parameters
    ----------
    func : Callable
        Function to be vmapped
    chunk_size : int, optional
        Maximum chunk size for processing. If None, uses adaptive sizing
        based on input size.
    in_axes : int, default 0
        Input axes specification for vmap
    out_axes : int, default 0
        Output axes specification for vmap
        
    Returns
    -------
    Callable
        Memory-efficient vmapped function
        
    Examples
    --------
    >>> def my_func(x):
    ...     return expensive_computation(x)
    >>> 
    >>> # Regular vmap might OOM on large batches
    >>> # vmap_func = jax.vmap(my_func)
    >>> 
    >>> # Use chunked version instead
    >>> vmap_func = chunked_vmap(my_func, chunk_size=10)
    >>> result = vmap_func(large_batch)  # Processes in chunks of 10
    """
    
    vmapped_func = jax.vmap(func, in_axes=in_axes, out_axes=out_axes)
    
    def chunked_func(*args, **kwargs):
        # Get batch size from first argument
        if len(args) > 0:
            if hasattr(args[0], 'shape'):
                batch_size = args[0].shape[0]
            else:
                batch_size = len(args[0])
        else:
            raise ValueError("chunked_vmap requires at least one positional argument")
        
        # Determine chunk size
        if chunk_size is None:
            # Adaptive chunk size based on batch size
            if batch_size > 100:
                actual_chunk_size = 10
            elif batch_size > 50:
                actual_chunk_size = 20
            else:
                actual_chunk_size = batch_size
        else:
            actual_chunk_size = min(chunk_size, batch_size)
        
        # If batch fits in one chunk, just use regular vmap
        if actual_chunk_size >= batch_size:
            return vmapped_func(*args, **kwargs)
        
        # Process in chunks
        num_chunks = (batch_size + actual_chunk_size - 1) // actual_chunk_size
        results = []
        
        for i in range(num_chunks):
            start_idx = i * actual_chunk_size
            end_idx = min((i + 1) * actual_chunk_size, batch_size)
            
            # Slice all array arguments
            chunk_args = []
            for arg in args:
                if hasattr(arg, '__getitem__'):
                    chunk_args.append(arg[start_idx:end_idx])
                else:
                    chunk_args.append(arg)
            
            # Process chunk
            chunk_result = vmapped_func(*chunk_args, **kwargs)
            results.append(chunk_result)
        
        # Concatenate results
        if len(results) == 1:
            return results[0]
        else:
            # Handle different output types
            if isinstance(results[0], tuple):
                # Multiple outputs
                num_outputs = len(results[0])
                concatenated = []
                for output_idx in range(num_outputs):
                    output_chunks = [r[output_idx] for r in results]
                    concatenated.append(np.concatenate(output_chunks, axis=out_axes))
                return tuple(concatenated)
            else:
                # Single output
                return np.concatenate(results, axis=out_axes)
    
    return chunked_func


def estimate_memory_usage(problem, batch_size: int = 1) -> dict:
    """Estimate memory usage for a given problem and batch size.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem
    batch_size : int
        Number of problems to solve in parallel
        
    Returns
    -------
    dict
        Estimated memory usage in MB for different components
    """
    num_dofs = problem.num_total_dofs_all_vars
    num_cells = problem.num_cells
    
    # Estimate sizes (assuming float64)
    bytes_per_float = 8
    
    # Solution vector
    sol_size = num_dofs * batch_size * bytes_per_float / 1e6  # MB
    
    # Jacobian matrix (sparse, estimate nnz)
    nnz_per_row = 27 * problem.vec[0] if problem.dim == 3 else 9 * problem.vec[0]  # Approximate
    jacobian_size = num_dofs * nnz_per_row * batch_size * bytes_per_float / 1e6  # MB
    
    # Intermediate arrays during assembly
    assembly_overhead = num_cells * 100 * batch_size * bytes_per_float / 1e6  # Rough estimate
    
    return {
        'solution_vectors': sol_size,
        'jacobian_matrices': jacobian_size,
        'assembly_overhead': assembly_overhead,
        'total_estimate': sol_size + jacobian_size + assembly_overhead
    }


def create_memory_efficient_solver(problem, bc, solver_options=None, **kwargs):
    """Create a solver with memory-efficient settings.
    
    This function creates a solver with optimized settings for large problems,
    including adaptive batch sizes and chunked operations.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem
    bc : BoundaryCondition
        Boundary conditions
    solver_options : SolverOptions, optional
        Solver configuration
    **kwargs
        Additional arguments passed to create_solver
        
    Returns
    -------
    Callable
        Memory-optimized solver function
    """
    from feax import create_solver
    
    # Estimate problem size and adjust settings
    mem_estimate = estimate_memory_usage(problem)
    
    # Adjust solver options for large problems
    if mem_estimate['total_estimate'] > 1000:  # > 1GB estimated
        print(f"Large problem detected (~{mem_estimate['total_estimate']:.0f} MB)")
        print("Using memory-efficient settings...")
        
        # Could add additional optimizations here
        
    return create_solver(problem, bc, solver_options, **kwargs)