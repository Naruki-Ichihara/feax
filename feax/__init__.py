"""feax - Finite Element Analysis with JAX"""
from .__version__ import __version__

__all__ = [
    'Problem', 'InternalVars', 
    'get_J', 'get_res', 'create_J_bc_function', 'create_res_bc_function',
    'Mesh', 'DirichletBC', 'apply_boundary_to_J', 'apply_boundary_to_res', 'newton_solve', 'SolverOptions', 'create_solver',
    'zero_like_initial_guess', 'DirichletBCSpec', 'DirichletBCConfig', 'dirichlet_bc_config',
    'linear_solve', 'newton_solve_fori', 'newton_solve_py'
]

def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        module_map = {
            'Problem': 'feax.problem',
            'InternalVars': 'feax.internal_vars',
            'get_J': 'feax.assembler',
            'get_res': 'feax.assembler',
            'create_J_bc_function': 'feax.assembler',
            'create_res_bc_function': 'feax.assembler',
            'Mesh': 'feax.mesh',
            'DirichletBC': 'feax.DCboundary',
            'apply_boundary_to_J': 'feax.DCboundary',
            'apply_boundary_to_res': 'feax.DCboundary',
            'newton_solve': 'feax.solver',
            'SolverOptions': 'feax.solver',
            'create_solver': 'feax.solver',
            'linear_solve': 'feax.solver',
            'newton_solve_fori': 'feax.solver',
            'newton_solve_py': 'feax.solver',
            'zero_like_initial_guess': 'feax.utils',
            'DirichletBCSpec': 'feax.bc_spec',
            'DirichletBCConfig': 'feax.bc_spec',
            'dirichlet_bc_config': 'feax.bc_spec',
        }
        mod = import_module(module_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'feax' has no attribute '{name}'")