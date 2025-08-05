"""
Problem class with modular design separating FE structure from material parameters.
"""

import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass
import functools

from feax.mesh import Mesh
from feax.fe import FiniteElement


@dataclass
class Problem:
    """Problem class that handles FE structure with separated material parameters.
    
    This design separates finite element structure from material properties,
    making it suitable for optimization and parameter studies.
    
    Attributes
    ----------
    mesh : Mesh
        Finite element mesh
    vec : int
        Number of vector components in solution
    dim : int
        Spatial dimension of the problem
    ele_type : str
        Element type (HEX8, TET4, etc.)
    gauss_order : int
        Order of Gaussian quadrature
    location_fns : list
        Location functions for surface integrals
    additional_info : tuple
        Additional problem-specific information
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = 'HEX8'
    gauss_order: int = None
    location_fns: list = None
    additional_info: tuple = ()

    def __post_init__(self):
        """Initialize all state data for the finite element problem."""
        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.gauss_order = [self.gauss_order]

        self.num_vars = len(self.mesh)

        self.fes = [FiniteElement(mesh=self.mesh[i], 
                                  vec=self.vec[i], 
                                  dim=self.dim, 
                                  ele_type=self.ele_type[i], 
                                  gauss_order=self.gauss_order[i] if type(self.gauss_order) == type([]) else self.gauss_order) \
                    for i in range(self.num_vars)] 

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0] 
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        def find_ind(*x):
            inds = []
            for i in range(len(x)):
                x[i].reshape(-1)
                crt_ind = self.fes[i].vec * x[i][:, None] + np.arange(self.fes[i].vec)[None, :] + self.offset[i]
                inds.append(crt_ind.reshape(-1))

            return np.hstack(inds)

        # (num_cells, num_nodes*vec + ...)
        inds = np.array(jax.vmap(find_ind)(*self.cells_list))
        self.I = np.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
        self.J = np.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list] # [(num_selected_faces, num_nodes), ...]
            inds_face = np.array(jax.vmap(find_ind)(*cells_list_face)) # (num_selected_faces, num_nodes*vec + ...)
            I_face = np.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
            J_face = np.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
            self.I = np.hstack((self.I, I_face))
            self.J = np.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)
     
        self.cells_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*self.cells_list) # (num_cells, num_nodes + ...)

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)
        
        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(dumb_sol_list)
        self.num_total_dofs_all_vars = len(dumb_dofs)

        self.num_nodes_cumsum = np.cumsum(np.array([0] + [fe.num_nodes for fe in self.fes]))
        # (num_cells, num_vars, num_quads)
        self.JxW = np.transpose(np.stack([fe.JxW for fe in self.fes]), axes=(1, 0, 2)) 
        # (num_cells, num_quads, num_nodes +..., dim)
        self.shape_grads = np.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = np.concatenate([fe.v_grads_JxW for fe in self.fes], axis=2)

        # TODO: assert all vars quad points be the same
        # (num_cells, num_quads, dim)
        self.physical_quad_points = self.fes[0].get_physical_quad_points()  

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for fe in self.fes:
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds)  
                selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
            s_shape_grads = np.concatenate(s_shape_grads, axis=2)
            # (num_selected_faces, num_vars, num_face_quads)
            n_scale = np.transpose(np.stack(n_scale), axes=(1, 0, 2))  
            # (num_selected_faces, num_face_quads, num_nodes + ...)
            s_shape_vals = np.concatenate(s_shape_vals, axis=2)
            # (num_selected_faces, num_face_quads, dim)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(boundary_inds) 

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            # TODO: assert all vars face quad points be the same
            self.physical_surface_quad_points.append(physical_surface_quad_points)

        # Initialize without internal_vars - kernels will be created separately
        self.custom_init(*self.additional_info)

    def custom_init(self, *args):
        """Child class should override if additional initialization is required."""
        pass

    def get_tensor_map(self):
        """Override in subclass to define volume physics."""
        raise NotImplementedError("Subclass must implement get_tensor_map")
    
    def get_surface_maps(self):
        """Override in subclass to define surface physics."""
        return []
    
    def get_mass_map(self):
        """Override in subclass to define mass matrix physics."""
        return None


# Register as JAX PyTree (straightforward without internal_vars)
def _problem_tree_flatten(obj):
    """Flatten the Problem - no dynamic parts since no internal_vars."""
    # No dynamic parts - everything is static
    dynamic = ()
    
    # All data is static
    static = {
        'mesh': obj.mesh,
        'vec': obj.vec,
        'dim': obj.dim,
        'ele_type': obj.ele_type,
        'gauss_order': obj.gauss_order,
        'dirichlet_bc_info': obj.dirichlet_bc_info,
        'location_fns': obj.location_fns,
        'additional_info': obj.additional_info,
    }
    return dynamic, static


def _problem_tree_unflatten(static, dynamic):
    """Reconstruct the Problem from flattened parts."""
    # Create a new instance with the original constructor parameters
    instance = Problem(
        mesh=static['mesh'],
        vec=static['vec'],
        dim=static['dim'],
        ele_type=static['ele_type'],
        gauss_order=static['gauss_order'],
        dirichlet_bc_info=static['dirichlet_bc_info'],
        location_fns=static['location_fns'],
        additional_info=static['additional_info'],
    )
    
    return instance


jax.tree_util.register_pytree_node(
    Problem,
    _problem_tree_flatten,
    _problem_tree_unflatten
)