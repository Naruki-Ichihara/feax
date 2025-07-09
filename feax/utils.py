import jax
import jax.numpy as np
import numpy as onp
import meshio
import json
import os
import time
from functools import wraps
from typing import Optional, List, Tuple, Union

from feax import logger
from feax.generate_mesh import get_meshio_cell_type, Mesh


def save_sol(fe, sol, sol_file, cell_infos=None, point_infos=None):
    cell_type = get_meshio_cell_type(fe.ele_type)
    sol_dir = os.path.dirname(sol_file)
    os.makedirs(sol_dir, exist_ok=True)
    out_mesh = meshio.Mesh(points=fe.points, cells={cell_type: fe.cells})
    out_mesh.point_data['sol'] = onp.array(sol, dtype=onp.float32)
    if cell_infos is not None:
        for cell_info in cell_infos:
            name, data = cell_info
            # TODO: vector-valued cell data
            assert data.shape == (fe.num_cells,), f"cell data wrong shape, get {data.shape}, while num_cells = {fe.num_cells}"
            out_mesh.cell_data[name] = [onp.array(data, dtype=onp.float32)]
    if point_infos is not None:
        for point_info in point_infos:
            name, data = point_info
            assert len(data) == len(sol), "point data wrong shape!"
            out_mesh.point_data[name] = onp.array(data, dtype=onp.float32)
    out_mesh.write(sol_file)


def save_as_vtk(
    mesh: Mesh,
    sol_file: str,
    cell_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
    point_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
) -> None:
    """Save mesh and solution data to VTK format.

    Args:
        mesh: FEALAx mesh object containing nodes and elements
        sol_file: Output file path for VTK file
        cell_infos: List of (name, data) tuples for cell-based data.
            Data shape should be (n_elements, ...) where ... can be:
            - () or (1,) for scalar data
            - (n,) for vector data
            - (3, 3) for tensor data (will be flattened to (9,))
        point_infos: List of (name, data) tuples for point-based data.
            Data shape should be (n_nodes, ...)

    Raises:
        ValueError: If neither cell_infos nor point_infos is provided.
    """
    if cell_infos is None and point_infos is None:
        raise ValueError("At least one of cell_infos or point_infos must be provided.")
    
    # Get meshio cell type from mesh element type
    # We need to infer element type from the mesh structure
    n_nodes_per_element = mesh.cells.shape[1]
    if n_nodes_per_element == 4:
        element_type = 'TET4'
    elif n_nodes_per_element == 10:
        element_type = 'TET10'
    elif n_nodes_per_element == 8:
        element_type = 'HEX8'
    elif n_nodes_per_element == 20:
        element_type = 'HEX20'
    else:
        raise ValueError(f"Unsupported element type with {n_nodes_per_element} nodes per element")
    
    cell_type = get_meshio_cell_type(element_type)
    
    # Create output directory if needed
    sol_dir = os.path.dirname(sol_file)
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
    
    # Convert JAX arrays to numpy for meshio
    nodes_np = onp.array(mesh.points)
    elements_np = onp.array(mesh.cells)
    
    # Create meshio mesh
    out_mesh = meshio.Mesh(
        points=nodes_np,
        cells={cell_type: elements_np}
    )
    
    # Process cell data
    if cell_infos is not None:
        out_mesh.cell_data = {}
        for name, data in cell_infos:
            # Convert to numpy if it's a JAX array
            data = onp.array(data, dtype=onp.float32)
            
            # Validate shape
            if data.shape[0] != mesh.cells.shape[0]:
                raise ValueError(
                    f"Cell data '{name}' has wrong shape: got {data.shape}, "
                    f"expected first dimension = {mesh.cells.shape[0]}"
                )
            
            # Handle different data dimensions
            if data.ndim == 3:
                # Tensor (n_elements, 3, 3) -> flatten to (n_elements, 9)
                data = data.reshape(mesh.cells.shape[0], -1)
            elif data.ndim == 2:
                # Vector (n_elements, n) is OK as is
                pass
            else:
                # Scalar (n_elements,) -> (n_elements, 1)
                data = data.reshape(mesh.cells.shape[0], 1)
            
            out_mesh.cell_data[name] = [data]
    
    # Process point data
    if point_infos is not None:
        out_mesh.point_data = {}
        for name, data in point_infos:
            # Convert to numpy if it's a JAX array
            data = onp.array(data, dtype=onp.float32)
            
            # Validate shape
            if data.shape[0] != mesh.points.shape[0]:
                raise ValueError(
                    f"Point data '{name}' has wrong shape: got {data.shape}, "
                    f"expected first dimension = {mesh.points.shape[0]}"
                )
            
            out_mesh.point_data[name] = data
    
    # Write the mesh
    out_mesh.write(sol_file)


def modify_vtu_file(input_file_path, output_file_path):
    """Convert version 2.2 of vtu file to version 1.0
    meshio does not accept version 2.2, raising error of
    meshio._exceptions.ReadError: Unknown VTU file version '2.2'.
    """
    fin = open(input_file_path, "r")
    fout = open(output_file_path, "w")
    for line in fin:
        fout.write(line.replace('<VTKFile type="UnstructuredGrid" version="2.2">', '<VTKFile type="UnstructuredGrid" version="1.0">'))
    fin.close()
    fout.close()


def read_abaqus_and_write_vtk(abaqus_file, vtk_file):
    """Used for a quick inspection. Paraview can't open .inp file so we convert it to .vtu
    """
    meshio_mesh = meshio.read(abaqus_file)
    meshio_mesh.write(vtk_file)


def json_parse(json_filepath):
    with open(json_filepath) as f:
        args = json.load(f)
    json_formatted_str = json.dumps(args, indent=4)
    print(json_formatted_str)
    return args


def make_video(data_dir):
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is
    # enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following
    # "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite

    # TODO
    os.system(
        f'ffmpeg -y -framerate 10 -i {data_dir}/png/tmp/u.%04d.png -pix_fmt yuv420p -vf \
               "crop=trunc(iw/2)*2:trunc(ih/2)*2" {data_dir}/mp4/test.mp4') # noqa


# A simpler decorator for printing the timing results of a function
def timeit(func):

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# Wrapper for writing timing results to a file
def walltime(txt_dir=None, filename=None):

    def decorate(func):

        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            logger.info(
                f"Time elapsed {time_elapsed} of function {func.__name__} "
                f"on platform {platform}"
            )
            if txt_dir is not None:
                os.makedirs(txt_dir, exist_ok=True)
                fname = 'walltime'
                if filename is not None:
                    fname = filename
                with open(os.path.join(txt_dir, f"{fname}_{platform}.txt"),
                          'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values

        return wrapper

    return decorate
