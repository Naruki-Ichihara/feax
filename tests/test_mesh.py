"""
Tests for mesh generation functionality.

This module tests mesh generation including:
- Box mesh creation
- Mesh properties validation
- Node and element counting
- Mesh size parameters
"""

import pytest
import jax.numpy as jnp
import feax as fe


# ============================================================================
# Slim Tests - Mesh Generation
# ============================================================================

@pytest.mark.cpu
def test_box_mesh_creation():
    """Test basic box mesh creation."""
    L, W, H = 10, 10, 10
    mesh_size = 5

    mesh = fe.mesh.box_mesh((L, W, H), mesh_size=mesh_size)

    # Check that mesh object is created
    assert mesh is not None

    # Check that mesh has required attributes
    assert hasattr(mesh, 'points')
    assert hasattr(mesh, 'cells')


@pytest.mark.cpu
def test_box_mesh_dimensions(simple_mesh):
    """Test that box mesh has correct spatial dimensions."""
    mesh = simple_mesh

    # Get mesh points
    points = mesh.points

    # Check that points are 3D
    assert points.shape[1] == 3

    # Check bounding box dimensions
    min_coords = jnp.min(points, axis=0)
    max_coords = jnp.max(points, axis=0)

    # Should span from 0 to 10 in each dimension
    assert jnp.allclose(min_coords, jnp.array([0., 0., 0.]), atol=1e-10)
    assert jnp.allclose(max_coords, jnp.array([10., 10., 10.]), atol=1e-10)


@pytest.mark.cpu
def test_box_mesh_node_count(simple_mesh):
    """Test exact node count for box mesh."""
    mesh = simple_mesh

    # For mesh_size=5 on a 10x10x10 box, we expect specific node count
    # mesh_size=5 means 2 elements per dimension, so 3 nodes per dimension
    # Total nodes = 3 * 3 * 3 = 27
    expected_nodes = 27

    actual_nodes = len(mesh.points)
    assert actual_nodes == expected_nodes, f"Expected {expected_nodes} nodes, got {actual_nodes}"


@pytest.mark.cpu
def test_box_mesh_element_count(simple_mesh):
    """Test exact element count for box mesh."""
    mesh = simple_mesh

    # For mesh_size=5 on a 10x10x10 box
    # 2 elements per dimension (10/5 = 2)
    # Total hexahedral elements = 2 * 2 * 2 = 8
    expected_elements = 8

    actual_elements = mesh.cells.shape[0]
    assert actual_elements == expected_elements


@pytest.mark.cpu
def test_box_mesh_cell_type(simple_mesh):
    """Test that box mesh uses hexahedral elements."""
    mesh = simple_mesh

    # Check element type
    assert mesh.ele_type == "HEX8"

    # Check cell connectivity shape
    assert mesh.cells.shape == (8, 8)  # 8 elements, 8 nodes per element


@pytest.mark.cpu
def test_box_mesh_different_sizes():
    """Test mesh generation with different mesh sizes."""
    L, W, H = 10, 10, 10

    # Coarser mesh
    mesh_coarse = fe.mesh.box_mesh((L, W, H), mesh_size=10)
    nodes_coarse = len(mesh_coarse.points)

    # Finer mesh
    mesh_fine = fe.mesh.box_mesh((L, W, H), mesh_size=2.5)
    nodes_fine = len(mesh_fine.points)

    # Finer mesh should have more nodes
    assert nodes_fine > nodes_coarse

    # Check exact counts
    # mesh_size=10: 1 element per dimension -> 2^3 = 8 nodes
    # mesh_size=2.5: 4 elements per dimension -> 5^3 = 125 nodes
    assert nodes_coarse == 8
    assert nodes_fine == 125


@pytest.mark.cpu
def test_box_mesh_aspect_ratio():
    """Test mesh generation with non-cubic dimensions."""
    L, W, H = 20, 10, 5
    mesh_size = 5

    mesh = fe.mesh.box_mesh((L, W, H), mesh_size=mesh_size)
    points = mesh.points

    # Check bounding box
    min_coords = jnp.min(points, axis=0)
    max_coords = jnp.max(points, axis=0)

    dimensions = max_coords - min_coords

    assert jnp.isclose(dimensions[0], 20., atol=1e-10)
    assert jnp.isclose(dimensions[1], 10., atol=1e-10)
    assert jnp.isclose(dimensions[2], 5., atol=1e-10)


@pytest.mark.cpu
def test_box_mesh_node_connectivity(simple_mesh):
    """Test that mesh elements reference valid node indices."""
    mesh = simple_mesh

    num_nodes = len(mesh.points)
    elements = mesh.cells

    # All node indices should be within valid range
    assert jnp.all(elements >= 0)
    assert jnp.all(elements < num_nodes)

    # No element should have duplicate nodes (each HEX8 element has 8 unique nodes)
    for elem in elements:
        assert len(jnp.unique(elem)) == 8


@pytest.mark.cpu
def test_box_mesh_reproducibility():
    """Test that mesh generation is deterministic."""
    L, W, H = 10, 10, 10
    mesh_size = 5

    mesh1 = fe.mesh.box_mesh((L, W, H), mesh_size=mesh_size)
    mesh2 = fe.mesh.box_mesh((L, W, H), mesh_size=mesh_size)

    # Same parameters should produce identical meshes
    assert jnp.allclose(mesh1.points, mesh2.points)
    assert jnp.array_equal(mesh1.cells, mesh2.cells)
