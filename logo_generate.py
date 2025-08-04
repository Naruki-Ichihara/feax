#!/usr/bin/env python3
"""
Generate FEAX logo with a simple tetrahedron.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Configuration variables
ELEVATION = 25          # Viewing angle elevation
AZIMUTH = 75           # Viewing angle azimuth
FACE_SEPARATION = 0.02  # Gap between faces
ACCENT_COLOR = "#B22222"  # Dark red accent color (FireBrick)
GRAY_COLOR = "#C0C0C0"    # Light gray color for other faces (Silver)
ACCENT_FACE = 2          # Which face to highlight (0-3)
FIGURE_SIZE = (8, 8)     # Figure dimensions
DPI = 500               # Output resolution
FACE_ALPHA = 0.8        # Face transparency

def create_tetrahedron():
    """Create tetrahedron vertices."""
    # Regular tetrahedron vertices
    vertices = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3)
    
    # Define faces (triangles)
    faces = [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[3]],
        [vertices[1], vertices[2], vertices[3]]
    ]
    
    return vertices, faces

def explode_faces(faces, separation=0.15):
    """Separate faces outward from center."""
    exploded_faces = []
    
    for face in faces:
        # Calculate face center
        face_center = np.mean(face, axis=0)
        
        # Calculate direction from origin to face center
        direction = face_center / np.linalg.norm(face_center)
        
        # Move face outward
        offset = direction * separation
        exploded_face = face + offset
        
        exploded_faces.append(exploded_face)
    
    return exploded_faces

def generate_logo(output_file='assets/logo.svg'):
    """Generate FEAX logo with realistic tetrahedron."""
    
    fig = plt.figure(figsize=FIGURE_SIZE, facecolor='none')
    ax = fig.add_subplot(111, projection='3d')
    
    # Create tetrahedron
    vertices, faces = create_tetrahedron()
    
    # Explode faces for separated view
    exploded_faces = explode_faces(faces, separation=FACE_SEPARATION)
    
    # Set up face colors with accent color
    face_colors = [GRAY_COLOR] * 4
    face_colors[ACCENT_FACE] = ACCENT_COLOR
    
    # Create individual faces with different shading
    for i, face in enumerate(exploded_faces):
        face_collection = Poly3DCollection([face], alpha=FACE_ALPHA, linewidths=0)
        face_collection.set_facecolor(face_colors[i])
        face_collection.set_edgecolor('none')
        ax.add_collection3d(face_collection)
    
    # Clean tetrahedron without vertex markers
    
    # Set axis properties with tight bounds around tetrahedron
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])
    
    # Set equal aspect ratio for true isometric view
    ax.set_box_aspect([1,1,1])
    
    # Remove axes for clean logo
    ax.set_axis_off()
    
    # Set perspective view like classic tetrahedron diagram
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    
    # Add transparent background
    ax.set_facecolor('none')
    
    # Clean logo without text
    
    # Save logo
    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    
    print(f"Logo saved as: {output_file}")
    return fig

if __name__ == "__main__":
    # Generate logo
    fig = generate_logo()
    plt.show()