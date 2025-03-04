#!/usr/bin/env python3
"""
Utilities for loading and processing mesh files for the PLAYGROUND project.
Initially focused on basic mesh loading, but will extend to handling
Gaussian Splats in future phases.
"""

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p


def load_mesh(file_path):
    """
    Load a mesh file (PLY, OBJ, STL, etc.) using trimesh.
    
    Args:
        file_path (str): Path to the mesh file
        
    Returns:
        trimesh.Trimesh: Loaded mesh
    """
    try:
        mesh = trimesh.load(file_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None


def visualize_mesh(mesh, title="Mesh Visualization"):
    """
    Visualize a mesh using matplotlib.
    
    Args:
        mesh (trimesh.Trimesh): Mesh to visualize
        title (str): Title for the plot
    """
    if mesh is None:
        print("No mesh provided for visualization")
        return
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Plot mesh
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                    triangles=faces, cmap='viridis', alpha=0.7)
    
    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Ensure axes are scaled equally
    max_range = np.max([
        np.ptp(vertices[:, 0]),
        np.ptp(vertices[:, 1]),
        np.ptp(vertices[:, 2])
    ])
    mid_x = np.mean([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    mid_y = np.mean([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    mid_z = np.mean([np.min(vertices[:, 2]), np.max(vertices[:, 2])])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()


def mesh_to_vhacd(input_file, output_file, log_file="vhacd_log.txt", pybullet_path=None):
    """
    Convert a mesh to a convex hull decomposition using PyBullet's VHACD.
    This is useful for creating collision meshes for complex objects.
    
    Args:
        input_file (str): Path to input mesh file
        output_file (str): Path to output mesh file
        log_file (str): Path to log file
        pybullet_path (str): Path to PyBullet executable (if None, uses python module)
    
    Returns:
        bool: Success or failure
    """
    try:
        # If pybullet_path is provided, use command line VHACD
        if pybullet_path:
            import subprocess
            cmd = [
                pybullet_path,
                "--converter=VHACD",
                f"--input={input_file}",
                f"--output={output_file}",
                f"--log={log_file}"
            ]
            subprocess.run(cmd, check=True)
        else:
            # Use PyBullet Python API
            p.vhacd(
                input_file,
                output_file,
                log_file
            )
        return True
    except Exception as e:
        print(f"Error creating convex decomposition: {e}")
        return False


def load_mesh_pybullet(bullet_client, mesh_path, position=[0, 0, 0], orientation=[0, 0, 0, 1], 
                      scale=1.0, mass=1.0, use_maximalcoordinates=False, use_fixed_base=False):
    """
    Load a mesh file into PyBullet simulation.
    
    Args:
        bullet_client: PyBullet client instance
        mesh_path (str): Path to the mesh file (.obj, .stl, etc.)
        position (list): Initial position [x, y, z]
        orientation (list): Initial orientation quaternion [x, y, z, w]
        scale (float): Scaling factor
        mass (float): Mass of the object (0 for static)
        use_maximalcoordinates (bool): Use maximal coordinates
        use_fixed_base (bool): Fix the base of the object
        
    Returns:
        int: Object ID in PyBullet
    """
    # Check if file exists
    if not os.path.exists(mesh_path):
        print(f"Mesh file not found: {mesh_path}")
        return None
    
    # Get file extension
    _, ext = os.path.splitext(mesh_path)
    ext = ext.lower()
    
    if ext not in ['.obj', '.stl', '.urdf']:
        print(f"Unsupported file format: {ext}. Convert to .obj, .stl, or .urdf")
        return None
    
    try:
        # For URDF files
        if ext == '.urdf':
            object_id = bullet_client.loadURDF(
                mesh_path,
                basePosition=position,
                baseOrientation=orientation,
                globalScaling=scale,
                useMaximalCoordinates=use_maximalcoordinates,
                useFixedBase=use_fixed_base
            )
        # For OBJ/STL files
        else:
            # Create collision and visual shape
            visual_shape_id = bullet_client.createVisualShape(
                shapeType=bullet_client.GEOM_MESH,
                fileName=mesh_path,
                rgbaColor=[1, 1, 1, 1],
                specularColor=[0.4, 0.4, 0.4],
                visualFramePosition=[0, 0, 0],
                meshScale=[scale, scale, scale]
            )
            
            collision_shape_id = bullet_client.createCollisionShape(
                shapeType=bullet_client.GEOM_MESH,
                fileName=mesh_path,
                collisionFramePosition=[0, 0, 0],
                meshScale=[scale, scale, scale]
            )
            
            # Create multi-body
            object_id = bullet_client.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=position,
                baseOrientation=orientation,
                useMaximalCoordinates=use_maximalcoordinates
            )
        
        return object_id
    
    except Exception as e:
        print(f"Error loading mesh in PyBullet: {e}")
        return None


def create_primitive_object(bullet_client, shape_type, position=[0, 0, 0], 
                           orientation=[0, 0, 0, 1], dimensions=None, mass=1.0):
    """
    Create a primitive shape in PyBullet.
    
    Args:
        bullet_client: PyBullet client instance
        shape_type (str): One of 'box', 'sphere', 'cylinder', or 'capsule'
        position (list): Initial position [x, y, z]
        orientation (list): Initial orientation quaternion [x, y, z, w]
        dimensions: Shape-specific dimensions:
            - box: [length, width, height]
            - sphere: radius
            - cylinder: [radius, height]
            - capsule: [radius, height]
        mass (float): Mass of the object (0 for static)
        
    Returns:
        int: Object ID in PyBullet
    """
    # Define default dimensions if none provided
    if dimensions is None:
        if shape_type == 'sphere':
            dimensions = 1.0
        elif shape_type == 'box':
            dimensions = [1.0, 1.0, 1.0]
        elif shape_type in ['cylinder', 'capsule']:
            dimensions = [0.5, 1.0]
    
    try:
        # Create collision shape
        if shape_type == 'box':
            collision_shape_id = bullet_client.createCollisionShape(
                shapeType=bullet_client.GEOM_BOX,
                halfExtents=[dimensions[0]/2, dimensions[1]/2, dimensions[2]/2]
            )
            visual_shape_id = bullet_client.createVisualShape(
                shapeType=bullet_client.GEOM_BOX,
                halfExtents=[dimensions[0]/2, dimensions[1]/2, dimensions[2]/2],
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
        elif shape_type == 'sphere':
            collision_shape_id = bullet_client.createCollisionShape(
                shapeType=bullet_client.GEOM_SPHERE,
                radius=dimensions
            )
            visual_shape_id = bullet_client.createVisualShape(
                shapeType=bullet_client.GEOM_SPHERE,
                radius=dimensions,
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
        elif shape_type == 'cylinder':
            collision_shape_id = bullet_client.createCollisionShape(
                shapeType=bullet_client.GEOM_CYLINDER,
                radius=dimensions[0],
                height=dimensions[1]
            )
            visual_shape_id = bullet_client.createVisualShape(
                shapeType=bullet_client.GEOM_CYLINDER,
                radius=dimensions[0],
                length=dimensions[1],
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
        elif shape_type == 'capsule':
            collision_shape_id = bullet_client.createCollisionShape(
                shapeType=bullet_client.GEOM_CAPSULE,
                radius=dimensions[0],
                height=dimensions[1]
            )
            visual_shape_id = bullet_client.createVisualShape(
                shapeType=bullet_client.GEOM_CAPSULE,
                radius=dimensions[0],
                length=dimensions[1],
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
        else:
            print(f"Unsupported shape type: {shape_type}")
            return None
        
        # Create multi-body
        object_id = bullet_client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=orientation
        )
        
        return object_id
    
    except Exception as e:
        print(f"Error creating primitive shape: {e}")
        return None


# Future Phase 2 additions:
# - Gaussian Splat loading and visualization
# - Conversion from Gaussian Splats to meshes
# - Segmentation utilities for processing generative AI outputs