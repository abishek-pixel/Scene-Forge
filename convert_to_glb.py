#!/usr/bin/env python3
"""Convert test OBJ mesh to GLB format."""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "SceneForge_Backend"
sys.path.insert(0, str(backend_path))

import open3d as o3d

def convert_to_glb():
    """Convert OBJ to GLB."""
    obj_file = 'test_outputs/test_refined_mesh.obj'
    glb_file = 'test_outputs/test_refined_mesh.glb'
    
    print('Loading OBJ mesh...')
    mesh = o3d.io.read_triangle_mesh(obj_file)
    
    print('Exporting to GLB format...')
    o3d.io.write_triangle_mesh(glb_file, mesh)
    
    if os.path.exists(glb_file):
        file_size = os.path.getsize(glb_file) / (1024 * 1024)
        print(f'\n✓ GLB file created successfully!')
        print(f'  Location: {os.path.abspath(glb_file)}')
        print(f'  File size: {file_size:.2f} MB')
        print(f'  Vertices: {len(mesh.vertices):,d}')
        print(f'  Triangles: {len(mesh.triangles):,d}')
        return True
    else:
        print('Error creating GLB file')
        return False

if __name__ == '__main__':
    success = convert_to_glb()
    sys.exit(0 if success else 1)
