#!/usr/bin/env python3
"""
Test script for the simplified, stable mesh refinement pipeline.
This tests the mesh generation and refinement without video processing.
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_simplified_refinement():
    """Test the simplified mesh refinement pipeline."""
    try:
        # Import required modules
        logger.info("Importing dependencies...")
        import numpy as np
        import open3d as o3d
        
        # Add backend to path
        backend_path = Path(__file__).parent / "SceneForge_Backend"
        sys.path.insert(0, str(backend_path))
        
        logger.info("Importing mesh services...")
        from app.core.services.mesh_generator import MeshGenerator
        from app.core.services.mesh_refinement import MeshRefinement
        
        # Create test point cloud
        logger.info("\n" + "="*60)
        logger.info("CREATING TEST POINT CLOUD")
        logger.info("="*60)
        
        # Generate synthetic data (bottle-like shape)
        np.random.seed(42)
        points = []
        
        # Main body (cylinder)
        theta = np.linspace(0, 2*np.pi, 200)
        for z in np.linspace(0, 1, 100):
            r = 0.3 + 0.1 * np.sin(z * 10)  # Wavy surface
            x = r * np.cos(theta) + np.random.normal(0, 0.01, len(theta))
            y = r * np.sin(theta) + np.random.normal(0, 0.01, len(theta))
            z_vals = np.full_like(x, z)
            points.extend(list(zip(x, y, z_vals)))
        
        # Neck (narrower cylinder)
        for z in np.linspace(1, 1.3, 50):
            r = 0.15 + 0.02 * np.random.randn()
            x = r * np.cos(theta) + np.random.normal(0, 0.005, len(theta))
            y = r * np.sin(theta) + np.random.normal(0, 0.005, len(theta))
            z_vals = np.full_like(x, z)
            points.extend(list(zip(x, y, z_vals)))
        
        # Base (wider, flat)
        for z in np.linspace(-0.1, 0, 30):
            r = 0.35 * (1 - abs(z) / 0.1)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z_vals = np.full_like(x, z)
            points.extend(list(zip(x, y, z_vals)))
        
        points = np.array(points, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        logger.info(f"Created point cloud with {len(pcd.points):,d} points")
        logger.info(f"Bounds: {pcd.get_axis_aligned_bounding_box()}")
        
        # Estimate normals
        logger.info("\nEstimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
        )
        logger.info("✓ Normals estimated")
        
        # Generate mesh
        logger.info("\n" + "="*60)
        logger.info("GENERATING MESH - POISSON RECONSTRUCTION")
        logger.info("="*60)
        
        mesh_gen = MeshGenerator()
        mesh = mesh_gen.generate_mesh_poisson(pcd, depth=10)
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.error("Mesh generation failed!")
            return False
        
        logger.info(f"Generated mesh: {len(mesh.vertices):,d} vertices, {len(mesh.triangles):,d} triangles")
        
        # Apply refinement pipeline
        logger.info("\n" + "="*60)
        logger.info("APPLYING SIMPLIFIED REFINEMENT PIPELINE")
        logger.info("="*60)
        
        refined_mesh = MeshRefinement.complete_refinement_pipeline(mesh, aggressive=False)
        
        if refined_mesh is None or len(refined_mesh.vertices) == 0:
            logger.error("Refinement pipeline failed or returned empty mesh!")
            return False
        
        logger.info(f"\n✓ Refinement complete:")
        logger.info(f"  Input:  {len(mesh.vertices):,d} vertices")
        logger.info(f"  Output: {len(refined_mesh.vertices):,d} vertices")
        
        # Save test meshes
        logger.info("\n" + "="*60)
        logger.info("SAVING TEST RESULTS")
        logger.info("="*60)
        
        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        
        original_file = output_dir / "test_original_mesh.obj"
        refined_file = output_dir / "test_refined_mesh.obj"
        
        o3d.io.write_triangle_mesh(str(original_file), mesh)
        o3d.io.write_triangle_mesh(str(refined_file), refined_mesh)
        
        logger.info(f"✓ Original mesh saved: {original_file}")
        logger.info(f"✓ Refined mesh saved: {refined_file}")
        
        logger.info("\n" + "="*60)
        logger.info("TEST COMPLETED SUCCESSFULLY ✓✓✓")
        logger.info("="*60)
        logger.info("\nThe simplified pipeline is working correctly!")
        logger.info("No NoneType errors or crashes occurred.")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_simplified_refinement()
    sys.exit(0 if success else 1)
