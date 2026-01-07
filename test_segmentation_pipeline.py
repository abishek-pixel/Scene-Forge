#!/usr/bin/env python3
"""
Test script: Segmentation-based Single Image 3D Reconstruction
Tests the new segmentation pipeline for single images.
"""

import os
import sys
import numpy as np
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SceneForge_Backend'))

def test_segmentation_pipeline():
    """Test the complete segmentation + meshing pipeline."""
    
    print("\n" + "="*70)
    print("TEST: Segmentation-Based Single Image 3D Reconstruction")
    print("="*70 + "\n")
    
    try:
        # Import components
        logger.info("Importing components...")
        from app.core.services.ai_processor import AIProcessor
        from app.core.services.mesh_generator import MeshGenerator
        from app.core.services.segmentation import SemanticSegmenter, SimpleSegmenter
        
        logger.info("✓ All imports successful\n")
        
        # Initialize
        logger.info("Initializing processors...")
        ai_processor = AIProcessor()
        mesh_generator = MeshGenerator()
        segmenter = SemanticSegmenter()
        logger.info("✓ All components initialized\n")
        
        # Load test image
        test_image_path = "SceneForge_Backend/test_image.jpg"
        if not os.path.exists(test_image_path):
            logger.warning(f"Test image not found at {test_image_path}")
            logger.info("Skipping real test, showing pipeline structure instead")
            print_pipeline_structure()
            return
        
        logger.info(f"Loading test image: {test_image_path}")
        rgb_image = np.array(Image.open(test_image_path).convert('RGB'))
        logger.info(f"✓ Image loaded: {rgb_image.shape}\n")
        
        # Step 1: Depth estimation
        logger.info("Step 1: Generating depth map...")
        depth_map = ai_processor._generate_depth_map(Image.fromarray(rgb_image))
        logger.info(f"✓ Depth map generated: {depth_map.shape}\n")
        
        # Step 2: Segmentation
        logger.info("Step 2: Segmenting foreground object...")
        if segmenter.model is not None:
            mask, _ = segmenter.segment_image(rgb_image)
            logger.info(f"✓ Semantic segmentation: {np.sum(mask)} foreground pixels")
        else:
            mask = SimpleSegmenter.segment_by_depth(depth_map)
            logger.info(f"✓ Depth-based segmentation: {np.sum(mask)} foreground pixels")
        
        # Step 3: Apply mask to depth
        logger.info("\nStep 3: Applying mask to depth...")
        segmented_depth = depth_map.copy()
        segmented_depth[mask == 0] = 0
        logger.info(f"✓ Masked depth: {np.sum(mask)} pixels have valid depth\n")
        
        # Step 4: Point cloud
        logger.info("Step 4: Creating point cloud from segmented depth...")
        pcd = mesh_generator.depth_to_point_cloud(segmented_depth, rgb_image)
        logger.info(f"✓ Point cloud: {len(np.asarray(pcd.points))} points\n")
        
        # Step 5: Downsampling
        logger.info("Step 5: Smart downsampling...")
        pcd_downsampled = mesh_generator.smart_downsample(pcd)
        logger.info(f"✓ Downsampled: {len(np.asarray(pcd_downsampled.points))} points\n")
        
        # Step 6: Mesh generation
        logger.info("Step 6: Generating mesh (Poisson reconstruction)...")
        mesh = mesh_generator.generate_mesh_poisson(pcd_downsampled, depth=11)
        vertices = len(np.asarray(mesh.vertices))
        triangles = len(np.asarray(mesh.triangles))
        logger.info(f"✓ Mesh generated: {vertices:,d} vertices, {triangles:,d} triangles\n")
        
        # Summary
        print("\n" + "="*70)
        print("SEGMENTATION PIPELINE TEST - RESULTS")
        print("="*70)
        print(f"\nInput Image:           {rgb_image.shape}")
        print(f"Depth Map:             {depth_map.shape}")
        print(f"Segmentation Mask:     {np.sum(mask)} foreground pixels")
        print(f"Point Cloud:           {len(np.asarray(pcd.points)):,d} points")
        print(f"Downsampled:           {len(np.asarray(pcd_downsampled.points)):,d} points")
        print(f"Final Mesh:            {vertices:,d} vertices, {triangles:,d} triangles")
        print("\n" + "="*70)
        print("✓ SEGMENTATION PIPELINE TEST PASSED")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def print_pipeline_structure():
    """Print the segmentation pipeline structure."""
    
    print("\n" + "="*70)
    print("SEGMENTATION-BASED SINGLE IMAGE PIPELINE")
    print("="*70 + "\n")
    
    pipeline = """
INPUT IMAGE (RGB + Depth map)
    ↓
[SEGMENTATION]
├─ SemanticSegmenter (Mask2Former)
│  └─ Generates binary foreground/background mask
├─ Fallback: SimpleSegmenter (depth-based)
│  └─ Identifies object region from depth discontinuities
    ↓
[MASK APPLICATION]
├─ Apply mask to depth map
├─ Set background pixels (mask==0) to depth=0
└─ Keep only object region
    ↓
[POINT CLOUD GENERATION]
├─ Convert masked depth → 3D points
├─ Filter out background (depth==0)
└─ Color with RGB image
    ↓
[SMART DOWNSAMPLING]
├─ Voxel-based downsampling
├─ Statistical outlier removal
└─ Result: Clean point cloud
    ↓
[MESH GENERATION]
├─ Poisson surface reconstruction (depth=11)
├─ Smooth, closed surface
└─ Result: Watertight mesh
    ↓
[MESH REFINEMENT]
├─ 10-step advanced pipeline
├─ Remove noise, smooth surfaces
└─ Professional quality
    ↓
[EXPORT]
├─ GLB format (PBR-ready)
├─ OBJ format (compatible)
└─ Metadata + preview images
    ↓
OUTPUT: CLEAN 3D MODEL (Only object, no background)
"""
    
    print(pipeline)
    
    print("="*70)
    print("KEY IMPROVEMENTS vs OLD PIPELINE")
    print("="*70)
    print("""
❌ OLD (Without Segmentation):
   RGB + Depth → Point Cloud → Mesh
   Problem: Includes entire scene (chair + floor + sky)
   Result: Shapes melt, background stretches, unusable models

✅ NEW (With Segmentation):
   RGB + Depth → Segment → Mask → Point Cloud → Mesh
   Benefit: Only the object is meshed
   Result: Clean, solid models with correct silhouettes
""")
    
    print("="*70)
    print("WHY THIS WORKS")
    print("="*70)
    print("""
1. Segmentation isolates the object region
2. Masked depth = background is invisible (depth=0)
3. Point cloud only has object points
4. Poisson reconstruction sees only object shape
5. Result: No more background artifacts

This is the industry-standard approach for single-image
3D reconstruction (used in all modern apps).
""")


if __name__ == "__main__":
    success = test_segmentation_pipeline()
    sys.exit(0 if success else 1)
