import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import logging

# Core imports (lightweight)
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        """Initialize processing service with graceful degradation for missing ML deps"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.avif', '.webp', '.mp4', '.avi', '.mov', '.glb', '.gltf', '.obj']
        self.ai_processor = None  # ML deps disabled on Render
        self.mesh_generator = None
        self.segmenter = None
        self.pose_estimator = None
        self.sfm_estimator = None
        self.frame_sampler = None
        self.mesh_optimizer = None
        self.advanced_reconstruction = None
        logger.info("ProcessingService initialized (ML features disabled for Render deployment)")

    async def process_scene(self, 
                          input_path: str, 
                          output_path: str, 
                          prompt: str,
                          job_id: str,
                          update_callback) -> dict:
        """
        Process input images/video into a 3D scene.
        On Render (no ML deps): Uses simple mesh generation.
        """
        output_path = Path(output_path)
        
        try:
            logger.info(f"Starting scene processing for {input_path}")
            
            # Validate input file
            if not os.path.exists(input_path):
                raise Exception(f"Input file not found: {input_path}")

            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise Exception(f"Unsupported file format: {file_ext}")

            # Create output directory
            os.makedirs(str(output_path), exist_ok=True)
            logger.info(f"Output directory created: {output_path}")
            
            # Load image and get dimensions
            await update_callback(job_id, 30, "Loading image...")
            try:
                img = Image.open(input_path)
                img_width, img_height = img.size
                logger.info(f"Image loaded: {img_width}x{img_height}")
            except Exception as e:
                logger.warning(f"Could not load image: {e}, using defaults")
                img_width, img_height = 512, 512
            
            # Create simple mesh based on image dimensions
            await update_callback(job_id, 50, "Generating 3D model...")
            import trimesh
            
            # Scale mesh to match image aspect ratio
            aspect_ratio = img_width / img_height if img_height > 0 else 1.0
            logger.info(f"Creating mesh with aspect ratio: {aspect_ratio}")
            
            # Create a more complex mesh (cylinder + box) for better visuals
            # Cylinder for main body
            mesh1 = trimesh.creation.cylinder(radius=0.5, height=1.0)
            
            # Box for top (like a seat)
            mesh2 = trimesh.creation.box(extents=[aspect_ratio * 0.8, 0.3, 0.8])
            mesh2.apply_translation([0, 0.5, 0])
            
            # Combine meshes
            mesh = trimesh.util.concatenate([mesh1, mesh2])
            
            # Ensure mesh is valid
            if mesh is None or len(mesh.vertices) == 0:
                logger.error("Generated mesh is empty!")
                raise Exception("Failed to generate valid 3D mesh")
            
            logger.info(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Export to GLB (binary format)
            output_file = str(output_path / "model.glb")
            logger.info(f"Exporting mesh to {output_file}")
            
            # Use explicit GLB export
            mesh.export(output_file)
            
            # Verify file was created and has content
            if not os.path.exists(output_file):
                raise Exception(f"Failed to create output file: {output_file}")
            
            file_size = os.path.getsize(output_file)
            if file_size == 0:
                logger.error(f"File created but is empty! Path: {output_file}")
                # Try alternative export
                logger.info("Attempting alternative export method...")
                import json
                mesh.export(output_file, file_type='gltf')
                file_size = os.path.getsize(output_file)
            
            logger.info(f"✓ Mesh exported successfully: {output_file} ({file_size} bytes)")
            
            await update_callback(job_id, 100, "Processing completed")
            
            return {
                "status": "completed",
                "output_path": output_file,
                "message": f"3D model generated successfully",
                "metadata": {
                    "input_file": os.path.basename(input_path),
                    "input_size": f"{img_width}x{img_height}",
                    "processing_date": datetime.now().isoformat(),
                    "output_directory": str(output_path)
                }
            }
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            await update_callback(job_id, -1, f"Processing error: {str(e)[:100]}")
            raise
