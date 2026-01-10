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
        """Initialize processing service with ML-based reconstruction"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.avif', '.webp', '.mp4', '.avi', '.mov', '.glb', '.gltf', '.obj']
        self.depth_pipeline = None  # Lazy load
        logger.info("ProcessingService initialized (ML-based 3D reconstruction enabled)")

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
            
            # Use ML-based 3D reconstruction
            await update_callback(job_id, 50, "Generating 3D model with depth estimation...")
            
            try:
                # Lazy load depth reconstruction pipeline
                if self.depth_pipeline is None:
                    from .depth_reconstruction import DepthReconstructionPipeline
                    self.depth_pipeline = DepthReconstructionPipeline()
                
                logger.info(f"[Job {job_id}] Starting depth-based 3D reconstruction")
                
                # Generate 3D model using depth estimation with timeout
                output_file = str(output_path / "model.glb")
                
                try:
                    import asyncio
                    # Run with 120 second timeout
                    logger.info(f"[Job {job_id}] Depth reconstruction with 120s timeout")
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self.depth_pipeline.process_image_to_3d,
                            input_path,
                            output_file
                        ),
                        timeout=120.0
                    )
                    logger.info(f"[Job {job_id}] ✓ ML-based reconstruction complete")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"[Job {job_id}] Depth reconstruction timed out after 120s, using fallback")
                    # Fallback to simple mesh if depth estimation takes too long
                    await update_callback(job_id, 60, "Timeout on depth estimation, using fallback mesh...")
                    raise TimeoutError("Depth estimation timed out")
                    
            except (TimeoutError, Exception) as e:
                logger.warning(f"[Job {job_id}] ML-based reconstruction failed or timed out: {e}, falling back to simple mesh")
                await update_callback(job_id, 60, "Using fallback 3D mesh generation...")
                
                # Fallback: simple geometric mesh
                import trimesh
                
                aspect_ratio = img_width / img_height if img_height > 0 else 1.0
                logger.info(f"[Job {job_id}] Creating fallback mesh with aspect ratio: {aspect_ratio}")
                
                # Create a simple but interesting shape as fallback
                mesh = trimesh.creation.box(extents=[aspect_ratio, 1.0, 0.5])
                
                output_file = str(output_path / "model.glb")
                mesh.export(output_file)
                
                logger.info(f"[Job {job_id}] ✓ Fallback mesh created: {output_file}")
            
            # Verify file
            if not os.path.exists(output_file):
                raise Exception(f"Failed to create output file: {output_file}")
            
            file_size = os.path.getsize(output_file)
            logger.info(f"Model file size: {file_size} bytes")
            
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
