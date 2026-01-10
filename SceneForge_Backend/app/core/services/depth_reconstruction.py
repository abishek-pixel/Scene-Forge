"""
Depth Estimation & 3D Reconstruction Module
Uses MiDaS for monocular depth estimation to create 3D models from single images.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
import logging
from typing import Tuple, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class DepthReconstructionPipeline:
    """
    Reconstructs 3D geometry from a single image using depth estimation.
    """
    
    # Class-level cache for MiDaS model (loaded once, reused)
    _midas_model = None
    _midas_transform = None
    _model_device = None
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"DepthReconstructionPipeline initialized on {self.device}")
    
    @classmethod
    def _get_midas_model(cls, device):
        """Get cached MiDaS model or load it once"""
        if cls._midas_model is None:
            logger.info(f"Loading MiDaS model for device: {device}")
            try:
                # Load MiDaS model
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                midas.to(device)
                midas.eval()
                
                # Load transforms
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                transform = midas_transforms.small_transform
                
                cls._midas_model = midas
                cls._midas_transform = transform
                cls._model_device = device
                logger.info("✓ MiDaS model loaded and cached")
            except Exception as e:
                logger.error(f"Failed to load MiDaS model: {e}")
                raise
        
        return cls._midas_model, cls._midas_transform
    
    def _load_midas_model(self):
        """Load or get cached MiDaS model"""
        self.midas_model, self.midas_transform = self._get_midas_model(self.device)
    
    def estimate_depth(self, image_path: str) -> np.ndarray:
        """
        Estimate depth map from image using MiDaS
        
        Args:
            image_path: Path to input image
            
        Returns:
            Normalized depth map (0-1)
        """
        logger.info(f"Estimating depth for {image_path}")
        self._load_midas_model()
        
        try:
            # Load and prepare image
            logger.info("Loading image...")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            logger.info(f"Original image size: {img.shape[1]}x{img.shape[0]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Downsample for faster processing on CPU
            # MiDaS works well with smaller images
            max_size = 512
            h, w = img.shape[:2]
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                w_new = int(w * scale)
                h_new = int(h * scale)
                logger.info(f"Downsampling to {w_new}x{h_new} for faster processing")
                img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            
            logger.info(f"Processing image: {img.shape[1]}x{img.shape[0]}")
            
            # Transform for MiDaS
            logger.info("Transforming image for MiDaS...")
            input_batch = self.midas_transform(img).to(self.device)
            
            # Estimate depth
            logger.info("Running depth estimation (this may take 10-30s on CPU)...")
            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                logger.info("Interpolating depth map...")
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            logger.info("Normalizing depth...")
            # Normalize depth to 0-1
            depth_min = prediction.min()
            depth_max = prediction.max()
            if depth_max > depth_min:
                depth_normalized = (prediction - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = prediction
            
            depth_map = depth_normalized.cpu().numpy()
            logger.info(f"✓ Depth estimation complete: {depth_map.shape}")
            
            return depth_map
        
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}", exc_info=True)
            raise
    
    def depth_to_point_cloud(self, depth_map: np.ndarray, image: Image.Image, 
                            focal_length: Optional[float] = None) -> np.ndarray:
        """
        Convert depth map to 3D point cloud
        
        Args:
            depth_map: Depth map (0-1 normalized)
            image: Original RGB image
            focal_length: Camera focal length (estimated if None)
            
        Returns:
            Point cloud as Nx3 array (x, y, z coordinates)
        """
        logger.info("Converting depth map to point cloud...")
        
        h, w = depth_map.shape
        
        # Estimate focal length based on image size if not provided
        if focal_length is None:
            focal_length = w / (2 * np.tan(np.deg2rad(55) / 2))
        
        # Create coordinate grids
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        # Scale depth to realistic range (0.5 to 5 units)
        depth_scaled = 0.5 + depth_map * 4.5
        
        # Convert to 3D coordinates (camera frame)
        z = depth_scaled
        x_3d = (xx - w / 2) * z / focal_length
        y_3d = (yy - h / 2) * z / focal_length
        
        # Stack into point cloud
        points = np.stack([x_3d, y_3d, z], axis=-1)
        points = points.reshape(-1, 3)
        
        logger.info(f"Point cloud created: {points.shape[0]} points")
        
        return points
    
    def create_mesh_from_depth(self, depth_map: np.ndarray, image: Image.Image) -> object:
        """
        Create 3D mesh from depth map using height field
        
        Args:
            depth_map: Depth map (normalized 0-1)
            image: Original RGB image for reference
            
        Returns:
            trimesh.Trimesh object
        """
        logger.info("Creating mesh from depth map...")
        import trimesh
        
        try:
            h, w = depth_map.shape
            logger.info(f"Depth map shape: {h}x{w}")
            
            # Ensure depth map is properly normalized
            if depth_map.max() > 1.0:
                depth_map = depth_map / depth_map.max()
            
            # Scale depth to height (0.1 to 5 units)
            heights = 0.1 + depth_map * 4.9
            logger.info(f"Height range: {heights.min():.3f} to {heights.max():.3f}")
            
            # Create vertices from height map
            vertices = []
            for i in range(h):
                for j in range(w):
                    vertices.append([j / w, i / h, heights[i, j]])
            
            vertices = np.array(vertices, dtype=np.float32)
            logger.info(f"Created {len(vertices)} vertices")
            
            # Create faces (triangles) from quad grid
            faces = []
            for i in range(h - 1):
                for j in range(w - 1):
                    # Current quad corners
                    v0 = i * w + j
                    v1 = i * w + (j + 1)
                    v2 = (i + 1) * w + (j + 1)
                    v3 = (i + 1) * w + j
                    
                    # Two triangles per quad
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])
            
            faces = np.array(faces, dtype=np.uint32)
            logger.info(f"Created {len(faces)} faces")
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            logger.info(f"✓ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Basic validation
            if not mesh.is_valid:
                logger.warning("Mesh is not valid, attempting to fix...")
                mesh.remove_degenerate_faces()
                mesh.remove_infinite_values()
            
            return mesh
            
        except Exception as e:
            logger.error(f"Failed to create mesh from depth: {e}", exc_info=True)
            raise
    
    def _smooth_mesh(self, mesh: object, iterations: int = 3) -> object:
        """
        Smooth mesh using Laplacian smoothing
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed mesh
        """
        try:
            for _ in range(iterations):
                # Simple Laplacian smoothing
                mesh = mesh.smoothed()
            
            logger.info(f"Mesh smoothed with {iterations} iterations")
            return mesh
        except Exception as e:
            logger.warning(f"Mesh smoothing failed: {e}, using original mesh")
            return mesh
    
    def process_image_to_3d(self, image_path: str, output_path: str) -> Tuple[str, dict]:
        """
        Complete pipeline: Image -> Depth -> Mesh -> GLB
        
        Args:
            image_path: Path to input image
            output_path: Path to save output GLB
            
        Returns:
            Tuple of (output_file_path, metadata)
        """
        import time
        start_time = time.time()
        logger.info(f"Starting 3D reconstruction pipeline for {image_path}")
        
        try:
            # Load original image
            logger.info("Loading image...")
            img_pil = Image.open(image_path)
            img_pil.verify()
            img_pil = Image.open(image_path)  # Re-open after verify
            logger.info(f"✓ Image loaded: {img_pil.size}")
            
            # Estimate depth
            logger.info("Estimating depth map...")
            depth_start = time.time()
            depth_map = self.estimate_depth(image_path)
            logger.info(f"✓ Depth estimation complete in {time.time() - depth_start:.2f}s")
            
            # Create mesh from depth
            logger.info("Converting depth to 3D mesh...")
            mesh_start = time.time()
            mesh = self.create_mesh_from_depth(depth_map, img_pil)
            logger.info(f"✓ Mesh creation complete in {time.time() - mesh_start:.2f}s")
            
            # Export to GLB
            logger.info(f"Exporting to GLB: {output_path}")
            export_start = time.time()
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export with error checking
            try:
                mesh.export(output_path)
            except Exception as e:
                logger.warning(f"GLB export failed: {e}, trying with file_type parameter...")
                mesh.export(output_path, file_type='glb')
            
            logger.info(f"✓ Export complete in {time.time() - export_start:.2f}s")
            
            # Verify file
            if not Path(output_path).exists():
                raise Exception(f"Failed to create output file: {output_path}")
            
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                raise Exception(f"Output file is empty: {output_path}")
            
            total_time = time.time() - start_time
            logger.info(f"✓ 3D reconstruction complete in {total_time:.2f}s")
            logger.info(f"  - File: {output_path}")
            logger.info(f"  - Size: {file_size / 1024:.1f} KB")
            logger.info(f"  - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
            
            metadata = {
                "input_file": Path(image_path).name,
                "output_file": output_path,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "file_size_bytes": file_size,
                "processing_time_seconds": total_time,
                "reconstruction_method": "MiDaS Depth + Height Field Mesh"
            }
            
            return output_path, metadata
            
        except Exception as e:
            logger.error(f"3D reconstruction failed: {e}", exc_info=True)
            raise
