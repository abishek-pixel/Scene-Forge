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
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.midas_model = None
        self.midas_transform = None
        logger.info(f"DepthReconstructionPipeline initialized on {self.device}")
    
    def _load_midas_model(self):
        """Load MiDaS model for depth estimation"""
        if self.midas_model is not None:
            return
        
        logger.info("Loading MiDaS depth estimation model...")
        try:
            # Load MiDaS model
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.to(self.device)
            midas.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.small_transform
            
            self.midas_model = midas
            self.midas_transform = transform
            logger.info("✓ MiDaS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise
    
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
        
        # Load and prepare image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Transform for MiDaS
        input_batch = self.midas_transform(img).to(self.device)
        
        # Estimate depth
        with torch.no_grad():
            prediction = self.midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Normalize depth to 0-1
        depth_min = prediction.min()
        depth_max = prediction.max()
        depth_normalized = (prediction - depth_min) / (depth_max - depth_min)
        
        depth_map = depth_normalized.cpu().numpy()
        logger.info(f"Depth estimation complete: {depth_map.shape}")
        
        return depth_map
    
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
        Create 3D mesh from depth map using point cloud + Poisson reconstruction
        
        Args:
            depth_map: Depth map
            image: Original RGB image for coloring
            
        Returns:
            trimesh.Trimesh object
        """
        logger.info("Creating mesh from depth map...")
        import trimesh
        
        try:
            # Get point cloud
            points = self.depth_to_point_cloud(depth_map, image)
            
            # Downsample for faster processing
            if len(points) > 50000:
                logger.info(f"Downsampling point cloud from {len(points)} to 50000 points")
                indices = np.random.choice(len(points), 50000, replace=False)
                points = points[indices]
            
            # Create simple mesh from depth map (height field)
            h, w = depth_map.shape
            
            # Scale depth to height
            heights = (depth_map * 2).astype(np.float32)
            
            # Create mesh from vertices and faces
            vertices = []
            faces = []
            
            for i in range(h - 1):
                for j in range(w - 1):
                    # Get 4 corners of quad
                    v1 = np.array([j, i, heights[i, j]])
                    v2 = np.array([j + 1, i, heights[i, j + 1]])
                    v3 = np.array([j + 1, i + 1, heights[i + 1, j + 1]])
                    v4 = np.array([j, i + 1, heights[i + 1, j]])
                    
                    idx_start = len(vertices)
                    vertices.extend([v1, v2, v3, v4])
                    
                    # Two triangles per quad
                    faces.append([idx_start, idx_start + 1, idx_start + 2])
                    faces.append([idx_start, idx_start + 2, idx_start + 3])
            
            vertices = np.array(vertices)
            faces = np.array(faces)
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Smooth mesh
            mesh = self._smooth_mesh(mesh)
            
            logger.info(f"✓ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
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
        Complete pipeline: Image -> Depth -> Point Cloud -> Mesh -> GLB
        
        Args:
            image_path: Path to input image
            output_path: Path to save output GLB
            
        Returns:
            Tuple of (output_file_path, metadata)
        """
        logger.info(f"Starting 3D reconstruction pipeline for {image_path}")
        
        try:
            # Load original image
            img_pil = Image.open(image_path)
            
            # Estimate depth
            depth_map = self.estimate_depth(image_path)
            
            # Create mesh from depth
            mesh = self.create_mesh_from_depth(depth_map, img_pil)
            
            # Export to GLB
            output_file = str(output_path)
            logger.info(f"Exporting to {output_file}")
            mesh.export(output_file)
            
            # Verify file
            if not Path(output_file).exists():
                raise Exception(f"Failed to create output file: {output_file}")
            
            file_size = Path(output_file).stat().st_size
            logger.info(f"✓ Model exported: {output_file} ({file_size} bytes)")
            
            metadata = {
                "input_file": Path(image_path).name,
                "output_file": output_file,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "file_size_bytes": file_size,
                "reconstruction_method": "MiDaS Depth + Height Field Mesh"
            }
            
            return output_file, metadata
            
        except Exception as e:
            logger.error(f"3D reconstruction failed: {e}", exc_info=True)
            raise
