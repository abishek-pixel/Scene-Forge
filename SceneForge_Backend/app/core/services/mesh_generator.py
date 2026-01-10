"""
Mesh generation from depth maps and RGB images.
Handles point cloud creation, mesh generation, and cleaning.
"""

from __future__ import annotations

import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import json
from PIL import Image
import logging
import gc

# Lazy imports for optional dependencies
def _import_o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None

def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None

logger = logging.getLogger(__name__)


class MeshGenerator:
    """Generate 3D meshes from depth maps and RGB images using point clouds."""
    
    def __init__(self, intrinsics: Optional[Dict[str, float]] = None):
        """
        Initialize mesh generator with camera intrinsics.
        
        Args:
            intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy', 'width', 'height'
                       If None, will use default intrinsics based on image size
        """
        self.intrinsics = intrinsics
    
    def depth_to_point_cloud(self, 
                            depth_map: np.ndarray, 
                            rgb_image: Optional[np.ndarray] = None,
                            intrinsics: Optional[Dict[str, float]] = None) -> o3d.geometry.PointCloud:
        """
        Convert depth map to point cloud.
        
        Args:
            depth_map: Depth map as numpy array (H x W)
            rgb_image: Optional RGB image for coloring (H x W x 3)
            intrinsics: Camera intrinsics (fx, fy, cx, cy). Auto-computed if None.
        
        Returns:
            Open3D PointCloud object
        """
        logger.info(f"Converting depth map to point cloud - shape: {depth_map.shape}")
        
        # Use provided intrinsics or fall back to stored ones
        if intrinsics is None:
            intrinsics = self.intrinsics
        
        # Auto-compute intrinsics from image size if not provided
        if intrinsics is None:
            h, w = depth_map.shape
            fx = fy = max(w, h)
            cx, cy = w / 2, h / 2
            logger.info(f"Auto-computed intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        else:
            fx = intrinsics.get('fx')
            fy = intrinsics.get('fy')
            cx = intrinsics.get('cx')
            cy = intrinsics.get('cy')
        
        h, w = depth_map.shape
        
        # Controlled downsampling to preserve detail for single-image reconstruction
        scale = 1
        # Use more conservative thresholds so single images keep more resolution
        if h * w > 2000000:  # Very large images
            scale = 3
        elif h * w > 1000000:
            scale = 2
        elif h * w > 500000:
            scale = 1  # keep full res for moderate sizes

        if scale > 1:
            depth_map = depth_map[::scale, ::scale]
            if rgb_image is not None:
                rgb_image = rgb_image[::scale, ::scale]
            logger.info(f"Downsampled depth map by {scale}x to reduce memory")
        
        h, w = depth_map.shape
        
        # Create meshgrid of pixel coordinates
        x = np.arange(w)
        y = np.arange(h)
        xv, yv = np.meshgrid(x, y)
        
        # Backproject to 3D using intrinsics
        z = depth_map.astype(np.float32)
        x_3d = (xv - cx) * z / fx
        y_3d = (yv - cy) * z / fy
        z_3d = z
        
        # Stack into point cloud
        points = np.stack([x_3d, y_3d, z_3d], axis=-1)  # (H, W, 3)
        points = points.reshape(-1, 3)  # (H*W, 3)
        
        # Remove invalid points (z <= 0)
        valid_mask = z_3d.reshape(-1) > 0
        points = points[valid_mask]
        
        logger.info(f"Point cloud created: {len(points)} valid points")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))  # Use float32 instead of float64 to save memory
        
        # Add colors if RGB image provided
        if rgb_image is not None:
            if rgb_image.shape[2] == 3:
                colors = rgb_image.reshape(-1, 3)[valid_mask].astype(np.float32) / 255.0  # float32
                pcd.colors = o3d.utility.Vector3dVector(colors)
                logger.info(f"Point cloud colored with RGB")
        
        # CRITICAL: Immediately downsample each frame's point cloud to prevent memory explosion
        # Video = 80 frames × 300K points per frame = 24M points without downsampling
        # For single images: moderate downsampling
        # For videos: VERY aggressive downsampling
        if len(pcd.points) > 300000:
            # Reduce voxel size to preserve more geometry while still limiting memory
            pcd = pcd.voxel_down_sample(voxel_size=0.02)
            logger.info(f"Aggressive downsampling (image): {len(pcd.points)} points remaining")
        elif len(pcd.points) > 200000:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            logger.info(f"Moderate downsampling: {len(pcd.points)} points remaining")
        
        gc.collect()
        return pcd
    
    def merge_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud],
                          downsample_voxel: float = 0.01) -> o3d.geometry.PointCloud:
        """
        Merge multiple point clouds into one with AGGRESSIVE memory optimization for video processing.
        
        Args:
            point_clouds: List of Open3D PointCloud objects (already pre-downsampled)
            downsample_voxel: Voxel size for downsampling merged cloud
        
        Returns:
            Merged and downsampled PointCloud
        """
        logger.info(f"Merging {len(point_clouds)} point clouds (streaming merge)")
        
        # ULTRA-AGGRESSIVE streaming merge - merge every 2-3 frames immediately
        merged = o3d.geometry.PointCloud()
        merge_frequency = max(2, len(point_clouds) // 40)  # Merge at least 40 times during processing
        
        for i, pcd in enumerate(point_clouds):
            # Add next point cloud
            merged += pcd
            
            # ULTRA-FREQUENT downsampling: every 2-3 frames or final merge
            if (i + 1) % merge_frequency == 0 or i == len(point_clouds) - 1:
                # Use moderate voxel size during streaming
                merged = merged.voxel_down_sample(voxel_size=0.02)
                points_after = len(merged.points)
                logger.info(f"Streaming merge: after frame {i+1}/{len(point_clouds)}, downsampled to {points_after} points")
                gc.collect()
                
                # If still too large, downsample even more aggressively
                if points_after > 800000:
                    merged = merged.voxel_down_sample(voxel_size=0.03)
                    logger.info(f"Emergency downsampling: reduced to {len(merged.points)} points")
                    gc.collect()
        
        logger.info(f"Merged cloud stabilized at {len(merged.points)} points")
        
        # Final safety downsample - ensure we never exceed ~500K points
        total_points = len(merged.points)
        if total_points > 800000:
            merged = merged.voxel_down_sample(voxel_size=0.03)
            logger.info(f"Final aggressive downsample: {total_points} → {len(merged.points)} points")
            gc.collect()
        elif total_points > 500000:
            merged = merged.voxel_down_sample(voxel_size=0.02)
            logger.info(f"Final moderate downsample: {total_points} → {len(merged.points)} points")
        
        logger.info(f"Merged point cloud FINAL size: {len(merged.points)} points")
        gc.collect()
        return merged
    
    def generate_mesh_poisson(self, 
                             point_cloud: o3d.geometry.PointCloud,
                             depth: int = 11,
                             min_density: float = 0.0) -> o3d.geometry.TriangleMesh:
        """
        Generate mesh from point cloud using Poisson reconstruction with memory optimization.
        
        Args:
            point_cloud: Input point cloud
            depth: Octree depth for Poisson (higher = finer detail, adaptive to point count)
            min_density: Remove low-density vertices (0.0-1.0, higher = more aggressive)
        
        Returns:
            Generated TriangleMesh
        """
        # Adaptive depth based on point cloud size to prevent memory issues
        num_points = len(point_cloud.points)
        if num_points > 2000000:
            depth = min(depth, 10)
            logger.info(f"Large point cloud ({num_points} points), using Poisson depth {depth}")
        elif num_points > 1000000:
            depth = min(depth, 11)
            logger.info(f"Large point cloud ({num_points} points), using Poisson depth {depth}")
        
        logger.info(f"Generating mesh using Poisson reconstruction (depth={depth})")
        
        # Estimate normals with improved parameters (required for Poisson)
        # Larger radius + more neighbors = better normals for noisy data
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        
        # Orient normals consistently
        point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
        logger.info("Normals estimated and oriented")
        
        # Poisson reconstruction with higher depth for better quality
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=depth, linear_fit=False
        )
        logger.info(f"Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Remove low-density vertices (artifacts from sparse regions)
        # Use a small default min_density to avoid removing too much geometry
        if min_density <= 0.0:
            min_density = 0.02  # Remove bottom 2% density vertices (gentle noise reduction)
        
        densities_array = np.asarray(densities)
        threshold = np.quantile(densities_array, min_density)
        vertices_to_keep = densities_array > threshold
        
        # Safety check: ensure density filtering doesn't return None
        filtered_mesh = mesh.remove_vertices_by_mask(vertices_to_keep)
        if filtered_mesh is not None and len(filtered_mesh.vertices) > 0:
            mesh = filtered_mesh
            logger.info(f"After density filtering (threshold={threshold:.4f}): {len(mesh.vertices)} vertices")
        else:
            logger.warning(f"Density filtering would remove all vertices, keeping original mesh")
            logger.info(f"Original mesh: {len(mesh.vertices)} vertices")
        
        return mesh
    
    def generate_mesh_ball_pivoting(self,
                                   point_cloud: o3d.geometry.PointCloud,
                                   radii: Optional[List[float]] = None) -> o3d.geometry.TriangleMesh:
        """
        Generate mesh using Ball Pivoting Algorithm.
        
        Args:
            point_cloud: Input point cloud
            radii: List of radii for BPA
        
        Returns:
            Generated TriangleMesh
        """
        logger.info("Generating mesh using Ball Pivoting Algorithm")
        
        # Estimate normals
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Default radii based on point cloud scale
        if radii is None:
            radius = estimate_radius(point_cloud)
            radii = [radius, radius * 2]
        
        logger.info(f"BPA radii: {radii}")
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud,
            o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        
        logger.info(f"Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    
    def clean_mesh(self, mesh: o3d.geometry.TriangleMesh,
                  remove_duplicates: bool = True,
                  remove_degenerate: bool = True,
                  remove_unreferenced: bool = True,
                  remove_isolated_small_components: bool = True,
                  min_component_size: int = 50) -> o3d.geometry.TriangleMesh:
        """
        Clean mesh by removing artifacts.
        
        Args:
            mesh: Input mesh
            remove_duplicates: Remove duplicate vertices
            remove_degenerate: Remove degenerate triangles
            remove_unreferenced: Remove unreferenced vertices
            remove_isolated_small_components: Remove small isolated components
            min_component_size: Minimum size for component to keep
        
        Returns:
            Cleaned mesh
        """
        logger.info("Cleaning mesh")
        
        if remove_duplicates:
            mesh.remove_duplicated_vertices()
            logger.info("Removed duplicate vertices")
        
        if remove_degenerate:
            mesh.remove_degenerate_triangles()
            logger.info("Removed degenerate triangles")
        
        if remove_unreferenced:
            mesh.remove_unreferenced_vertices()
            logger.info("Removed unreferenced vertices")
        
        if remove_isolated_small_components:
            triangle_clusters, cluster_n_triangles, cluster_area = \
                mesh.cluster_connected_triangles()
            
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_component_size
            mesh.remove_triangles_by_mask(triangles_to_remove)
            logger.info(f"Removed small components (< {min_component_size} triangles)")
        
        logger.info(f"Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    
    def export_mesh_obj(self, mesh: o3d.geometry.TriangleMesh, 
                       output_path: str) -> bool:
        """Export mesh as OBJ file."""
        try:
            # Validate mesh has data
            if mesh is None or len(mesh.vertices) == 0:
                logger.error(f"Cannot export empty mesh (vertices: {len(mesh.vertices) if mesh else 'None'})")
                return False
            
            logger.info(f"Exporting mesh to {output_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
            success = o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=True)
            if success:
                logger.info("OBJ export successful")
            else:
                logger.error("OBJ export returned False from Open3D")
            return success
        except Exception as e:
            logger.error(f"Failed to export OBJ: {e}")
            return False
    
    def export_mesh_glb(self, mesh: o3d.geometry.TriangleMesh,
                       output_path: str) -> bool:
        """Export mesh as GLB file with proper formatting."""
        try:
            # Validate mesh has data
            if mesh is None or len(mesh.vertices) == 0:
                logger.error(f"Cannot export empty mesh (vertices: {len(mesh.vertices) if mesh else 'None'})")
                return False
            
            logger.info(f"Exporting mesh to {output_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
            
            # Ensure mesh has normals for proper rendering
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # Enable vertex colors if available
            if not mesh.has_vertex_colors() and mesh.has_triangles():
                # Default grey color for better visualization
                mesh.paint_uniform_color([0.8, 0.8, 0.8])
            
            success = o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False, compressed=True)
            if success:
                logger.info("GLB export successful")
            else:
                logger.error("GLB export returned False from Open3D")
            return success
        except Exception as e:
            logger.error(f"Failed to export GLB: {e}")
            return False
    
    def process_image(self, rgb_image: np.ndarray, depth_map: np.ndarray,
                     output_dir: Path, method: str = 'poisson',
                     clean: bool = True, denoise: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: depth → point cloud → denoise → mesh → REFINE → export.
        
        Args:
            rgb_image: RGB image (H x W x 3), values 0-255
            depth_map: Depth map (H x W)
            output_dir: Directory to save outputs
            method: 'poisson' or 'bpa'
            clean: Whether to clean the mesh
            denoise: Whether to denoise the point cloud
        
        Returns:
            Dict with 'obj_path', 'glb_path', 'preview_path', 'metadata'
        """
        logger.info("=" * 60)
        logger.info("Starting image processing pipeline with denoising")
        logger.info("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # STEP 1: VALIDATE DEPTH (critical check)
        logger.info("=" * 60)
        logger.info("DEPTH MAP VALIDATION (Single Image)")
        logger.info("=" * 60)
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        valid_pixels = np.count_nonzero(depth_map > 0)
        total_pixels = depth_map.size
        valid_percent = 100 * valid_pixels / total_pixels
        
        logger.info(f"Depth range: min={min_depth:.6f}, max={max_depth:.6f}")
        logger.info(f"Valid pixels: {valid_pixels}/{total_pixels} ({valid_percent:.1f}%)")
        
        if max_depth == 0:
            logger.error("⚠️  CRITICAL: Depth map is all zeros!")
        elif valid_percent < 5:
            logger.error("⚠️  CRITICAL: Less than 5% valid depth pixels!")
        elif max_depth > 1000:
            logger.warning(f"⚠️  Depth values very large (max={max_depth}) - may need scaling")
        
        logger.info("=" * 60)
        
        # Step 1: Depth → Point Cloud
        pcd = self.depth_to_point_cloud(depth_map, rgb_image)
        logger.info(f"Step 1 - Point cloud created: {len(pcd.points)} points")
        
        if len(pcd.points) == 0:
            logger.error("FATAL: Point cloud is empty after depth conversion")
            # Return empty mesh with error metadata
            empty_mesh = o3d.geometry.TriangleMesh()
            return {
                'obj_path': str(output_dir / 'scene.obj'),
                'glb_path': str(output_dir / 'scene.glb'),
                'preview_path': None,
                'metadata': {
                    'error': 'Point cloud generation failed - empty cloud',
                    'vertex_count': 0,
                    'triangle_count': 0
                }
            }
        
        # Step 2: Denoise point cloud
        if denoise:
            logger.info("Denoising point cloud...")
            try:
                pcd = self.denoise_point_cloud(pcd)
                logger.info(f"Step 2 - After denoising: {len(pcd.points)} points")
            except Exception as e:
                logger.warning(f"Denoising failed, continuing without denoise: {e}")
        
        # Step 3: Smart downsample
        logger.info("Smart downsampling...")
        pcd = self.smart_downsample(pcd)
        logger.info(f"Step 3 - After downsampling: {len(pcd.points)} points")
        
        if len(pcd.points) == 0:
            logger.error("FATAL: Point cloud became empty after downsampling")
            empty_mesh = o3d.geometry.TriangleMesh()
            return {
                'obj_path': str(output_dir / 'scene.obj'),
                'glb_path': str(output_dir / 'scene.glb'),
                'preview_path': None,
                'metadata': {
                    'error': 'Point cloud became empty after downsampling',
                    'vertex_count': 0,
                    'triangle_count': 0
                }
            }
        
        # Step 4: Generate Mesh
        if method == 'poisson':
            # Use Poisson reconstruction for clean meshes
            mesh = self.generate_mesh_poisson(pcd, depth=11)
            logger.info(f"Step 4 - Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        elif method == 'bpa':
            mesh = self.generate_mesh_ball_pivoting(pcd)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Step 5: Premium Mesh Refinement (removes noise, smooth fabric, etc.)
        try:
            from .mesh_refinement import MeshRefinement
            logger.info("Starting premium mesh refinement (6-step pipeline)...")
            
            # Detect if mesh is very dense (needs aggressive refinement)
            vertex_count = len(np.asarray(mesh.vertices))
            is_dense = vertex_count > 1000000
            
            if is_dense:
                logger.info(f"High vertex count detected ({vertex_count:,d}) - using aggressive refinement")
            
            mesh = MeshRefinement.complete_refinement_pipeline(mesh, aggressive=is_dense)
            logger.info("✓ Premium mesh refinement completed successfully")
        except Exception as e:
            logger.error(f"Mesh refinement encountered error: {str(e)}")
            logger.warning(f"Continuing without refinement (mesh may appear noisy)")
        
        # Step 6: Clean and Optimize Mesh
        if clean:
            mesh = self.clean_mesh(mesh, min_component_size=20)
            mesh = self.optimize_mesh(mesh, smooth_iterations=2)
        
        # Step 7: Export
        obj_path = output_dir / "scene.obj"
        glb_path = output_dir / "scene.glb"
        
        self.export_mesh_obj(mesh, str(obj_path))
        self.export_mesh_glb(mesh, str(glb_path))
        
        # Step 8: Save depth preview
        depth_preview_path = self._save_depth_preview(depth_map, output_dir)
        
        # Step 9: Metadata
        metadata = {
            'method': method,
            'point_count': len(pcd.points),
            'vertex_count': len(mesh.vertices),
            'triangle_count': len(mesh.triangles),
            'bounds': {
                'min': np.asarray(mesh.get_min_bound()).tolist(),
                'max': np.asarray(mesh.get_max_bound()).tolist(),
            }
        }
        
        metadata_path = output_dir / "mesh_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("Image processing complete")
        logger.info("=" * 60)
        
        return {
            'obj_path': str(obj_path),
            'glb_path': str(glb_path),
            'depth_preview_path': str(depth_preview_path),
            'metadata_path': str(metadata_path),
            'metadata': metadata
        }
    
    def _save_depth_preview(self, depth_map: np.ndarray, output_dir: Path) -> Path:
        """Save depth map as preview image."""
        # Normalize to 0-255
        depth_normalized = ((depth_map - depth_map.min()) / 
                           (depth_map.max() - depth_map.min() + 1e-6) * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        preview_path = output_dir / "depth_preview.png"
        Image.fromarray(depth_colored).save(preview_path)
        
        logger.info(f"Depth preview saved to {preview_path}")
        return preview_path

    def smooth_depth_map(self, depth_map: np.ndarray, 
                         kernel_size: int = 7,
                         sigma_spatial: float = 3.0,
                         sigma_intensity: float = 0.15) -> np.ndarray:
        """
        Smooth depth map while preserving edges using aggressive bilateral filtering.
        Removes noise without blurring object boundaries - optimized for professional quality.
        
        Args:
            depth_map: Input depth map (H x W)
            kernel_size: Size of the bilateral filter kernel (increased for better smoothing)
            sigma_spatial: Standard deviation for spatial Gaussian (increased for more smoothing)
            sigma_intensity: Standard deviation for depth difference Gaussian
        
        Returns:
            Smoothed depth map
        """
        logger.info(f"Smoothing depth map with ENHANCED quality (kernel={kernel_size}, sigma_spatial={sigma_spatial})")
        
        if depth_map.max() == 0:
            logger.warning("Depth map is empty, returning as-is")
            return depth_map
        
        try:
            # Apply multiple bilateral filtering passes for better quality
            smoothed = depth_map.astype(np.float32)
            for pass_num in range(2):
                smoothed = cv2.bilateralFilter(
                    smoothed,
                    d=kernel_size,
                    sigmaColor=sigma_intensity,
                    sigmaSpace=sigma_spatial
                )
            
            logger.info(f"Depth map smoothed successfully (2 enhanced passes)")
            return smoothed
        except Exception as e:
            logger.warning(f"Bilateral filtering failed: {e}, trying stronger Gaussian blur")
            # Fallback to stronger Gaussian blur
            kernel = (kernel_size, kernel_size)
            smoothed = depth_map.astype(np.float32)
            for _ in range(3):
                smoothed = cv2.GaussianBlur(smoothed, kernel, sigma_spatial)
            return smoothed
    
    def denoise_point_cloud(self, point_cloud: o3d.geometry.PointCloud,
                           nb_neighbors: int = 20,
                           std_ratio: float = 3.0) -> o3d.geometry.PointCloud:
        """
        Remove noise from point cloud using statistical outlier removal.
        Uses distance-based filtering if statistical method unavailable.
        
        Args:
            point_cloud: Input point cloud
            nb_neighbors: Number of neighbors to consider for each point (more = less aggressive)
            std_ratio: Standard deviation ratio for outlier threshold (higher = less aggressive)
        
        Returns:
            Denoised point cloud
        """
        logger.info(f"Denoising point cloud ({len(point_cloud.points)} points, nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
        
        try:
            # Try statistical outlier removal (Open3D >= 0.13)
            # Higher std_ratio = more conservative (keeps more points)
            cl, ind = point_cloud.remove_statistical_outliers(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            denoised = point_cloud.select_by_index(ind)
            removed = len(point_cloud.points) - len(denoised.points)
            logger.info(f"Outliers removed (statistical): {removed} points ({100*removed/len(point_cloud.points):.1f}%)")
            
        except (AttributeError, RuntimeError) as e:
            # Fallback: Distance-based filtering
            logger.info(f"Statistical removal unavailable, using distance-based filtering: {e}")
            
            points = np.asarray(point_cloud.points)
            if len(points) == 0:
                logger.warning("Point cloud is empty")
                return point_cloud
            
            # Compute distances to nearest neighbors
            distances = point_cloud.compute_nearest_neighbor_distance()
            distances = np.asarray(distances)
            
            # Filter based on distance threshold
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = mean_dist + std_ratio * std_dist
            
            valid_indices = np.where(distances < threshold)[0]
            denoised = point_cloud.select_by_index(valid_indices)
            
            logger.info(f"Outliers removed (distance-based): {len(point_cloud.points) - len(denoised.points)} points (threshold: {threshold:.6f})")
        
        logger.info(f"Denoised point cloud: {len(denoised.points)} points")
        
        return denoised
    
    def smart_downsample(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Intelligently downsample point cloud based on density.
        Preserves detail in high-density regions while reducing noise in sparse areas.
        
        Args:
            point_cloud: Input point cloud
        
        Returns:
            Downsampled point cloud
        """
        num_points = len(point_cloud.points)
        logger.info(f"Smart downsampling point cloud ({num_points} points)")
        
        # Analyze point cloud density
        distances = point_cloud.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        
        # Adaptive voxel size based on point density - LESS AGGRESSIVE
        if num_points > 3000000:
            voxel_size = avg_distance * 2.5  # Was 4.0
        elif num_points > 1000000:
            voxel_size = avg_distance * 1.5  # Was 2.0
        elif num_points > 500000:
            voxel_size = avg_distance * 1.2  # Was 1.5
        else:
            voxel_size = avg_distance * 0.8  # Was 1.0
        
        logger.info(f"Computed voxel size: {voxel_size:.6f} (avg neighbor distance: {avg_distance:.6f})")
        
        # Voxel downsample
        downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        
        logger.info(f"Downsampled to {len(downsampled.points)} points (reduction: {100 * (1 - len(downsampled.points)/num_points):.1f}%)")
        
        return downsampled
    
    def load_mesh(self, file_path: str) -> o3d.geometry.TriangleMesh:
        """
        Load a 3D mesh from file (.glb, .gltf, .obj, etc).
        
        Args:
            file_path: Path to the mesh file
        
        Returns:
            Open3D TriangleMesh object
        """
        logger.info(f"Loading mesh from {file_path}")
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return mesh
        except Exception as e:
            logger.error(f"Failed to load mesh: {e}")
            raise
    
    def optimize_mesh(self, mesh: o3d.geometry.TriangleMesh,
                     smooth_iterations: int = 3,
                     target_reduction: float = 0.0) -> o3d.geometry.TriangleMesh:
        """
        Optimize mesh for better quality and rendering.
        
        Args:
            mesh: Input mesh
            smooth_iterations: Number of Laplacian smoothing iterations
            target_reduction: Target polygon reduction ratio (0 = no reduction)
        
        Returns:
            Optimized mesh
        """
        logger.info("Optimizing mesh quality")
        
        # Ensure normals are computed for proper rendering
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            logger.info("Computed vertex normals")
        
        # Apply smoothing to reduce noise
        for i in range(smooth_iterations):
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=1, lambda_filter=0.5)
        logger.info(f"Applied {smooth_iterations} smoothing iterations")
        
        # Simplify mesh while preserving quality if target_reduction > 0
        if target_reduction > 0:
            original_triangles = len(mesh.triangles)
            mesh = mesh.simplify_quadric_decimation(target_count=int(original_triangles * (1 - target_reduction)))
            logger.info(f"Simplified from {original_triangles} to {len(mesh.triangles)} triangles")
        
        return mesh
    
    def validate_glb(self, file_path: str) -> bool:
        """
        Validate GLB file integrity.
        
        Args:
            file_path: Path to GLB file
        
        Returns:
            True if valid, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"GLB file not found: {file_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < 20:  # Minimum GLB file size
                logger.warning(f"GLB file too small: {file_size} bytes")
                return False
            
            # Try to load and validate
            mesh = o3d.io.read_triangle_mesh(file_path)
            
            if len(mesh.vertices) == 0:
                logger.warning("GLB file has no vertices")
                return False
            
            if len(mesh.triangles) == 0:
                logger.warning("GLB file has no triangles")
                return False
            
            logger.info(f"GLB validation passed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return True
            
        except Exception as e:
            logger.error(f"GLB validation failed: {e}")
            return False
    
    def repair_glb(self, file_path: str) -> bool:
        """
        Attempt to repair a corrupted GLB file.
        
        Args:
            file_path: Path to GLB file
        
        Returns:
            True if repair successful, False otherwise
        """
        try:
            logger.info(f"Attempting to repair GLB file: {file_path}")
            
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(file_path)
            
            # Remove degenerate and duplicate elements
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            
            # Ensure normals are computed
            mesh.compute_vertex_normals()
            
            # Re-export with proper formatting
            success = self.export_mesh_glb(mesh, file_path)
            
            if success:
                logger.info("GLB repair successful")
            
            return success
            
        except Exception as e:
            logger.error(f"GLB repair failed: {e}")
            return False


def estimate_radius(point_cloud: o3d.geometry.PointCloud) -> float:
    """Estimate a good radius for ball pivoting based on point cloud density."""
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    return avg_dist * 3  # Use 3x average distance as radius
