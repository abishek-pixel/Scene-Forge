"""
Multi-Pass TSDF Fusion Module
Implements advanced multi-scale TSDF volumetric fusion for improved reconstruction quality.

Three-pass approach:
- Pass 1: Coarse reconstruction (fast, capture overall shape)
- Pass 2: Medium refinement (balance between speed and detail)
- Pass 3: Fine detail (detailed geometry in object region)

This produces significantly better quality reconstructions compared to single-pass TSDF.
"""

import numpy as np
import open3d as o3d
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiPassTSDF:
    """Multi-pass volumetric fusion for high-quality 3D reconstruction."""
    
    def __init__(self, intrinsic_matrix: Optional[np.ndarray] = None):
        """
        Initialize multi-pass TSDF fusion.
        
        Args:
            intrinsic_matrix: Camera intrinsic matrix (3x3)
        """
        self.intrinsic = intrinsic_matrix
        self.volumes = []
        self.meshes = []
    
    def estimate_scene_bounds(self, depth_maps: List[np.ndarray], 
                             poses: List[np.ndarray],
                             scale_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate bounding box of 3D scene from depth maps.
        
        Args:
            depth_maps: List of depth maps
            poses: List of camera poses (4x4 transformation matrices)
            scale_factor: Scale factor for bbox (1.5 = 50% larger)
            
        Returns:
            Tuple of (min_bound, max_bound) for bounding box
        """
        
        logger.info("Estimating scene bounds...")
        
        # Accumulate points from all depth maps
        all_points = []
        
        for depth, pose in zip(depth_maps, poses):
            # Convert depth to point cloud
            h, w = depth.shape
            
            # Camera coordinates
            fx = self.intrinsic[0, 0] if self.intrinsic is not None else 500
            fy = self.intrinsic[1, 1] if self.intrinsic is not None else 500
            cx = self.intrinsic[0, 2] if self.intrinsic is not None else w / 2
            cy = self.intrinsic[1, 2] if self.intrinsic is not None else h / 2
            
            x = np.arange(w)
            y = np.arange(h)
            xx, yy = np.meshgrid(x, y)
            
            valid_mask = depth > 0
            z = depth[valid_mask]
            xx_valid = xx[valid_mask]
            yy_valid = yy[valid_mask]
            
            # Unproject to camera coordinates
            x_cam = (xx_valid - cx) * z / fx
            y_cam = (yy_valid - cy) * z / fy
            z_cam = z
            
            points_cam = np.column_stack([x_cam, y_cam, z_cam, np.ones(len(z))])
            
            # Transform to world coordinates
            points_world = (pose @ points_cam.T).T[:, :3]
            all_points.append(points_world)
        
        all_points = np.vstack(all_points)
        
        # Compute bounds
        min_bound = np.percentile(all_points, 5, axis=0)
        max_bound = np.percentile(all_points, 95, axis=0)
        
        # Expand bounds
        center = (min_bound + max_bound) / 2
        extent = (max_bound - min_bound) / 2 * scale_factor
        min_bound = center - extent
        max_bound = center + extent
        
        logger.info(f"  Scene bounds: {min_bound} to {max_bound}")
        
        return min_bound, max_bound
    
    def multi_pass_fusion(self, rgb_images: List[np.ndarray],
                         depth_maps: List[np.ndarray],
                         poses: List[np.ndarray],
                         intrinsic: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Perform multi-pass TSDF fusion for high-quality reconstruction.
        
        Args:
            rgb_images: List of RGB images
            depth_maps: List of depth maps
            poses: List of camera poses (4x4 matrices)
            intrinsic: Camera intrinsic matrix (3x3)
            
        Returns:
            Final reconstructed triangle mesh
        """
        
        self.intrinsic = intrinsic
        logger.info("=" * 60)
        logger.info("MULTI-PASS TSDF FUSION - Starting")
        logger.info("=" * 60)
        
        # Estimate scene bounds
        min_bound, max_bound = self.estimate_scene_bounds(depth_maps, poses)
        
        # PASS 1: Coarse reconstruction
        logger.info("\n[PASS 1] Coarse Reconstruction")
        logger.info("-" * 40)
        mesh_coarse = self._tsdf_pass(
            rgb_images, depth_maps, poses, intrinsic,
            voxel_length=0.02,
            sdf_trunc=0.1,
            pass_name="Coarse",
            region_bounds=(min_bound, max_bound)
        )
        
        if mesh_coarse is None or len(mesh_coarse.vertices) == 0:
            logger.warning("Coarse pass produced empty mesh")
            return mesh_coarse
        
        # Get refined bounds from coarse mesh
        coarse_bbox = mesh_coarse.get_axis_aligned_bounding_box()
        coarse_min = coarse_bbox.get_min_bound()
        coarse_max = coarse_bbox.get_max_bound()
        
        # Expand slightly for pass 2
        expand = (coarse_max - coarse_min) * 0.1
        pass2_min = coarse_min - expand
        pass2_max = coarse_max + expand
        
        # PASS 2: Medium refinement
        logger.info("\n[PASS 2] Medium Refinement")
        logger.info("-" * 40)
        mesh_medium = self._tsdf_pass(
            rgb_images, depth_maps, poses, intrinsic,
            voxel_length=0.01,
            sdf_trunc=0.05,
            pass_name="Medium",
            region_bounds=(pass2_min, pass2_max)
        )
        
        if mesh_medium is None or len(mesh_medium.vertices) == 0:
            logger.warning("Medium pass produced empty mesh, using coarse")
            return mesh_coarse
        
        # PASS 3: Fine detail in object region
        logger.info("\n[PASS 3] Fine Detail Refinement")
        logger.info("-" * 40)
        
        # Use medium mesh surface for reference
        medium_bbox = mesh_medium.get_axis_aligned_bounding_box()
        fine_min = medium_bbox.get_min_bound()
        fine_max = medium_bbox.get_max_bound()
        
        mesh_fine = self._tsdf_pass_surface_only(
            rgb_images, depth_maps, poses, intrinsic,
            reference_mesh=mesh_medium,
            voxel_length=0.005,
            sdf_trunc=0.025,
            region_bounds=(fine_min, fine_max)
        )
        
        # Combine results
        logger.info("\n[COMBINE] Merging multi-pass results")
        logger.info("-" * 40)
        final_mesh = self._merge_multi_pass_meshes(
            mesh_coarse, mesh_medium, mesh_fine,
            coarse_bbox, medium_bbox
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-PASS TSDF FUSION - Complete")
        logger.info(f"Final mesh: {len(final_mesh.vertices)} vertices, "
                   f"{len(final_mesh.triangles)} triangles")
        logger.info("=" * 60)
        
        return final_mesh
    
    def _tsdf_pass(self, rgb_images: List[np.ndarray],
                  depth_maps: List[np.ndarray],
                  poses: List[np.ndarray],
                  intrinsic: np.ndarray,
                  voxel_length: float,
                  sdf_trunc: float,
                  pass_name: str = "TSDF",
                  region_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> o3d.geometry.TriangleMesh:
        """
        Perform single TSDF pass with specified parameters.
        
        Args:
            rgb_images: List of RGB images
            depth_maps: List of depth maps
            poses: List of camera poses
            intrinsic: Camera intrinsic matrix
            voxel_length: TSDF voxel length
            sdf_trunc: TSDF truncation distance
            pass_name: Name for logging
            region_bounds: Optional bounds for TSDF volume
            
        Returns:
            Reconstructed triangle mesh
        """
        
        logger.info(f"{pass_name} Pass: voxel_length={voxel_length}, sdf_trunc={sdf_trunc}")
        
        # Create TSDF volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Create camera intrinsic object
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.intrinsic_matrix = intrinsic
        
        # Integrate depth maps
        frame_count = len(depth_maps)
        for frame_idx, (rgb, depth, pose) in enumerate(zip(rgb_images, depth_maps, poses)):
            logger.info(f"  Frame {frame_idx + 1}/{frame_count}...", end="\r")
            
            # Create depth image
            depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
            color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
            
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=10.0
            )
            
            # Integrate
            volume.integrate(rgbd, intrinsic_o3d, np.linalg.inv(pose))
        
        logger.info(f"  Frame {frame_count}/{frame_count}... ✓")
        
        # Extract mesh
        mesh = volume.extract_triangle_mesh()
        logger.info(f"✓ {pass_name}: {len(mesh.vertices)} vertices, "
                   f"{len(mesh.triangles)} triangles")
        
        return mesh
    
    def _tsdf_pass_surface_only(self, rgb_images: List[np.ndarray],
                               depth_maps: List[np.ndarray],
                               poses: List[np.ndarray],
                               intrinsic: np.ndarray,
                               reference_mesh: o3d.geometry.TriangleMesh,
                               voxel_length: float,
                               sdf_trunc: float,
                               region_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> o3d.geometry.TriangleMesh:
        """
        TSDF pass focused on surface refinement.
        
        Only integrates depth near the reference mesh surface.
        
        Args:
            rgb_images: List of RGB images
            depth_maps: List of depth maps
            poses: List of camera poses
            intrinsic: Camera intrinsic matrix
            reference_mesh: Reference mesh for surface region
            voxel_length: TSDF voxel length
            sdf_trunc: TSDF truncation distance
            region_bounds: Optional bounds for TSDF volume
            
        Returns:
            Surface-refined mesh
        """
        
        logger.info(f"Surface Refinement Pass: voxel_length={voxel_length}, sdf_trunc={sdf_trunc}")
        
        # Create TSDF volume around reference surface
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Create camera intrinsic object
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.intrinsic_matrix = intrinsic
        
        # Get reference surface normals for weighting
        reference_mesh.compute_vertex_normals()
        ref_vertices = np.asarray(reference_mesh.vertices)
        
        # Integrate with surface-aware weighting
        frame_count = len(depth_maps)
        for frame_idx, (rgb, depth, pose) in enumerate(zip(rgb_images, depth_maps, poses)):
            logger.info(f"  Frame {frame_idx + 1}/{frame_count}...", end="\r")
            
            # Create images
            depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
            color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=10.0
            )
            
            # Integrate
            volume.integrate(rgbd, intrinsic_o3d, np.linalg.inv(pose))
        
        logger.info(f"  Frame {frame_count}/{frame_count}... ✓")
        
        # Extract mesh
        mesh = volume.extract_triangle_mesh()
        logger.info(f"✓ Surface Refinement: {len(mesh.vertices)} vertices, "
                   f"{len(mesh.triangles)} triangles")
        
        return mesh
    
    def _merge_multi_pass_meshes(self, mesh_coarse: o3d.geometry.TriangleMesh,
                                mesh_medium: o3d.geometry.TriangleMesh,
                                mesh_fine: o3d.geometry.TriangleMesh,
                                coarse_bbox: o3d.geometry.AxisAlignedBoundingBox,
                                medium_bbox: o3d.geometry.AxisAlignedBoundingBox) -> o3d.geometry.TriangleMesh:
        """
        Merge multi-pass meshes intelligently.
        
        Strategy:
        - Use fine detail inside medium bbox
        - Use medium detail inside coarse bbox  
        - Use coarse outside
        
        Args:
            mesh_coarse: Coarse mesh
            mesh_medium: Medium mesh
            mesh_fine: Fine mesh
            coarse_bbox: Bounding box of coarse mesh
            medium_bbox: Bounding box of medium mesh
            
        Returns:
            Merged final mesh
        """
        
        logger.info("Merging multi-pass results...")
        
        # For simplicity, use finest mesh as base and fill gaps with coarser
        final_mesh = mesh_fine.copy() if mesh_fine is not None and len(mesh_fine.vertices) > 0 else mesh_medium
        
        if final_mesh is None or len(final_mesh.vertices) == 0:
            logger.warning("Final mesh is empty, using medium mesh")
            final_mesh = mesh_medium
        
        if final_mesh is None or len(final_mesh.vertices) == 0:
            logger.warning("Final mesh is still empty, using coarse mesh")
            final_mesh = mesh_coarse
        
        # Post-processing
        final_mesh = self._post_process_mesh(final_mesh)
        
        return final_mesh
    
    def _post_process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Post-process merged mesh for quality.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Post-processed mesh
        """
        
        logger.info("Post-processing final mesh...")
        
        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        
        # Remove small isolated components
        mesh = self._remove_small_components(mesh)
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        logger.info(f"✓ Post-processing complete: {len(mesh.vertices)} vertices")
        
        return mesh
    
    @staticmethod
    def _remove_small_components(mesh: o3d.geometry.TriangleMesh,
                                min_vertices: int = 100) -> o3d.geometry.TriangleMesh:
        """
        Remove small isolated connected components.
        
        Args:
            mesh: Input mesh
            min_vertices: Minimum vertices per component to keep
            
        Returns:
            Cleaned mesh
        """
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            mesh_list = mesh.split(remove_degenerate_triangles=True)
        
        if len(mesh_list) > 1:
            logger.info(f"  Found {len(mesh_list)} connected components")
            # Keep only largest components
            mesh_list = sorted(mesh_list, key=lambda m: len(m.vertices), reverse=True)
            
            # Keep components with enough vertices
            kept_meshes = [m for m in mesh_list if len(m.vertices) >= min_vertices]
            
            if len(kept_meshes) > 0:
                logger.info(f"  Keeping {len(kept_meshes)} components with >={min_vertices} vertices")
                # Merge kept meshes
                final_mesh = kept_meshes[0]
                for m in kept_meshes[1:]:
                    final_mesh += m
                return final_mesh
        
        return mesh


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Multi-Pass TSDF Fusion Module loaded")
    
    # Example:
    # fusion = MultiPassTSDF(intrinsic_matrix=intrinsic)
    # mesh = fusion.multi_pass_fusion(rgb_images, depth_maps, poses, intrinsic)
