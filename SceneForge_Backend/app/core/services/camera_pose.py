"""
Camera pose estimation and multi-view registration for 3D reconstruction.
Aligns multiple point clouds from different viewpoints into a unified coordinate frame.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

# Lazy import for optional dependency
def _import_o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None

logger = logging.getLogger(__name__)


class CameraPoseEstimator:
    """Estimate camera poses and align multiple point clouds."""
    
    @staticmethod
    def estimate_pose_from_depth_rotation(depth_map: np.ndarray,
                                         rotation_angle: float = 0.0,
                                         distance_estimate: float = 1.0) -> np.ndarray:
        """
        Estimate camera transformation matrix from depth map and assumed rotation.
        Useful for video sequences with rotating camera.
        
        Args:
            depth_map: Depth map (H x W)
            rotation_angle: Estimated rotation angle in degrees (for multi-view video)
            distance_estimate: Estimated distance scale
        
        Returns:
            4x4 transformation matrix
        """
        logger.info(f"Estimating pose: rotation={rotation_angle}°, distance={distance_estimate}")
        
        # Compute center of depth map (object centroid in image space)
        valid_mask = depth_map > 0
        if not np.any(valid_mask):
            logger.warning("No valid depth values, returning identity transform")
            return np.eye(4)
        
        # Get centroid in 3D space
        valid_depths = depth_map[valid_mask]
        mean_depth = np.mean(valid_depths)
        
        # Create transformation based on rotation
        angle_rad = np.radians(rotation_angle)
        
        # Rotation matrix around Y-axis (typical for rotating camera)
        Ry = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
        
        # Translation based on depth
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, mean_depth * distance_estimate],
            [0, 0, 0, 1]
        ])
        
        # Combined transformation
        transform = T @ Ry
        
        return transform
    
    @staticmethod
    def register_point_clouds_icp(source: o3d.geometry.PointCloud,
                                  target: o3d.geometry.PointCloud,
                                  max_iterations: int = 50,
                                  threshold: float = 0.02) -> Tuple[np.ndarray, float]:
        """
        Register source point cloud to target using ICP (Iterative Closest Point).
        Finds the best alignment between two point clouds.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            max_iterations: Maximum ICP iterations
            threshold: Distance threshold for point correspondences
        
        Returns:
            Tuple of (transformation_matrix, fitness_score)
        """
        logger.info(f"Registering point clouds (ICP, max_iter={max_iterations})")
        
        try:
            # Downsample for faster registration
            source_down = source.voxel_down_sample(voxel_size=0.01)
            target_down = target.voxel_down_sample(voxel_size=0.01)
            
            # Estimate normals if not present
            if not source_down.has_normals():
                source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            if not target_down.has_normals():
                target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            
            # ICP registration
            result = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                max_correspondence_distance=threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
            )
            
            logger.info(f"ICP fitness: {result.fitness:.6f}, RMSE: {result.inlier_rmse:.6f}")
            
            return result.transformation, result.fitness
            
        except Exception as e:
            logger.error(f"ICP registration failed: {e}")
            return np.eye(4), 0.0
    
    @staticmethod
    def align_multiple_clouds(point_clouds: List[o3d.geometry.PointCloud],
                             transforms: Optional[List[np.ndarray]] = None) -> o3d.geometry.PointCloud:
        """
        Align multiple point clouds using estimated or provided transformations.
        
        Args:
            point_clouds: List of point clouds to align
            transforms: Optional list of 4x4 transformation matrices
        
        Returns:
            Merged point cloud in unified coordinate frame
        """
        logger.info(f"Aligning {len(point_clouds)} point clouds")
        
        if len(point_clouds) == 0:
            logger.warning("No point clouds to align")
            return o3d.geometry.PointCloud()
        
        if len(point_clouds) == 1:
            logger.info("Only one point cloud, returning as-is")
            return point_clouds[0]
        
        # Use provided transforms or estimate them
        if transforms is None:
            logger.info("No transforms provided, using sequential ICP")
            transforms = [np.eye(4)]  # First cloud is reference frame
            
            for i in range(1, len(point_clouds)):
                logger.info(f"Registering cloud {i}/{len(point_clouds)-1}")
                transform, fitness = CameraPoseEstimator.register_point_clouds_icp(
                    point_clouds[i],
                    point_clouds[0]  # Register all to first cloud
                )
                transforms.append(transform)
        
        # Transform all clouds to unified frame
        aligned_clouds = []
        for i, (pcd, transform) in enumerate(zip(point_clouds, transforms)):
            pcd_aligned = pcd.transform(transform)
            aligned_clouds.append(pcd_aligned)
            logger.info(f"Transformed cloud {i}")
        
        # Merge all clouds
        merged = o3d.geometry.PointCloud()
        for pcd in aligned_clouds:
            merged += pcd
        
        logger.info(f"Merged point cloud: {len(merged.points)} points")
        
        return merged
    
    @staticmethod
    def estimate_video_poses(num_frames: int,
                            assumed_rotation_range: float = 180.0) -> List[np.ndarray]:
        """
        Estimate camera poses for video frames assuming smooth rotation.
        Useful when actual camera poses unknown.
        
        Args:
            num_frames: Number of frames in video
            assumed_rotation_range: Total rotation range in degrees
        
        Returns:
            List of transformation matrices for each frame
        """
        logger.info(f"Estimating poses for {num_frames} frames (rotation_range={assumed_rotation_range}°)")
        
        transforms = []
        for i in range(num_frames):
            # Smooth rotation progression
            rotation_angle = (i / max(1, num_frames - 1)) * assumed_rotation_range - (assumed_rotation_range / 2)
            transform = CameraPoseEstimator.estimate_pose_from_depth_rotation(
                np.ones((10, 10)),  # Dummy depth
                rotation_angle=rotation_angle,
                distance_estimate=1.0
            )
            transforms.append(transform)
            logger.info(f"Frame {i}: rotation={rotation_angle:.1f}°")
        
        return transforms
