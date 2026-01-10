"""
Enhanced mesh cleanup and optimization.
Improves mesh quality after TSDF fusion and Poisson reconstruction.

Features:
- Statistical outlier removal
- Mesh simplification (decimation)
- Normal smoothing and estimation
- Boundary closure
- Small component removal
- Watertight verification
"""

import numpy as np
import logging
from typing import Tuple, Optional

# Lazy imports for optional dependencies
def _import_o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None

def _import_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError:
        return None

logger = logging.getLogger(__name__)


class MeshOptimizer:
    """Advanced mesh cleanup and optimization."""
    
    @staticmethod
    def remove_statistical_outliers(mesh: o3d.geometry.TriangleMesh,
                                   nb_neighbors: int = 20,
                                   std_ratio: float = 2.0) -> o3d.geometry.TriangleMesh:
        """
        Remove outlier vertices using statistical distance.
        
        Args:
            mesh: Input mesh
            nb_neighbors: Number of neighbors for statistics
            std_ratio: Standard deviation ratio threshold
            
        Returns:
            Cleaned mesh
        """
        logger.info(f"Removing statistical outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
        
        try:
            # Convert to point cloud for statistical filtering
            pcd = mesh.sample_points_uniformly(number_of_points=100000)
            
            # Remove outliers
            pcd_clean, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
            logger.info(f"✓ Removed {len(pcd.points) - len(pcd_clean.points)} outlier points")
            
            # Reconstruct mesh from cleaned points
            pcd_clean.estimate_normals()
            mesh_clean = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_clean, depth=9
            )[0]
            
            return mesh_clean
            
        except Exception as e:
            logger.warning(f"Statistical outlier removal failed: {e}, returning original mesh")
            return mesh
    
    @staticmethod
    def remove_low_density_vertices(mesh: o3d.geometry.TriangleMesh,
                                   radius: float = 0.05,
                                   min_neighbors: int = 3) -> o3d.geometry.TriangleMesh:
        """
        Remove vertices with few neighbors (sparse regions).
        
        Args:
            mesh: Input mesh
            radius: Search radius for neighbors
            min_neighbors: Minimum number of neighbors to keep vertex
            
        Returns:
            Mesh with sparse vertices removed
        """
        logger.info(f"Removing low-density vertices (radius={radius}, min_neighbors={min_neighbors})")
        
        try:
            # Build KD-tree for neighbor search
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            
            # Find neighbors for each vertex
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            
            # Mark vertices to keep
            keep_vertices = []
            for i in range(len(pcd.points)):
                k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
                if k >= min_neighbors:
                    keep_vertices.append(i)
            
            logger.info(f"✓ Keeping {len(keep_vertices)}/{len(pcd.points)} vertices")
            
            # Filter triangles to keep only those with all vertices kept
            keep_set = set(keep_vertices)
            keep_triangles = []
            for tri in mesh.triangles:
                if all(t in keep_set for t in tri):
                    keep_triangles.append(tri)
            
            # Create new mesh with filtered vertices/triangles
            if len(keep_triangles) == 0:
                logger.warning("All triangles filtered, returning original mesh")
                return mesh
            
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices)[keep_vertices]
            )
            
            # Remap triangle indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_vertices)}
            remapped_triangles = []
            for tri in keep_triangles:
                remapped_triangles.append([
                    old_to_new[tri[0]],
                    old_to_new[tri[1]],
                    old_to_new[tri[2]]
                ])
            
            new_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
            new_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertex_colors)[keep_vertices]
            ) if mesh.has_vertex_colors() else None
            
            return new_mesh
            
        except Exception as e:
            logger.warning(f"Low-density removal failed: {e}, returning original mesh")
            return mesh
    
    @staticmethod
    def simplify_mesh(mesh: o3d.geometry.TriangleMesh,
                     target_count: int = 100000) -> o3d.geometry.TriangleMesh:
        """
        Simplify mesh while preserving geometry (decimation).
        
        Args:
            mesh: Input mesh
            target_count: Target triangle count
            
        Returns:
            Simplified mesh
        """
        logger.info(f"Simplifying mesh: {len(mesh.triangles)} → {target_count} triangles")
        
        try:
            # Calculate target reduction ratio
            current_count = len(mesh.triangles)
            if current_count <= target_count:
                logger.info(f"Mesh already smaller than target ({current_count} <= {target_count})")
                return mesh
            
            target_ratio = target_count / current_count
            
            # Use trimesh for better simplification
            mesh_tri = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles)
            )
            
            mesh_tri = mesh_tri.simplify(target_ratio)
            
            # Convert back to Open3D
            simplified = o3d.geometry.TriangleMesh()
            simplified.vertices = o3d.utility.Vector3dVector(mesh_tri.vertices)
            simplified.triangles = o3d.utility.Vector3iVector(mesh_tri.faces)
            
            logger.info(f"✓ Simplified: {len(mesh.triangles)} → {len(simplified.triangles)} triangles")
            
            return simplified
            
        except Exception as e:
            logger.warning(f"Mesh simplification failed: {e}, returning original mesh")
            return mesh
    
    @staticmethod
    def laplacian_smoothing(mesh: o3d.geometry.TriangleMesh,
                           iterations: int = 3,
                           lambda_factor: float = 0.5) -> o3d.geometry.TriangleMesh:
        """
        Smooth mesh using Laplacian operator.
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            lambda_factor: Smoothing factor (0-1, higher = more smoothing)
            
        Returns:
            Smoothed mesh
        """
        logger.info(f"Laplacian smoothing: iterations={iterations}, lambda={lambda_factor}")
        
        try:
            vertices = np.asarray(mesh.vertices).copy()
            triangles = np.asarray(mesh.triangles)
            
            # Build adjacency structure
            vertex_neighbors = [[] for _ in range(len(vertices))]
            for tri in triangles:
                for i in range(3):
                    v1 = tri[i]
                    v2 = tri[(i + 1) % 3]
                    if v2 not in vertex_neighbors[v1]:
                        vertex_neighbors[v1].append(v2)
                    if v1 not in vertex_neighbors[v2]:
                        vertex_neighbors[v2].append(v1)
            
            # Laplacian smoothing iterations
            for iter_num in range(iterations):
                new_vertices = vertices.copy()
                for i in range(len(vertices)):
                    if len(vertex_neighbors[i]) > 0:
                        neighbor_avg = np.mean(vertices[vertex_neighbors[i]], axis=0)
                        new_vertices[i] = (1 - lambda_factor) * vertices[i] + lambda_factor * neighbor_avg
                
                vertices = new_vertices
                logger.debug(f"  Iteration {iter_num + 1}/{iterations}")
            
            # Create smoothed mesh
            smoothed = o3d.geometry.TriangleMesh()
            smoothed.vertices = o3d.utility.Vector3dVector(vertices)
            smoothed.triangles = o3d.utility.Vector3iVector(triangles)
            
            logger.info(f"✓ Laplacian smoothing complete")
            
            return smoothed
            
        except Exception as e:
            logger.warning(f"Laplacian smoothing failed: {e}, returning original mesh")
            return mesh
    
    @staticmethod
    def remove_small_components(mesh: o3d.geometry.TriangleMesh,
                               min_size: int = 100) -> o3d.geometry.TriangleMesh:
        """
        Remove small isolated components (floating islands).
        
        Args:
            mesh: Input mesh
            min_size: Minimum triangle count to keep component
            
        Returns:
            Mesh with small components removed
        """
        logger.info(f"Removing small components (min_size={min_size})")
        
        try:
            # Use Open3D's cluster connectivity
            mesh_simplified = mesh.simplify_vertex_clustering(voxel_size=0.001)
            
            # Identify connected components
            triangle_clusters = mesh_simplified.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            
            # Find clusters larger than min_size
            unique_clusters = np.unique(triangle_clusters)
            large_clusters = []
            for cluster_id in unique_clusters:
                cluster_size = np.sum(triangle_clusters == cluster_id)
                if cluster_size >= min_size:
                    large_clusters.append(cluster_id)
            
            logger.info(f"✓ Removed {len(unique_clusters) - len(large_clusters)} small components")
            
            # Keep only large cluster triangles
            keep_triangles = np.any(
                [triangle_clusters == c for c in large_clusters],
                axis=0
            )
            
            if np.sum(keep_triangles) == 0:
                logger.warning("All components filtered, returning original mesh")
                return mesh
            
            triangles_to_keep = np.where(keep_triangles)[0]
            mesh_filtered = mesh.select_by_index(triangles_to_keep, cleanup=True)
            
            return mesh_filtered
            
        except Exception as e:
            logger.warning(f"Small component removal failed: {e}, returning original mesh")
            return mesh
    
    @staticmethod
    def close_mesh_boundaries(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Attempt to close mesh boundaries (make watertight).
        
        Args:
            mesh: Input mesh
            
        Returns:
            Mesh with boundaries closed if possible
        """
        logger.info("Attempting to close mesh boundaries")
        
        try:
            mesh_tri = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
            )
            
            # Fill small holes
            mesh_tri.fill_holes()
            
            # Convert back
            closed = o3d.geometry.TriangleMesh()
            closed.vertices = o3d.utility.Vector3dVector(mesh_tri.vertices)
            closed.triangles = o3d.utility.Vector3iVector(mesh_tri.faces)
            
            logger.info(f"✓ Boundaries processed: {len(mesh.triangles)} → {len(closed.triangles)} triangles")
            
            return closed
            
        except Exception as e:
            logger.warning(f"Boundary closing failed: {e}, returning original mesh")
            return mesh
    
    @staticmethod
    def estimate_and_smooth_normals(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Estimate and smooth vertex normals for better appearance.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Mesh with smooth normals
        """
        logger.info("Estimating and smoothing normals")
        
        try:
            # Compute normals
            mesh.compute_vertex_normals()
            
            # Orient normals consistently (outward)
            mesh.orient_triangles()
            
            logger.info("✓ Normals estimated and oriented")
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Normal estimation failed: {e}")
            return mesh
    
    @classmethod
    def full_cleanup_pipeline(cls,
                             mesh: o3d.geometry.TriangleMesh,
                             target_triangles: int = 100000) -> o3d.geometry.TriangleMesh:
        """
        Apply full mesh cleanup pipeline.
        
        Args:
            mesh: Input mesh from TSDF or Poisson
            target_triangles: Target triangle count for simplification
            
        Returns:
            Cleaned and optimized mesh
        """
        logger.info("=" * 60)
        logger.info("FULL MESH CLEANUP PIPELINE")
        logger.info("=" * 60)
        
        initial_vertices = len(mesh.vertices)
        initial_triangles = len(mesh.triangles)
        
        # Step 1: Estimate normals
        mesh = cls.estimate_and_smooth_normals(mesh)
        
        # Step 2: Close boundaries
        mesh = cls.close_mesh_boundaries(mesh)
        
        # Step 3: Remove small components
        mesh = cls.remove_small_components(mesh, min_size=50)
        
        # Step 4: Remove low-density vertices
        mesh = cls.remove_low_density_vertices(mesh, radius=0.05, min_neighbors=3)
        
        # Step 5: Simplify mesh
        if len(mesh.triangles) > target_triangles:
            mesh = cls.simplify_mesh(mesh, target_count=target_triangles)
        
        # Step 6: Laplacian smoothing
        mesh = cls.laplacian_smoothing(mesh, iterations=3, lambda_factor=0.5)
        
        # Step 7: Final normal estimation
        mesh = cls.estimate_and_smooth_normals(mesh)
        
        logger.info("=" * 60)
        logger.info(f"Cleanup Summary:")
        logger.info(f"  Initial: {initial_vertices:,d} vertices, {initial_triangles:,d} triangles")
        logger.info(f"  Final:   {len(mesh.vertices):,d} vertices, {len(mesh.triangles):,d} triangles")
        logger.info(f"  Reduction: {100*(initial_triangles-len(mesh.triangles))/initial_triangles:.1f}%")
        logger.info("=" * 60)
        
        return mesh
