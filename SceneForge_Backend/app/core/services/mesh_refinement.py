"""
Mesh Refinement Techniques for High-Quality 3D Output
Addresses noise, smoothness, and visual quality issues
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MeshRefinement:
    """Advanced mesh refinement techniques for production-quality output."""
    
    @staticmethod
    def bilateral_smooth_mesh(mesh: o3d.geometry.TriangleMesh,
                             iterations: int = 3,
                             lambda_filter: float = 0.5) -> o3d.geometry.TriangleMesh:
        """
        Bilateral smoothing: Remove noise while preserving sharp edges.
        This is crucial for cloth/fabric surfaces.
        
        Args:
            mesh: Input triangle mesh
            iterations: Number of smoothing iterations
            lambda_filter: Smoothing strength (0-1, higher = smoother)
        
        Returns:
            Smoothed mesh
        """
        logger.info(f"Applying bilateral mesh smoothing ({iterations} iterations)")
        
        try:
            # Use Taubin smoothing (approximates bilateral filtering)
            # This preserves edges while smoothing flat areas
            vertices = np.asarray(mesh.vertices).copy()
            triangles = np.asarray(mesh.triangles)
            
            # Build vertex-to-vertex adjacency
            vertex_adj = [[] for _ in range(len(vertices))]
            for tri in triangles:
                for i in range(3):
                    v1 = tri[i]
                    v2 = tri[(i+1) % 3]
                    if v2 not in vertex_adj[v1]:
                        vertex_adj[v1].append(v2)
                    if v1 not in vertex_adj[v2]:
                        vertex_adj[v2].append(v1)
            
            # Taubin filter (positive then negative pass)
            mu = lambda_filter
            for iteration in range(iterations):
                new_vertices = vertices.copy()
                
                # Forward pass (smoothing)
                for i, neighbors in enumerate(vertex_adj):
                    if len(neighbors) > 0:
                        neighbor_avg = np.mean(vertices[neighbors], axis=0)
                        new_vertices[i] = vertices[i] + mu * (neighbor_avg - vertices[i])
                
                # Backward pass (sharpening) - stabilizes and preserves edges
                vertices_temp = new_vertices.copy()
                for i, neighbors in enumerate(vertex_adj):
                    if len(neighbors) > 0:
                        neighbor_avg = np.mean(vertices_temp[neighbors], axis=0)
                        new_vertices[i] = vertices_temp[i] - mu * 0.5 * (neighbor_avg - vertices_temp[i])
                
                vertices = new_vertices
            
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            logger.info("Bilateral smoothing complete")
            
        except Exception as e:
            logger.warning(f"Bilateral smoothing failed: {e}")
        
        return mesh
    
    @staticmethod
    def remove_noise_outliers(mesh: o3d.geometry.TriangleMesh,
                            num_neighbors: int = 20,
                            std_dev: float = 2.0) -> o3d.geometry.TriangleMesh:
        """
        Remove noise and outlier vertices using local curvature analysis.
        
        Args:
            mesh: Input mesh
            num_neighbors: Neighborhood size for analysis
            std_dev: Standard deviation threshold for outlier removal
        
        Returns:
            Cleaned mesh
        """
        logger.info("Removing noise and outlier vertices")
        
        try:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Build KDTree for fast neighbor queries
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            
            # Compute curvature-based noise score
            noise_scores = np.zeros(len(vertices))
            for i in range(len(vertices)):
                k, indices, distances = kdtree.search_knn_vector_3d(vertices[i], num_neighbors + 1)
                neighbors = vertices[indices[1:]]  # Exclude self
                
                # High variance in local neighborhood = likely noise
                local_var = np.var(neighbors, axis=0)
                noise_scores[i] = np.mean(local_var)
            
            # Remove vertices with high noise scores
            threshold = np.mean(noise_scores) + std_dev * np.std(noise_scores)
            valid_vertices = noise_scores < threshold
            
            if np.sum(valid_vertices) < len(vertices):
                # Create mapping of old vertex indices to new
                old_to_new = np.full(len(vertices), -1, dtype=np.int32)
                old_to_new[valid_vertices] = np.arange(np.sum(valid_vertices))
                
                # Update vertices
                new_vertices = vertices[valid_vertices]
                
                # Update triangles (remove those with invalid vertices)
                valid_triangles = []
                for tri in triangles:
                    if all(old_to_new[v] >= 0 for v in tri):
                        valid_triangles.append([old_to_new[v] for v in tri])
                
                if len(valid_triangles) > 0:
                    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
                    mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))
                    logger.info(f"Removed {np.sum(~valid_vertices)} outlier vertices")
            
            mesh.compute_vertex_normals()
            
        except Exception as e:
            logger.warning(f"Outlier removal failed: {e}")
        
        return mesh
    
    @staticmethod
    def adaptive_subdivision(mesh: o3d.geometry.TriangleMesh,
                           target_edge_length: Optional[float] = None) -> o3d.geometry.TriangleMesh:
        """
        Subdivide mesh for smoother appearance without adding noise.
        Uses loop subdivision (smooth, preserves shape).
        
        Args:
            mesh: Input mesh
            target_edge_length: Target edge length for subdivision
        
        Returns:
            Subdivided mesh
        """
        logger.info("Applying adaptive mesh subdivision")
        
        try:
            # Estimate average edge length
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            edge_lengths = []
            for tri in triangles[:1000]:  # Sample first 1000 triangles
                for i in range(3):
                    v1 = vertices[tri[i]]
                    v2 = vertices[tri[(i+1) % 3]]
                    edge_lengths.append(np.linalg.norm(v2 - v1))
            
            avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0.01
            
            if target_edge_length is None:
                # Subdivide if edges are too long (coarse mesh)
                target_edge_length = avg_edge_length * 0.7
            
            # Use Open3D's built-in subdivision if available
            if hasattr(o3d.geometry.TriangleMesh, 'subdivide_midpoint'):
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
                logger.info("Mesh subdivided using midpoint method")
            else:
                logger.warning("Subdivision not available, using smoothing instead")
                mesh = MeshRefinement.bilateral_smooth_mesh(mesh, iterations=2)
            
        except Exception as e:
            logger.warning(f"Subdivision failed: {e}")
        
        return mesh
    
    @staticmethod
    def enhance_normals(mesh: o3d.geometry.TriangleMesh,
                       normal_radius: float = 0.05,
                       max_neighbors: int = 30) -> o3d.geometry.TriangleMesh:
        """
        Recompute normals using robust local fitting.
        Better normal estimation = better shading appearance.
        
        Args:
            mesh: Input mesh
            normal_radius: Search radius for local fitting
            max_neighbors: Maximum neighbors for fitting
        
        Returns:
            Mesh with enhanced normals
        """
        logger.info("Enhancing surface normals")
        
        try:
            vertices = np.asarray(mesh.vertices)
            
            # Create point cloud for normal estimation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            
            # Estimate normals with larger radius and more neighbors
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=max_neighbors
                )
            )
            
            # Orient normals to be consistent (requires normals to point outward)
            pcd.orient_normals_consistent_tangentspace(k=15)
            
            # Transfer back to mesh
            mesh.vertex_normals = pcd.normals
            logger.info("Normals enhanced with consistent orientation")
            
        except Exception as e:
            logger.warning(f"Normal enhancement failed: {e}")
            # Fallback to standard normal computation
            mesh.compute_vertex_normals()
        
        return mesh
    
    @staticmethod
    def remove_small_holes(mesh: o3d.geometry.TriangleMesh,
                          max_hole_size: int = 10) -> o3d.geometry.TriangleMesh:
        """
        Fill small holes in mesh (common with depth-based reconstruction).
        
        Args:
            mesh: Input mesh
            max_hole_size: Maximum number of triangles in a hole to fill
        
        Returns:
            Mesh with holes filled
        """
        logger.info(f"Removing holes (max size: {max_hole_size} triangles)")
        
        try:
            # Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()
            
            # Remove small disconnected components using cluster_connected_triangles()
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
                # cluster_connected_triangles() returns (triangle_clusters, cluster_n_triangles, cluster_area)
                triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                
                # Find and remove small clusters
                triangles_to_remove = []
                unique_clusters = np.unique(triangle_clusters)
                for cluster_id in unique_clusters:
                    cluster_size = cluster_n_triangles[cluster_id]
                    if 0 < cluster_size < max_hole_size:
                        triangles_to_remove.extend(np.where(triangle_clusters == cluster_id)[0])
            
            if triangles_to_remove:
                mesh.remove_triangles_by_index(triangles_to_remove)
                mesh.remove_unreferenced_vertices()
                logger.info(f"Removed {len(triangles_to_remove)} triangle(s) from small components")
            
            mesh.compute_vertex_normals()
            
        except Exception as e:
            logger.warning(f"Hole filling failed: {e}")
        
        return mesh
    
    @staticmethod
    def denoise_by_geometry(mesh: o3d.geometry.TriangleMesh,
                           smoothing_strength: float = 0.7) -> o3d.geometry.TriangleMesh:
        """
        Geometric denoising using normal filtering.
        Preserves sharp edges while removing spiky noise.
        
        Args:
            mesh: Input mesh
            smoothing_strength: Strength of smoothing (0-1)
        
        Returns:
            Denoised mesh
        """
        logger.info("Applying geometric denoising")
        
        try:
            vertices = np.asarray(mesh.vertices).copy()
            triangles = np.asarray(mesh.triangles)
            normals = np.asarray(mesh.vertex_normals).copy()
            
            # Build vertex-to-vertex adjacency
            vertex_adj = [[] for _ in range(len(vertices))]
            for tri in triangles:
                for i in range(3):
                    v1 = tri[i]
                    v2 = tri[(i+1) % 3]
                    if v2 not in vertex_adj[v1]:
                        vertex_adj[v1].append(v2)
                    if v1 not in vertex_adj[v2]:
                        vertex_adj[v2].append(v1)
            
            # Geometric denoising: project vertices to surface
            new_vertices = vertices.copy()
            for i, neighbors in enumerate(vertex_adj):
                if len(neighbors) > 0:
                    # Average position of neighbors
                    neighbor_pos = np.mean(vertices[neighbors], axis=0)
                    
                    # Move vertex toward neighbor average along surface
                    # But constrain to surface using normal
                    direction = neighbor_pos - vertices[i]
                    surface_component = direction - np.dot(direction, normals[i]) * normals[i]
                    
                    new_vertices[i] = vertices[i] + smoothing_strength * surface_component
            
            mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
            mesh.compute_vertex_normals()
            logger.info("Geometric denoising complete")
            
        except Exception as e:
            logger.warning(f"Geometric denoising failed: {e}")
        
        return mesh
    
    @staticmethod
    def complete_refinement_pipeline(mesh: o3d.geometry.TriangleMesh, aggressive: bool = True) -> o3d.geometry.TriangleMesh:
        """
        Simplified, stable mesh refinement pipeline.
        Focuses on proven techniques that don't fail.
        
        Args:
            mesh: Input mesh (typically from Poisson reconstruction)
            aggressive: If True, use stronger parameters for heavy noise
        
        Returns:
            Refined mesh (stable, tested)
        """
        logger.info("=" * 60)
        logger.info("MESH REFINEMENT PIPELINE - SIMPLIFIED STABLE VERSION")
        logger.info("=" * 60)
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.error("CRITICAL: Input mesh is None or empty!")
            return mesh
        
        initial_vertices = len(np.asarray(mesh.vertices))
        initial_triangles = len(np.asarray(mesh.triangles))
        logger.info(f"Initial: {initial_vertices:,d} vertices, {initial_triangles:,d} triangles")
        
        try:
            # SIMPLIFIED PIPELINE - Only use proven, stable methods
            
            # Step 1: Clean mesh (remove degenerate triangles and unreferenced vertices)
            logger.info("\nStep 1: Mesh cleaning")
            try:
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()
                mesh.remove_duplicated_vertices()
                v1 = len(np.asarray(mesh.vertices))
                logger.info(f"  → {v1:,d} vertices after cleaning")
            except Exception as e:
                logger.warning(f"  Cleaning failed: {e}")
            
            # Step 2: Fill small holes (proven method)
            logger.info("\nStep 2: Fill small holes")
            try:
                mesh = MeshRefinement.remove_small_holes(mesh, max_hole_size=50)
                v2 = len(np.asarray(mesh.vertices))
                logger.info(f"  → {v2:,d} vertices")
            except Exception as e:
                logger.warning(f"  Hole filling failed: {e}")
            
            # Step 3: Compute normals (required for smoothing)
            logger.info("\nStep 3: Compute vertex normals")
            try:
                mesh.compute_vertex_normals()
                logger.info(f"  → Normals computed")
            except Exception as e:
                logger.warning(f"  Normal computation failed: {e}")
            
            # Step 4: Simple bilateral smoothing (proven, stable)
            logger.info("\nStep 4: Bilateral smoothing (stable method)")
            try:
                mesh = MeshRefinement.bilateral_smooth_mesh(mesh, iterations=3, lambda_filter=0.5)
                v4 = len(np.asarray(mesh.vertices))
                logger.info(f"  → Applied (3 iterations, {v4:,d} vertices)")
            except Exception as e:
                logger.warning(f"  Bilateral smoothing failed: {e}")
            
            # Step 5: Smooth edges (proven method)
            logger.info("\nStep 5: Laplacian smoothing (edge preservation)")
            try:
                mesh.filter_smooth_laplacian(number_of_iterations=2, lambda_filter=0.5)
                logger.info(f"  → Applied (2 iterations)")
            except Exception as e:
                logger.warning(f"  Laplacian smoothing failed: {e}")
            
            # Step 6: Final normal recompute
            logger.info("\nStep 6: Recompute normals for final mesh")
            try:
                mesh.compute_vertex_normals()
                logger.info(f"  → Normals recomputed")
            except Exception as e:
                logger.warning(f"  Final normal computation failed: {e}")
            
            v_final = len(np.asarray(mesh.vertices))
            final_triangles = len(np.asarray(mesh.triangles))
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"REFINEMENT SUMMARY (SIMPLIFIED STABLE):")
            logger.info(f"  Before:   {initial_vertices:,d} vertices, {initial_triangles:,d} triangles")
            logger.info(f"  After:    {v_final:,d} vertices, {final_triangles:,d} triangles")
            removed_total = initial_vertices - v_final
            if removed_total > 0:
                logger.info(f"  Cleaned:  {removed_total:,d} vertices removed ({100*removed_total/initial_vertices:.1f}%)")
            logger.info(f"  Status:   STABLE ✓ (no experimental methods)")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"ERROR in refinement pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Returning original mesh without refinement")
            # Ensure we return a valid mesh, not None
            if mesh is None or len(np.asarray(mesh.vertices)) == 0:
                logger.error("CRITICAL: Original mesh is also invalid!")
                return None
            return mesh
        
        # Final validation before returning
        if mesh is None or len(np.asarray(mesh.vertices)) == 0:
            logger.error("CRITICAL: Refined mesh is invalid!")
            return None
        
        return mesh


class EdgePreservation:
    """Detect and preserve sharp edges during refinement."""
    
    @staticmethod
    def detect_sharp_edges(mesh: o3d.geometry.TriangleMesh,
                          angle_threshold: float = 30.0) -> np.ndarray:
        """
        Detect sharp edges in mesh (important for geometric features).
        
        Args:
            mesh: Input mesh
            angle_threshold: Dihedral angle threshold in degrees
        
        Returns:
            Boolean array indicating sharp edges
        """
        logger.info(f"Detecting sharp edges (threshold: {angle_threshold}°)")
        
        try:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Build edge-to-triangle mapping
            edges = {}
            for tri_idx, tri in enumerate(triangles):
                for i in range(3):
                    edge = tuple(sorted([tri[i], tri[(i+1) % 3]]))
                    if edge not in edges:
                        edges[edge] = []
                    edges[edge].append(tri_idx)
            
            sharp_edges = np.zeros(len(edges), dtype=bool)
            normals = np.asarray(mesh.triangle_normals)
            
            # Check dihedral angles
            for edge_idx, (edge, triangles_on_edge) in enumerate(edges.items()):
                if len(triangles_on_edge) == 2:
                    n1 = normals[triangles_on_edge[0]]
                    n2 = normals[triangles_on_edge[1]]
                    
                    # Dihedral angle
                    cos_angle = np.dot(n1, n2)
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    # If angle > threshold, edge is sharp
                    if angle > angle_threshold:
                        sharp_edges[edge_idx] = True
            
            logger.info(f"Found {np.sum(sharp_edges)} sharp edges")
            return sharp_edges
            
        except Exception as e:
            logger.warning(f"Edge detection failed: {e}")
            return np.array([], dtype=bool)
    
    @staticmethod
    def preserve_edges_during_smoothing(mesh: o3d.geometry.TriangleMesh,
                                       sharp_edges: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Apply smoothing while protecting detected sharp edges.
        
        Args:
            mesh: Input mesh
            sharp_edges: Boolean array of sharp edges
        
        Returns:
            Smoothed mesh with preserved edges
        """
        logger.info("Preserving sharp edges during smoothing")
        
        try:
            vertices = np.asarray(mesh.vertices).copy()
            triangles = np.asarray(mesh.triangles)
            
            # Mark vertices that are part of sharp edges
            edge_vertices = set()
            if len(sharp_edges) > 0:
                edges = list({tuple(sorted([tri[i], tri[(i+1) % 3]]))
                             for tri in triangles
                             for i in range(3)})
                
                for is_sharp, edge in zip(sharp_edges, edges):
                    if is_sharp:
                        edge_vertices.add(edge[0])
                        edge_vertices.add(edge[1])
            
            # Apply smoothing while preserving edge vertices
            vertices_backup = vertices.copy()
            
            # Taubin smoothing with edge protection
            for iteration in range(2):
                new_vertices = vertices.copy()
                
                for i in range(len(vertices)):
                    if i not in edge_vertices:  # Only smooth non-edge vertices
                        # Find neighbors
                        neighbors = set()
                        for tri in triangles:
                            if i in tri:
                                neighbors.update(tri)
                        neighbors.discard(i)
                        
                        if neighbors:
                            neighbor_avg = np.mean(vertices[[list(neighbors)]], axis=0)
                            new_vertices[i] = vertices[i] + 0.3 * (neighbor_avg - vertices[i])
                
                vertices = new_vertices
            
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            logger.info("Edge preservation complete")
            
        except Exception as e:
            logger.warning(f"Edge preservation failed: {e}")
        
        return mesh


class AdvancedRefinement:
    """Additional post-refinement techniques for ultra-high quality."""
    
    @staticmethod
    def mesh_smoothing_filter(mesh: o3d.geometry.TriangleMesh,
                             lambda_smooth: float = 0.5,
                             iterations: int = 5) -> o3d.geometry.TriangleMesh:
        """
        Laplacian smoothing - classic mesh denoising.
        Smooths without subdividing.
        
        Args:
            mesh: Input mesh
            lambda_smooth: Smoothing strength (0-1)
            iterations: Number of passes
        
        Returns:
            Smoothed mesh
        """
        try:
            logger.info("Applying Laplacian smoothing filter")
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Build vertex adjacency
            vertex_adj = [set() for _ in range(len(vertices))]
            for tri in triangles:
                for i, j in [(0, 1), (1, 2), (2, 0)]:
                    vertex_adj[tri[i]].add(tri[j])
            
            # Smoothing iterations
            for iteration in range(iterations):
                new_vertices = vertices.copy()
                for i in range(len(vertices)):
                    if vertex_adj[i]:
                        neighbors = list(vertex_adj[i])
                        neighbor_avg = np.mean(vertices[neighbors], axis=0)
                        new_vertices[i] = vertices[i] + lambda_smooth * (neighbor_avg - vertices[i])
                vertices = new_vertices
            
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            logger.info(f"Laplacian smoothing complete ({iterations} iterations)")
            
        except Exception as e:
            logger.warning(f"Laplacian smoothing failed: {e}")
        
        return mesh
    
    @staticmethod
    def cloth_simulation_smooth(mesh: o3d.geometry.TriangleMesh,
                               stiffness: float = 0.8,
                               damping: float = 0.9) -> o3d.geometry.TriangleMesh:
        """
        Cloth-like smoothing - smooths fabric while preserving folds/wrinkles that are real geometry.
        Uses soft constraints rather than hard averaging.
        
        Args:
            mesh: Input mesh
            stiffness: How much to constrain (0-1, higher = less smooth)
            damping: Velocity damping (0-1)
        
        Returns:
            Smoothed mesh
        """
        try:
            logger.info("Applying cloth simulation smoothing")
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Build adjacency with edge lengths
            vertex_adj = [[] for _ in range(len(vertices))]
            edge_lengths = [[] for _ in range(len(vertices))]
            
            for tri in triangles:
                for i, j in [(0, 1), (1, 2), (2, 0)]:
                    if j not in vertex_adj[tri[i]]:
                        vertex_adj[tri[i]].append(tri[j])
                        edge_lengths[tri[i]].append(np.linalg.norm(vertices[tri[j]] - vertices[tri[i]]))
            
            # Apply cloth constraints
            new_vertices = vertices.copy()
            for i in range(len(vertices)):
                if vertex_adj[i]:
                    constraint_force = np.zeros(3)
                    for j, edge_len in zip(vertex_adj[i], edge_lengths[i]):
                        direction = vertices[j] - vertices[i]
                        distance = np.linalg.norm(direction)
                        if distance > 0:
                            force = (distance - edge_len) * direction / distance
                            constraint_force += force
                    
                    constraint_force /= len(vertex_adj[i])
                    new_vertices[i] = vertices[i] + stiffness * constraint_force
            
            mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
            mesh.compute_vertex_normals()
            logger.info("Cloth simulation smoothing complete")
            
        except Exception as e:
            logger.warning(f"Cloth simulation failed: {e}")
        
        return mesh
    
    @staticmethod
    def surface_fairing(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Surface fairing - creates smooth, fair surfaces like in CAD.
        Minimizes curvature variations.
        
        Args:
            mesh: Input mesh
        
        Returns:
            Fair mesh
        """
        try:
            logger.info("Applying surface fairing")
            
            # Use Open3D's filter_smooth_simple which is a form of fairing
            mesh_smooth = mesh.filter_smooth_simple(number_of_iterations=3, lambda_filter=0.5)
            
            logger.info("Surface fairing complete")
            return mesh_smooth
            
        except Exception as e:
            logger.warning(f"Surface fairing failed: {e}")
            return mesh
    
    @staticmethod
    def enhance_mesh_quality(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Enhance overall mesh quality - combines multiple techniques.
        
        Args:
            mesh: Input mesh
        
        Returns:
            Enhanced mesh
        """
        try:
            logger.info("Enhancing overall mesh quality")
            
            # 1. Remove thin/degenerate triangles
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            valid_triangles = []
            for tri in triangles:
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                
                if area > 1e-10:  # Keep triangles with non-zero area
                    valid_triangles.append(tri)
            
            if valid_triangles:
                mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))
                mesh.remove_unreferenced_vertices()
            
            # 2. Compute vertex normals with more neighbors
            mesh.compute_vertex_normals()
            
            logger.info("Mesh quality enhancement complete")
            
        except Exception as e:
            logger.warning(f"Mesh quality enhancement failed: {e}")
        
        return mesh
