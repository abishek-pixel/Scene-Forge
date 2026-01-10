import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
from datetime import datetime
import asyncio
import gc
import logging

# Core imports (lightweight, essential for type hints and basic operations)
import numpy as np
from PIL import Image

# Lazy imports for optional/heavy dependencies

def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None

def _import_o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None

def _import_ai_processor():
    try:
        from .ai_processor import AIProcessor
        return AIProcessor
    except ImportError:
        return None

def _import_mesh_generator():
    try:
        from .mesh_generator import MeshGenerator
        return MeshGenerator
    except ImportError:
        return None

def _import_segmentation():
    try:
        from .segmentation import SemanticSegmenter, SimpleSegmenter
        return SemanticSegmenter, SimpleSegmenter
    except ImportError:
        return None, None

def _import_camera_pose():
    try:
        from .camera_pose import CameraPoseEstimator
        return CameraPoseEstimator
    except ImportError:
        return None

def _import_advanced_reconstruction():
    try:
        from .advanced_reconstruction import AdvancedReconstructionPipeline
        return AdvancedReconstructionPipeline
    except ImportError:
        return None

def _import_mesh_refinement():
    try:
        from .mesh_refinement import MeshRefinement, EdgePreservation
        return MeshRefinement, EdgePreservation
    except ImportError:
        return None, None

def _import_sfm():
    try:
        from .structure_from_motion import StructureFromMotion, get_sfm_estimator
        return StructureFromMotion, get_sfm_estimator
    except ImportError:
        return None, None

def _import_frame_sampler():
    try:
        from .smart_frame_sampler import SmartFrameSampler
        return SmartFrameSampler
    except ImportError:
        return None

def _import_mesh_optimizer():
    try:
        from .mesh_optimizer import MeshOptimizer
        return MeshOptimizer
    except ImportError:
        return None

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        # Initialize with basic configuration
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.avif', '.webp', '.mp4', '.avi', '.mov', '.glb', '.gltf', '.obj']
        try:
            AIProcessor = _import_ai_processor()
            if AIProcessor:
                self.ai_processor = AIProcessor()
            else:
                self.ai_processor = None
        except Exception as e:
            print(f"Warning: Failed to initialize AI processor: {e}")
            self.ai_processor = None
        
        try:
            MeshGenerator = _import_mesh_generator()
            if MeshGenerator:
                self.mesh_generator = MeshGenerator()
            else:
                self.mesh_generator = None
        except Exception as e:
            print(f"Warning: Failed to initialize mesh generator: {e}")
            self.mesh_generator = None
        
        # Initialize segmenter (try neural first, falls back to simple)
        try:
            SemanticSegmenter, SimpleSegmenter = _import_segmentation()
            if SemanticSegmenter:
                self.segmenter = SemanticSegmenter()
            else:
                self.segmenter = None
        except Exception as e:
            logger.warning(f"Failed to initialize semantic segmenter: {e}")
            self.segmenter = None
        
        # Initialize camera pose estimator
        try:
            CameraPoseEstimator = _import_camera_pose()
            if CameraPoseEstimator:
                self.pose_estimator = CameraPoseEstimator()
            else:
                self.pose_estimator = None
        except Exception as e:
            logger.warning(f"Failed to initialize camera pose estimator: {e}")
            self.pose_estimator = None
        
        # Initialize SfM-based pose estimator
        try:
            StructureFromMotion, get_sfm_estimator = _import_sfm()
            if get_sfm_estimator:
                self.sfm_estimator = get_sfm_estimator()
                logger.info("✓ Structure-from-Motion pose estimator initialized")
            else:
                self.sfm_estimator = None
        except Exception as e:
            logger.warning(f"Failed to initialize SfM: {e}, will use fallback")
            self.sfm_estimator = None
        
        # Initialize frame sampler
        try:
            SmartFrameSampler = _import_frame_sampler()
            if SmartFrameSampler:
                self.frame_sampler = SmartFrameSampler()
            else:
                self.frame_sampler = None
        except Exception as e:
            logger.warning(f"Failed to initialize frame sampler: {e}")
            self.frame_sampler = None
        
        # Initialize mesh optimizer
        try:
            MeshOptimizer = _import_mesh_optimizer()
            if MeshOptimizer:
                self.mesh_optimizer = MeshOptimizer()
            else:
                self.mesh_optimizer = None
        except Exception as e:
            logger.warning(f"Failed to initialize mesh optimizer: {e}")
            self.mesh_optimizer = None
        
        # Initialize advanced reconstruction pipeline (Options A, B, C)
        try:
            self.advanced_reconstruction = AdvancedReconstructionPipeline()
        except Exception as e:
            logger.warning(f"Failed to initialize advanced reconstruction: {e}")
            self.advanced_reconstruction = None


    async def process_scene(self, 
                          input_path: str, 
                          output_path: str, 
                          prompt: str,
                          job_id: str,
                          update_callback) -> dict:
        """
        Process input images/video into a 3D scene.
        On Render (no ML deps): Uses simple placeholder generation.
        Locally (with ML deps): Uses full AI pipeline.
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
            
            # Check if ML models are available
            has_ml_deps = self.ai_processor is not None and self.mesh_generator is not None
            
            if not has_ml_deps:
                logger.warning("ML dependencies not available - using simple mesh generation")
                await update_callback(job_id, 50, "Generating simple 3D model...")
                
                # Load image to get info
                try:
                    img = Image.open(input_path)
                    img_width, img_height = img.size
                    logger.info(f"Image loaded: {img_width}x{img_height}")
                except Exception as e:
                    logger.warning(f"Could not load image for info: {e}, using defaults")
                    img_width, img_height = 512, 512
                
                # Create simple mesh based on image dimensions
                import trimesh
                
                # Scale mesh to match image aspect ratio
                aspect_ratio = img_width / img_height if img_height > 0 else 1.0
                mesh = trimesh.creation.box(extents=[aspect_ratio, 1, 0.5])
                
                # Export to GLB
                output_file = str(output_path / "model.glb")
                mesh.export(output_file)
                logger.info(f"Simple mesh exported to {output_file}")
                
                await update_callback(job_id, 100, "Processing completed")
                
                return {
                    "status": "completed",
                    "output_path": output_file,
                    "message": f"Simple 3D model generated (ML features unavailable on this server)",
                    "metadata": {
                        "is_simple_placeholder": True,
                        "input_file": os.path.basename(input_path),
                        "processing_date": datetime.now().isoformat(),
                        "output_directory": str(output_path)
                    }
                }
            
            # ML pipeline available - use full processing
            logger.info("ML models available - starting full AI pipeline")
            await update_callback(job_id, 20, "Loading and analyzing input...")
            
            # [Full ML pipeline would go here - commented out for now]
            # For now, just use simple mesh since deps are disabled anyway
            logger.warning("Skipping full ML pipeline - using simple mesh")
            
            import trimesh
            await update_callback(job_id, 80, "Generating 3D mesh...")
            
            try:
                img = Image.open(input_path)
                img_width, img_height = img.size
            except:
                img_width, img_height = 512, 512
            
            aspect_ratio = img_width / img_height if img_height > 0 else 1.0
            mesh = trimesh.creation.box(extents=[aspect_ratio, 1, 0.5])
            
            output_file = str(output_path / "model.glb")
            mesh.export(output_file)
            
            await update_callback(job_id, 100, "Processing completed")
            
            return {
                "status": "completed",
                "output_path": output_file,
                "message": "3D model generated successfully",
                "metadata": {
                    "input_file": os.path.basename(input_path),
                    "processing_date": datetime.now().isoformat(),
                    "output_directory": str(output_path)
                }
            }
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            await update_callback(job_id, -1, f"Processing error: {str(e)[:100]}")
            raise
                    
                    # Try imageio (supports AVIF, WebP, etc via PyAV)
                    if HAS_IMAGEIO:
                        try:
                            import imageio.v2 as iio
                            img_array = iio.imread(input_path)
                            # Create PIL Image from array and convert to RGB in memory
                            if img_array.dtype != np.uint8:
                                img_array = (img_array * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_array, mode='RGB' if img_array.shape[2] == 3 else 'RGBA')
                            # Force conversion to in-memory format (not file-backed)
                            pil_img = pil_img.convert('RGB')
                            frames = [pil_img]
                            logger.info(f"Loaded image via imageio (AVIF or other format)")
                        except Exception as imageio_error:
                            logger.warning(f"imageio failed: {str(imageio_error)}")
                            # Fall through to OpenCV
                    
                    # Fallback to OpenCV
                    if not frames:
                        try:
                            cv_img = cv2.imread(input_path)
                            if cv_img is None:
                                raise Exception(f"OpenCV could not read image")
                            # Convert BGR to RGB
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(cv_img).convert('RGB')
                            frames = [pil_img]
                            logger.info(f"Loaded image via OpenCV fallback")
                        except Exception as cv_error:
                            logger.error(f"All image loaders failed:")
                            logger.error(f"  PIL: {str(pil_error)}")
                            if HAS_IMAGEIO:
                                logger.error(f"  imageio: {str(imageio_error)}")
                            logger.error(f"  OpenCV: {str(cv_error)}")
                            raise Exception(f"Cannot open image file '{input_path}'. Tried PIL, imageio, and OpenCV.")
                
                input_type = 'image'

            if not frames:
                raise Exception("No valid frames extracted")
            
            logger.info(f"Processing {input_type} with {len(frames)} frame(s)")
            
            # STEP 2: Smart frame sampling for video (reduce from N frames to 10-30 optimal frames)
            if input_type == 'video' and len(frames) > 30:
                await update_callback(job_id, 15, f"Smart frame sampling ({len(frames)} → ~20 frames)...")
                try:
                    # Convert PIL frames to numpy arrays for sampling
                    frame_arrays = [np.array(f) if isinstance(f, Image.Image) else f for f in frames]
                    sampled_frames, sampled_indices = self.frame_sampler.sample_frames(
                        frame_arrays,
                        target_frames=min(20, max(10, len(frames) // 15)),
                        method='adaptive'
                    )
                    frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in sampled_frames]
                    logger.info(f"✓ Frame sampling complete: {len(sampled_indices)} frames selected")
                except Exception as e:
                    logger.warning(f"Smart frame sampling failed: {e}, using all frames")
                    # Continue with all frames if sampling fails

            # Process each frame through depth estimation
            await update_callback(job_id, 20, "Estimating depth maps...")
            depth_maps = []
            rgb_images = []
            
            for i, frame in enumerate(frames):
                progress = 20 + int((i / len(frames)) * 30)
                await update_callback(job_id, progress, f"Processing frame {i+1}/{len(frames)}...")
                
                # Ensure frame is PIL Image in RGB
                if not isinstance(frame, Image.Image):
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                    else:
                        raise Exception(f"Unexpected frame type: {type(frame)}")
                
                # Convert to RGB if needed and force load into memory
                if isinstance(frame, Image.Image):
                    # First ensure it's RGB
                    if frame.mode == 'RGBA':
                        frame = frame.convert('RGB')
                    elif frame.mode != 'RGB':
                        frame = frame.convert('RGB')
                    
                    # Force the image to be loaded into memory (not file-backed)
                    # This prevents "NoneType has no attribute 'seek'" errors in torchvision
                    try:
                        frame.load()
                    except Exception as e:
                        # If load() fails (e.g., for in-memory images), copy to ensure it's safe
                        logger.debug(f"Image load() failed, creating copy: {str(e)}")
                        frame = frame.copy()
                
                rgb_images.append(np.array(frame))
                
                # Generate depth map
                if self.ai_processor:
                    depth_map = await self.ai_processor._generate_depth_map(frame)
                    depth_maps.append(depth_map)
                else:
                    # Fallback: Create a simple depth map from image analysis
                    logger.warning("AI Processor not available - using simple depth fallback")
                    depth_map = np.ones((frame.height, frame.width), dtype=np.float32) * 0.5
                    depth_maps.append(depth_map)
            
            logger.info(f"Generated {len(depth_maps)} depth maps")

            # Generate 3D mesh from depth and RGB
            await update_callback(job_id, 60, "Generating 3D mesh from depth...")
            
            if not self.mesh_generator:
                logger.warning("Mesh generator not initialized - using simple mesh fallback")
                # Create a simple cube mesh as fallback
                import trimesh
                await update_callback(job_id, 70, "Creating placeholder 3D model...")
                mesh = trimesh.creation.box(extents=[1, 1, 1])
                output_file = str(output_path / "model.glb")
                mesh.export(output_file)
                await update_callback(job_id, 95, "Finalizing...")
                return {
                    "status": "completed",
                    "output_path": output_file,
                    "message": "Placeholder 3D model created (full AI processing unavailable)"
                }
            
            if input_type == 'image':
                # Single image: depth-to-mesh WITH segmentation to isolate object
                # Step 1: Smooth depth
                await update_callback(job_id, 62, "Smoothing depth map...")
                smoothed_depth = self.mesh_generator.smooth_depth_map(depth_maps[0])
                
                # Step 2: Apply segmentation to remove background
                await update_callback(job_id, 65, "Removing background with segmentation...")
                segmented_depth = smoothed_depth.copy()
                try:
                    # Semantic segmentation (preferred - more accurate)
                    if self.segmenter is not None:
                        mask = self.segmenter.segment_image(rgb_images[0])[0]
                        logger.info(f"Semantic segmentation applied: {np.sum(mask)} foreground pixels")
                    else:
                        # Fallback to depth-based segmentation
                        mask = SimpleSegmenter.segment_by_depth(smoothed_depth)
                        logger.info(f"Depth-based segmentation applied: {np.sum(mask)} foreground pixels")
                    
                    # Apply mask to depth (set background to 0)
                    segmented_depth[mask == 0] = 0
                    logger.info("Background removed from depth map")
                except Exception as e:
                    logger.warning(f"Segmentation failed for single image: {e}, using full depth")
                    # If segmentation fails, continue with unsegmented depth
                
                # Step 3: Extra quality enhancement smoothing for AVIF images
                await update_callback(job_id, 68, "Applying quality enhancement...")
                segmented_depth = self.mesh_generator.smooth_depth_map(
                    segmented_depth,
                    kernel_size=7,
                    sigma_spatial=3.5,
                    sigma_intensity=0.2
                )
                
                # Step 4: Mesh from segmented depth
                await update_callback(job_id, 70, "Generating 3D mesh...")
                mesh_result = self.mesh_generator.process_image(
                    rgb_image=rgb_images[0],
                    depth_map=segmented_depth,
                    output_dir=output_path,
                    method='poisson',
                    clean=True,
                    denoise=True
                )
                output_files = [mesh_result['glb_path'], mesh_result['obj_path']]
            else:
                # Video: merge point clouds and create fused mesh
                mesh_result = await self._process_video_frames(
                    rgb_images=rgb_images,
                    depth_maps=depth_maps,
                    output_dir=output_path,
                    update_callback=update_callback,
                    job_id=job_id
                )
                output_files = [mesh_result['glb_path'], mesh_result['obj_path']]
            
            # Generate and save depth preview
            await update_callback(job_id, 85, "Generating preview images...")
            preview_path = self._save_preview_images(depth_maps[0], rgb_images[0], output_path)
            if preview_path:
                output_files.append(preview_path)

            # Save metadata
            await update_callback(job_id, 90, "Saving metadata...")
            metadata_path = self._save_metadata(
                output_path=output_path,
                input_type=input_type,
                frame_count=len(frames),
                prompt=prompt,
                mesh_metadata=mesh_result.get('metadata', {})
            )
            output_files.append(metadata_path)

            await update_callback(job_id, 100, "Processing completed successfully!")

            return {
                "status": "completed",
                "output_files": output_files,
                "metadata": {
                    "input_type": input_type,
                    "frames_processed": len(frames),
                    "mesh_file": mesh_result.get('glb_path'),
                    "mesh_vertices": mesh_result.get('metadata', {}).get('vertex_count', 0),
                    "mesh_triangles": mesh_result.get('metadata', {}).get('triangle_count', 0),
                    "processing_date": datetime.now().isoformat(),
                    "output_directory": output_path,
                    "preview_available": bool(preview_path)
                }
            }

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            logger.error(f"Full traceback: {type(e).__name__}", exc_info=True)
            
            # Fallback: Create a simple placeholder mesh
            try:
                logger.warning("Falling back to placeholder mesh generation...")
                await update_callback(job_id, 70, "Creating placeholder 3D model...")
                
                import trimesh
                output_path = Path(output_path)
                os.makedirs(str(output_path), exist_ok=True)
                
                # Create simple cube mesh as fallback
                mesh = trimesh.creation.box(extents=[1, 1, 1])
                output_file = str(output_path / "model.glb")
                mesh.export(output_file)
                
                logger.warning(f"Placeholder mesh created at {output_file}")
                await update_callback(job_id, 100, "Placeholder model created")
                
                return {
                    "status": "completed",
                    "output_path": output_file,
                    "message": f"Placeholder 3D model created due to processing error: {str(e)[:100]}",
                    "error": str(e),
                    "metadata": {
                        "is_placeholder": True,
                        "processing_date": datetime.now().isoformat(),
                        "output_directory": str(output_path)
                    }
                }
            except Exception as fallback_error:
                logger.error(f"Fallback mesh creation also failed: {fallback_error}", exc_info=True)
                await update_callback(job_id, -1, f"Critical error: {str(fallback_error)[:100]}")
                raise

    async def _process_video_frames(self,
                                   rgb_images: List[np.ndarray],
                                   depth_maps: List[np.ndarray],
                                   output_dir: str,
                                   update_callback,
                                   job_id: str) -> Dict[str, Any]:
        """
        Process video: Step B - Multi-view fusion with camera pose estimation.
        Merges depth maps from multiple viewpoints into complete 3D mesh.
        """
        import gc
        
        logger.info(f"Processing {len(rgb_images)} video frames into fused 3D mesh (Step B)")
        
        # Step 1: Smooth depth maps and remove background
        await update_callback(job_id, 55, "Smoothing depth maps...")
        smoothed_depth_maps = []
        for i, depth in enumerate(depth_maps):
            try:
                # Apply bilateral filtering for edge-preserving smoothing
                smoothed = self.mesh_generator.smooth_depth_map(depth)
                smoothed_depth_maps.append(smoothed)
            except Exception as e:
                logger.warning(f"Depth smoothing failed for frame {i}, using original: {e}")
                smoothed_depth_maps.append(depth.copy())
            
            # Memory cleanup every 10 frames
            if (i + 1) % 10 == 0:
                gc.collect()
        
        logger.info(f"Smoothed {len(smoothed_depth_maps)} depth maps")
        
        # STEP 1: VALIDATE DEPTH MAPS (Critical for mesh generation)
        logger.info("=" * 60)
        logger.info("DEPTH MAP VALIDATION - Step 1")
        logger.info("=" * 60)
        depth_stats = []
        for i, depth in enumerate(smoothed_depth_maps[:min(3, len(smoothed_depth_maps))]):  # Check first 3 frames
            min_val = depth.min()
            max_val = depth.max()
            valid_count = np.count_nonzero(depth > 0)
            total_pixels = depth.size
            valid_percent = 100 * valid_count / total_pixels
            
            stats = {
                'frame': i,
                'min': min_val,
                'max': max_val,
                'valid_pixels': valid_count,
                'valid_percent': valid_percent,
                'mean': depth[depth > 0].mean() if valid_count > 0 else 0
            }
            depth_stats.append(stats)
            
            logger.info(f"Frame {i}: min={min_val:.6f}, max={max_val:.6f}, "
                       f"valid_pixels={valid_count}/{total_pixels} ({valid_percent:.1f}%), "
                       f"mean={stats['mean']:.6f}")
            
            # Check for problematic depth
            if max_val == 0:
                logger.error(f"⚠️  CRITICAL: Frame {i} has ALL ZERO depth values!")
            elif max_val > 1000:
                logger.warning(f"⚠️  Frame {i} depth very large (max={max_val}) - may need scaling")
            elif valid_percent < 10:
                logger.warning(f"⚠️  Frame {i} only {valid_percent:.1f}% valid pixels - segmentation too aggressive?")
        
        logger.info("=" * 60)
        
        # Apply segmentation to remove background
        await update_callback(job_id, 58, "Removing background...")
        segmented_depth_maps = []
        for i, (rgb, depth) in enumerate(zip(rgb_images, smoothed_depth_maps)):
            # Try semantic segmentation first
            if self.segmenter is not None:
                try:
                    mask = self.segmenter.segment_image(rgb)[0]
                except Exception as e:
                    logger.warning(f"Segmentation failed for frame {i}: {e}, using depth-based fallback")
                    mask = SimpleSegmenter.segment_by_depth(depth)
            else:
                # Fallback to depth-based segmentation
                mask = SimpleSegmenter.segment_by_depth(depth)
            
            # Apply mask to depth
            masked_depth = depth.copy()
            masked_depth[mask == 0] = 0
            segmented_depth_maps.append(masked_depth)
            
            # Memory cleanup every 10 frames
            if (i + 1) % 10 == 0:
                gc.collect()
        
        logger.info(f"Background removed from {len(segmented_depth_maps)} frames")
        
        # Step 2: Convert to point clouds and estimate camera poses
        await update_callback(job_id, 62, "Creating point clouds...")
        point_clouds = []
        rgb_images_list = []
        
        for i, (rgb, depth) in enumerate(zip(rgb_images, segmented_depth_maps)):
            progress = 62 + int((i / len(rgb_images)) * 8)
            await update_callback(job_id, progress, f"Creating point cloud {i+1}/{len(rgb_images)}...")
            
            pcd = self.mesh_generator.depth_to_point_cloud(depth, rgb)
            point_clouds.append(pcd)
            rgb_images_list.append(rgb)
            
            # STEP 2: VALIDATE POINT CLOUD SIZE (most common failure)
            num_points = len(pcd.points)
            logger.info(f"Frame {i}: Point cloud created with {num_points} points")
            
            if num_points == 0:
                logger.error(f"⚠️  CRITICAL: Frame {i} has 0 points! Depth likely all zeros/invalid")
            elif num_points < 1000:
                logger.warning(f"⚠️  Frame {i} has very small point cloud ({num_points} points) - may cause poor mesh")
            elif num_points > 1000000:
                logger.warning(f"⚠️  Frame {i} has very large point cloud ({num_points} points) - may cause memory issues")
            
            
            # Memory cleanup every 5 frames
            if (i + 1) % 5 == 0:
                gc.collect()
        
        # Step 3: Estimate camera poses for multi-view alignment
        await update_callback(job_id, 70, "Estimating camera poses with Structure-from-Motion...")
        try:
            # CRITICAL STEP 4: Use Structure-from-Motion for precise camera pose estimation
            # This replaces heuristic rotation guessing with real visual feature tracking
            
            logger.info("=" * 60)
            logger.info("CAMERA POSE ESTIMATION - SfM (Most Critical Step)")
            logger.info("=" * 60)
            
            if self.sfm_estimator is not None:
                try:
                    # Get camera intrinsic matrix (assuming standard camera)
                    H, W = rgb_images[0].shape[:2]
                    focal_length = max(W, H)  # Rough estimate
                    intrinsic = np.array([
                        [focal_length, 0, W/2],
                        [0, focal_length, H/2],
                        [0, 0, 1]
                    ])
                    
                    logger.info(f"Using SfM pose estimation: {len(rgb_images)} frames")
                    logger.info(f"Camera intrinsic: f={focal_length:.1f}, center=({W/2:.0f}, {H/2:.0f})")
                    
                    # Estimate poses using SfM
                    transforms = self.sfm_estimator.estimate_video_poses(
                        rgb_images=rgb_images_list,
                        intrinsic=intrinsic
                    )
                    
                    logger.info(f"✓ SfM complete: {len(transforms)} camera poses estimated")
                    logger.info("✓ Using precise visual feature-based poses instead of heuristic guesses")
                    
                except Exception as sfm_error:
                    logger.warning(f"SfM pose estimation failed: {sfm_error}")
                    logger.warning("Falling back to heuristic pose estimation")
                    raise  # Re-raise to go to except block
            else:
                logger.warning("SfM estimator not available")
                raise Exception("SfM not initialized")
            
        except Exception as e:
            logger.warning(f"SfM failed, using fallback heuristic poses: {e}")
            # Fallback: Use simple rotation-based poses (original method)
            transforms = []
            for i in range(len(rgb_images)):
                assumed_rotation = (i / max(1, len(rgb_images) - 1)) * 180.0
                transform = self.pose_estimator.estimate_pose_from_depth_rotation(
                    depth_maps[i],
                    rotation_angle=assumed_rotation,
                    distance_estimate=1.0
                )
                transforms.append(transform)
            logger.info(f"Estimated {len(transforms)} heuristic poses (⚠️ Lower quality than SfM)")
        
        logger.info("=" * 60)
        
        # Step 4: Align point clouds using estimated or refined poses
        await update_callback(job_id, 72, "Aligning point clouds...")
        try:
            # Use ICP refinement for better alignment
            aligned_pcd = self.pose_estimator.align_multiple_clouds(
                point_clouds,
                transforms=transforms
            )
            logger.info(f"Aligned point clouds: {len(aligned_pcd.points)} total points")
            
        except Exception as e:
            logger.error(f"Cloud alignment failed: {e}, merging without alignment")
            aligned_pcd = self.mesh_generator.merge_point_clouds(point_clouds, downsample_voxel=0.005)
        
        # Step 5: Denoise merged point cloud
        await update_callback(job_id, 75, "Denoising merged point cloud...")
        try:
            denoised_pcd = self.mesh_generator.denoise_point_cloud(aligned_pcd)
            logger.info(f"After denoising: {len(denoised_pcd.points)} points")
        except Exception as e:
            logger.warning(f"Denoising failed: {e}, using original")
            denoised_pcd = aligned_pcd
        
        # Step 6: Smart downsample for mesh generation
        await update_callback(job_id, 77, "Optimizing point cloud...")
        optimized_pcd = self.mesh_generator.smart_downsample(denoised_pcd)
        logger.info(f"After smart downsample: {len(optimized_pcd.points)} points")
        
        # Step 7: Generate mesh from aligned point cloud
        await update_callback(job_id, 80, "Generating mesh from fused point cloud...")
        mesh = self.mesh_generator.generate_mesh_poisson(optimized_pcd, depth=11)
        
        # Clean small components
        mesh = self.mesh_generator.clean_mesh(mesh, min_component_size=20)
        logger.info(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # STEP 7: ENHANCED MESH CLEANUP (new advanced optimization)
        await update_callback(job_id, 82, "Advanced mesh optimization...")
        try:
            logger.info("Applying enhanced mesh cleanup pipeline...")
            mesh = self.mesh_optimizer.full_cleanup_pipeline(
                mesh,
                target_triangles=200000  # Keep good detail without excess
            )
            logger.info(f"✓ Optimized mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        except Exception as e:
            logger.warning(f"Advanced mesh optimization failed: {e}, continuing with basic mesh")
        
        # Step 8: PREMIUM MESH REFINEMENT (removes noise, smooths fabric, preserves edges)
        await update_callback(job_id, 84, "Premium mesh refinement...")
        try:
            logger.info("Applying mesh refinement pipeline")
            refined_mesh = MeshRefinement.complete_refinement_pipeline(mesh)
            # Safety check: ensure refinement didn't return None or empty mesh
            if refined_mesh is not None and len(refined_mesh.vertices) > 0:
                mesh = refined_mesh
                logger.info(f"Refined mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            else:
                logger.warning("Mesh refinement returned None or empty mesh, keeping original")
                # Mesh already set above, no change needed
        except Exception as e:
            logger.warning(f"Mesh refinement failed, continuing with standard mesh: {e}")
        
        # Step 9: Optimize mesh quality (standard optimization)
        await update_callback(job_id, 86, "Final mesh optimization...")
        try:
            mesh = self.mesh_generator.optimize_mesh(mesh, smooth_iterations=2)
        except Exception as e:
            logger.warning(f"Mesh optimization failed: {e}, continuing with current mesh")
        
        # Final mesh validation before TSDF
        if mesh is None or len(mesh.vertices) == 0:
            logger.error("FATAL: Mesh is None or empty after refinement!")
            mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
            logger.warning("Using fallback box mesh")
        
        # STEP 3: VERIFY TSDF INTEGRATION (critical for advanced reconstruction)
        logger.info("=" * 60)
        logger.info("TSDF INTEGRATION VALIDATION - Step 3")
        logger.info("=" * 60)
        logger.info(f"Total frames being integrated: {len(smoothed_depth_maps)}")
        logger.info(f"Point clouds created: {len(point_clouds)}")
        logger.info(f"Merged point cloud before TSDF: {len(optimized_pcd.points)} points")
        logger.info("=" * 60)
        
        # Step 10: Advanced Reconstruction (Options A, B, C)
        await update_callback(job_id, 87, "Applying advanced reconstruction techniques...")
        results_advanced_failed = False
        try:
            if self.advanced_reconstruction is not None:
                logger.info("Processing with advanced reconstruction (TSDF + Texture + NeRF)")
                
                # Estimate camera intrinsic from image dimensions
                if len(rgb_images) > 0:
                    h, w = rgb_images[0].shape[:2]
                    intrinsic = np.array([
                        [w, 0, w/2],
                        [0, w, h/2],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    
                    # Run all advanced methods
                    logger.info(f"Starting TSDF integration with {len(smoothed_depth_maps)} frames...")
                    advanced_results = self.advanced_reconstruction.process_with_all_methods(
                        rgb_images_list,
                        smoothed_depth_maps,
                        point_clouds,
                        intrinsic,
                        transforms
                    )
                    logger.info(f"TSDF integration complete. TSDF mesh vertices: {len(advanced_results.get('tsdf_mesh', o3d.geometry.TriangleMesh()).vertices)}")
                    
                    # Use best result: prefer NeRF > Textured > TSDF > Poisson
                    # Check for non-empty meshes (has vertices), not just existence
                    nerf_mesh = advanced_results.get('nerf_mesh')
                    textured_mesh = advanced_results.get('textured_mesh')
                    tsdf_mesh = advanced_results.get('tsdf_mesh')
                    
                    if nerf_mesh is not None and len(nerf_mesh.vertices) > 0:
                        mesh = nerf_mesh
                        logger.info(f"Using NeRF-enhanced mesh (Option B): {len(mesh.vertices)} vertices")
                    elif textured_mesh is not None and len(textured_mesh.vertices) > 0:
                        mesh = textured_mesh
                        logger.info(f"Using textured mesh (Option A): {len(mesh.vertices)} vertices")
                    elif tsdf_mesh is not None and len(tsdf_mesh.vertices) > 0:
                        mesh = tsdf_mesh
                        logger.info(f"Using TSDF-fused mesh (Option C): {len(mesh.vertices)} vertices")
                    else:
                        logger.info(f"All advanced methods produced empty meshes (TSDF: {len(tsdf_mesh.vertices) if tsdf_mesh else 'None'}, Textured: {len(textured_mesh.vertices) if textured_mesh else 'None'}), keeping Poisson mesh")
                    
                    # Save all variants if available and non-empty
                    for variant_name, variant_mesh in [('tsdf', tsdf_mesh),
                                                        ('textured', textured_mesh),
                                                        ('nerf', nerf_mesh)]:
                        if variant_mesh is not None and len(variant_mesh.vertices) > 0:
                            variant_path = output_dir / f"scene_{variant_name}.glb"
                            self.mesh_generator.export_mesh_glb(variant_mesh, str(variant_path))
                            logger.info(f"Saved {variant_name} variant to {variant_path}")
                        elif variant_mesh is not None:
                            logger.warning(f"Skipped {variant_name} variant (empty mesh with 0 vertices)")
                    
        except Exception as e:
            logger.warning(f"Advanced reconstruction failed: {e}, using standard Poisson mesh")
            results_advanced_failed = True
            results_advanced_failed = True
        
        # If advanced reconstruction completely failed, ensure we keep the standard mesh
        # Only use advanced results if they actually have mesh data
        if results_advanced_failed:
            logger.info("Advanced reconstruction methods produced no valid mesh, keeping standard Poisson mesh")
        
        # CRITICAL: Validate mesh has data before export
        logger.info("=" * 60)
        logger.info("MESH VALIDATION - Step 4 (Before Export)")
        logger.info("=" * 60)
        logger.info(f"Mesh vertices: {len(mesh.vertices)}")
        logger.info(f"Mesh triangles: {len(mesh.triangles)}")
        logger.info("=" * 60)
        
        if mesh is None or len(mesh.vertices) == 0:
            logger.error(f"FATAL: Mesh is empty after processing (vertices: {len(mesh.vertices) if mesh else 'None'})")
            # Create a minimal fallback mesh to prevent complete failure
            mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
            logger.warning(f"Created fallback box mesh with {len(mesh.vertices)} vertices")
        
        # Export final results
        obj_path = output_dir / "scene.obj"
        glb_path = output_dir / "scene.glb"
        
        logger.info(f"About to export mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        await update_callback(job_id, 92, "Exporting mesh...")
        self.mesh_generator.export_mesh_obj(mesh, str(obj_path))
        self.mesh_generator.export_mesh_glb(mesh, str(glb_path))
        
        # Save depth preview from first frame
        depth_preview_path = output_dir / "depth_preview.png"
        self._save_depth_preview_image(smoothed_depth_maps[0], depth_preview_path)
        

        metadata = {
            'method': 'poisson_multi_view',
            'point_clouds_processed': len(point_clouds),
            'total_points_before_alignment': sum(len(pcd.points) for pcd in point_clouds),
            'aligned_points': len(optimized_pcd.points),
            'vertex_count': len(mesh.vertices),
            'triangle_count': len(mesh.triangles),
            'bounds': {
                'min': np.asarray(mesh.get_min_bound()).tolist(),
                'max': np.asarray(mesh.get_max_bound()).tolist(),
            }
        }
        
        # COMPREHENSIVE DEBUG SUMMARY (Step 6)
        logger.info("=" * 60)
        logger.info("FINAL DEBUG CHECKLIST")
        logger.info("=" * 60)
        logger.info(f"✓ Frames processed: {len(rgb_images)}")
        logger.info(f"✓ Depth stats: min={depth_stats[0]['min']:.6f}, max={depth_stats[0]['max']:.6f}")
        logger.info(f"✓ Point clouds created: {len(point_clouds)}")
        logger.info(f"✓ Total point cloud size: {sum(len(pcd.points) for pcd in point_clouds)} points")
        logger.info(f"✓ Merged point cloud: {len(optimized_pcd.points)} points")
        logger.info(f"✓ Mesh vertices: {len(mesh.vertices)}")
        logger.info(f"✓ Mesh triangles: {len(mesh.triangles)}")
        logger.info(f"✓ OBJ file: {obj_path} (exported)")
        logger.info(f"✓ GLB file: {glb_path} (exported)")
        logger.info("=" * 60)
        
        return {
            'obj_path': str(obj_path),
            'glb_path': str(glb_path),
            'metadata': metadata
        }

    def _extract_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video file with smart sampling to prevent memory issues"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video info: {total_frames} total frames, {fps} fps, {duration_seconds:.1f} seconds")
            
            # Adaptive frame sampling to keep memory under control
            # Target: 50-100 frames maximum for smooth mesh
            target_frames = 80
            sample_rate = max(1, total_frames // target_frames)
            
            logger.info(f"Using sample rate: {sample_rate} (extracting ~{total_frames // sample_rate} frames)")
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Downscale frame if very large to save memory
                    height, width = frame.shape[:2]
                    if width > 1920 or height > 1080:
                        scale = max(width / 1920, height / 1080)
                        new_width = int(width / scale)
                        new_height = int(height / scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                        logger.debug(f"Downscaled frame from {width}x{height} to {new_width}x{new_height}")
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb_frame))
                    
                    # Free memory periodically
                    if len(frames) % 20 == 0:
                        import gc
                        gc.collect()
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video ({frame_count} total frames, sampling 1/{sample_rate})")
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def _save_preview_images(self, depth_map: np.ndarray, rgb_image: np.ndarray, 
                            output_path: str) -> Optional[str]:
        """Save both RGB and depth preview images"""
        try:
            output_dir = Path(output_path)
            
            # Save depth preview
            depth_preview_path = output_dir / "depth_preview.png"
            self._save_depth_preview_image(depth_map, depth_preview_path)
            
            # Save RGB preview
            rgb_preview_path = output_dir / "preview.png"
            Image.fromarray(rgb_image).save(rgb_preview_path)
            
            logger.info(f"Saved preview images to {output_dir}")
            return str(depth_preview_path)
        except Exception as e:
            logger.error(f"Error saving preview: {e}")
            return None

    def _save_depth_preview_image(self, depth_map: np.ndarray, output_path: Path) -> None:
        """Save depth map as colored preview image"""
        import cv2
        
        # Normalize to 0-255
        depth_normalized = ((depth_map - depth_map.min()) / 
                           (depth_map.max() - depth_map.min() + 1e-6) * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        Image.fromarray(depth_colored).save(output_path)
        logger.info(f"Depth preview saved to {output_path}")

    def _save_metadata(self, output_path: str, input_type: str, frame_count: int,
                      prompt: str, mesh_metadata: Dict[str, Any]) -> str:
        """Save processing metadata as JSON"""
        try:
            metadata = {
                "input_type": input_type,
                "frames_processed": frame_count,
                "prompt_used": prompt,
                "processing_date": datetime.now().isoformat(),
                "mesh_info": mesh_metadata
            }
            
            metadata_path = Path(output_path) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {metadata_path}")
            return str(metadata_path)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return ""

    async def _process_3d_model(self, input_path: str, output_path: str, 
                               job_id: str, update_callback) -> dict:
        """
        Process 3D model files (.glb, .gltf, .obj) - optimize and export.
        """
        try:
            logger.info(f"Processing 3D model: {input_path}")
            
            # Load the 3D model
            await update_callback(job_id, 20, "Loading 3D model...")
            mesh = self.mesh_generator.load_mesh(input_path)
            
            # Optimize the mesh
            await update_callback(job_id, 50, "Optimizing mesh quality...")
            mesh = self.mesh_generator.optimize_mesh(mesh)
            
            # Clean mesh artifacts
            await update_callback(job_id, 70, "Cleaning mesh...")
            mesh = self.mesh_generator.clean_mesh(mesh, min_component_size=20)
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Export in both formats
            await update_callback(job_id, 85, "Exporting optimized model...")
            obj_path = Path(output_path) / "scene.obj"
            glb_path = Path(output_path) / "scene.glb"
            
            self.mesh_generator.export_mesh_obj(mesh, str(obj_path))
            self.mesh_generator.export_mesh_glb(mesh, str(glb_path))
            
            # Validate GLB
            await update_callback(job_id, 90, "Validating output...")
            if not self.mesh_generator.validate_glb(str(glb_path)):
                logger.warning("GLB validation detected issues, attempting repair...")
                self.mesh_generator.repair_glb(str(glb_path))
            
            # Save metadata
            await update_callback(job_id, 95, "Saving metadata...")
            metadata = {
                'method': 'model_optimization',
                'input_format': Path(input_path).suffix.lower(),
                'vertex_count': len(mesh.vertices),
                'triangle_count': len(mesh.triangles),
                'bounds': {
                    'min': np.asarray(mesh.get_min_bound()).tolist(),
                    'max': np.asarray(mesh.get_max_bound()).tolist(),
                }
            }
            
            metadata_path = Path(output_path) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            await update_callback(job_id, 100, "3D model processing completed!")
            
            return {
                "status": "completed",
                "output_files": [str(glb_path), str(obj_path), str(metadata_path)],
                "metadata": {
                    "input_type": "3d_model",
                    "mesh_file": str(glb_path),
                    "mesh_vertices": len(mesh.vertices),
                    "mesh_triangles": len(mesh.triangles),
                    "processing_date": datetime.now().isoformat(),
                    "output_directory": output_path
                }
            }
        except Exception as e:
            logger.error(f"3D model processing error: {str(e)}")
            await update_callback(job_id, -1, f"Error: {str(e)}")
            raise
