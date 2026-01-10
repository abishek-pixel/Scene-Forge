import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image
import cv2
from stl import mesh as stl_mesh
import meshio
from matplotlib import pyplot as plt
from skimage import measure
import asyncio
import json
from datetime import datetime

# Lazy imports for heavy dependencies
def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

def _import_transformers():
    try:
        from transformers import pipeline
        from huggingface_hub import hf_hub_download
        return pipeline, hf_hub_download
    except ImportError:
        return None, None

def _import_sam():
    try:
        from segment_anything import sam_model_registry, SamPredictor
        return sam_model_registry, SamPredictor
    except ImportError:
        return None, None

from .mesh_generator import MeshGenerator

class AIProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize mesh generator
        self.mesh_generator = MeshGenerator()
        
        # Initialize all models
        self.models = {}
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all AI models"""
        import traceback
        
        # 1. MiDaS for depth estimation (CRITICAL - must succeed)
        try:
            print("Loading MiDaS depth estimation model...")
            print(f"Device: {self.device}")
            
            # Load the model from torch hub
            print("Downloading/loading DPT_Large model from torch hub...")
            self.models['depth'] = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            
            print("Moving model to device...")
            self.models['depth'].to(self.device)
            
            print("Setting model to evaluation mode...")
            self.models['depth'].eval()
            
            print("✓ MiDaS loaded successfully")
            
        except Exception as e:
            print(f"✗ CRITICAL ERROR initializing MiDaS depth model: {e}")
            traceback.print_exc()
            self.models['depth'] = None
            raise Exception(f"Failed to load depth model: {e}")
        
        # 2. Segment Anything Model (SAM) for precise segmentation (OPTIONAL)
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.models['sam'] = None  # Initialize as None
        try:
            if os.path.exists(sam_checkpoint):
                print(f"Loading SAM model from {sam_checkpoint}...")
                sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
                sam.to(self.device)
                self.models['sam'] = SamPredictor(sam)
                print("✓ SAM loaded successfully")
            else:
                print("Note: SAM checkpoint not found - object segmentation disabled")
        except Exception as e:
            print(f"Warning: Could not load SAM model: {e}")
            traceback.print_exc()
            self.models['sam'] = None
        
        print("✓ Model initialization completed")
        
    async def process_frame(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single frame through all necessary models"""
        try:
            print(f"\n=== Processing Frame ===")
            print(f"Input type: {type(image)}, Image info: {image if isinstance(image, Image.Image) else 'numpy array'}")
            
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype('uint8'))
                else:
                    raise TypeError(f"Unexpected image type: {type(image)}")
            
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                print(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            print(f"Image after preprocessing - Mode: {image.mode}, Size: {image.size}")
            
            # 1. Depth Estimation
            print("Starting depth estimation...")
            depth_map = await self._generate_depth_map(image)
            print(f"Depth map generated - shape: {depth_map.shape}")
            
            # 2. Segmentation (optional, can fail gracefully)
            try:
                print("Starting segmentation...")
                image_np = np.array(image)
                segments = await self._segment_objects(image_np)
                print(f"Segmentation completed - {len(segments.get('masks', []))} masks")
            except Exception as e:
                print(f"Warning: Segmentation failed: {e}")
                import traceback
                traceback.print_exc()
                segments = {'masks': []}
            
            # 3. Generate initial 3D mesh
            print("Generating mesh...")
            mesh_data = await self._generate_mesh(depth_map, segments)
            print(f"Mesh generated - vertices: {len(mesh_data.get('vertices', []))}")
            
            # 4. Generate textures
            print("Generating textures...")
            textures = await self._generate_textures(image, mesh_data)
            print(f"Textures generated - {len(textures.get('vertex_colors', []))} colors")
            
            result = {
                'depth_map': depth_map.tolist() if isinstance(depth_map, np.ndarray) else depth_map,
                'segments': segments,
                'mesh': mesh_data,
                'textures': textures
            }
            print("=== Frame Processing Complete ===\n")
            return result
        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    async def _generate_depth_map(self, image: Image.Image) -> np.ndarray:
        """Generate depth map using MiDaS"""
        if 'depth' not in self.models or self.models['depth'] is None:
            raise Exception("Depth model not loaded")
        
        try:
            # Ensure image is RGB PIL Image
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_width, image_height = image.size  # PIL returns (width, height)
            else:
                # Handle numpy array
                image = Image.fromarray(image.astype('uint8'))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_width, image_height = image.size
            
            print(f"Image info - Mode: {image.mode}, Size: {image.size}, Type: {type(image)}")
            
            # Convert PIL Image to numpy array first
            image_np = np.array(image)
            print(f"Image numpy shape: {image_np.shape}, dtype: {image_np.dtype}")
            
            # Preprocess image for MiDaS using numpy array
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            
            # Create the transform pipeline
            # Use larger resize to preserve more spatial detail for depth estimation
            resize = Resize(512, interpolation=Image.BILINEAR)
            to_tensor = ToTensor()
            normalize = Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
            
            # Apply preprocessing
            image_resized = resize(image)
            input_tensor = to_tensor(image_resized)
            input_batch = normalize(input_tensor).unsqueeze(0).to(self.device)
            
            print(f"Input batch shape: {input_batch.shape}, dtype: {input_batch.dtype}")
            
            with torch.no_grad():
                prediction = self.models['depth'](input_batch)
                print(f"Prediction shape before interpolate: {prediction.shape}")
                
                # Interpolate to original size: (height, width)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(image_height, image_width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                print(f"Prediction shape after interpolate: {prediction.shape}")
                
            depth_map = prediction.cpu().numpy().astype('float32')
            print(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")

            # Post-process depth: median + gaussian smoothing to reduce noise
            try:
                import cv2
                # Normalize temporarily for filtering stability
                dmin, dmax = float(depth_map.min()), float(depth_map.max())
                if dmax > dmin:
                    dn = (depth_map - dmin) / (dmax - dmin)
                else:
                    dn = depth_map.copy()

                # median to remove salt-and-pepper
                dn = cv2.medianBlur((dn * 255).astype('uint8'), 5).astype('float32') / 255.0
                # gaussian for smoothness
                dn = cv2.GaussianBlur(dn, (5, 5), 0)

                # restore original scale
                if dmax > dmin:
                    depth_map = dn * (dmax - dmin) + dmin
                else:
                    depth_map = dn
                depth_map = depth_map.astype('float32')
            except Exception:
                # If cv2 not available or fails, continue with raw depth_map
                pass

            return depth_map
        except Exception as e:
            print(f"Error generating depth map: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    async def _segment_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Segment objects using SAM"""
        if 'sam' not in self.models or self.models['sam'] is None:
            # SAM not available, return empty segments
            return {'masks': []}
        
        try:
            self.models['sam'].set_image(image)
            
            # Generate automatic masks
            masks = []
            pred_masks = self.models['sam'].generate()
            
            # Convert masks to list format
            for mask in pred_masks:
                masks.append(mask.tolist())
            
            return {'masks': masks}
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return {'masks': []}
        
    async def _generate_mesh(self, depth_map: np.ndarray, segments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D mesh using depth and segmentation"""
        try:
            print(f"Mesh generation input - depth_map shape: {depth_map.shape}, dtype: {depth_map.dtype}")
            
            # Normalize depth map
            depth_min, depth_max = float(depth_map.min()), float(depth_map.max())
            print(f"Depth range: {depth_min} to {depth_max}")
            
            # Avoid division by zero
            if depth_min == depth_max:
                print("Warning: Depth map is flat, creating simple plane mesh")
                depth_normalized = np.zeros_like(depth_map, dtype=np.float32)
            else:
                depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
            
            print(f"Normalized depth shape: {depth_normalized.shape}")
            
            # Create a 3D volume from depth map
            try:
                volume = np.zeros((depth_normalized.shape[0], depth_normalized.shape[1], 50), dtype=np.uint8)
                for i in range(volume.shape[2]):
                    volume[:, :, i] = (depth_normalized > (i / 50.0)).astype(np.uint8)
                
                print(f"Volume created: {volume.shape}, non-zero voxels: {np.count_nonzero(volume)}")
                
                # Extract surface mesh using marching cubes
                try:
                    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)
                    print(f"Marching cubes result - vertices: {len(verts)}, faces: {len(faces)}")
                except ValueError as e:
                    print(f"Marching cubes failed (likely no valid surface): {e}")
                    # Fallback: create a simple point cloud mesh
                    h, w = depth_normalized.shape
                    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                    verts = np.stack([x_coords.flatten(), y_coords.flatten(), 
                                     (depth_normalized * 10).flatten()], axis=1)  # Scale Z
                    
                    # Create simple faces from grid
                    faces = []
                    for i in range(h - 1):
                        for j in range(w - 1):
                            v0 = i * w + j
                            v1 = i * w + (j + 1)
                            v2 = (i + 1) * w + j
                            v3 = (i + 1) * w + (j + 1)
                            faces.append([v0, v1, v2])
                            faces.append([v1, v3, v2])
                    
                    verts = verts.astype(np.float32)
                    faces = np.array(faces, dtype=np.int32)
                    normals = np.zeros_like(verts)
                    print(f"Fallback mesh created - vertices: {len(verts)}, faces: {len(faces)}")
                
                # Scale vertices back to original depth range
                if len(verts) > 0:
                    verts[:, 2] = verts[:, 2] * (depth_max - depth_min) + depth_min
                
                # Convert to dictionary format
                mesh_dict = {
                    'vertices': verts.tolist() if len(verts) > 0 else [],
                    'faces': faces.tolist() if len(faces) > 0 else [],
                    'normals': normals.tolist() if len(normals) > 0 else []
                }
                
                print(f"Final mesh - vertices: {len(mesh_dict['vertices'])}, faces: {len(mesh_dict['faces'])}")
                return mesh_dict
                
            except Exception as e:
                print(f"Error in volume processing: {e}")
                import traceback
                traceback.print_exc()
                raise
                
        except Exception as e:
            print(f"Error generating mesh: {e}")
            import traceback
            traceback.print_exc()
            # Return empty mesh
            return {
                'vertices': [],
                'faces': [],
                'normals': []
            }
        
    async def _generate_textures(self, image: Image.Image, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic UV mapping and textures"""
        try:
            # Ensure image is in proper format
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_np = np.array(image)
            else:
                image_np = image
            
            # Create a simple UV mapping based on vertex positions
            vertices = np.array(mesh_data.get('vertices', []))
            
            if len(vertices) == 0:
                print("Warning: No vertices in mesh data")
                return {'uv_coords': [], 'vertex_colors': []}
            
            # Normalize vertex coordinates to [0, 1] range for UV mapping
            uv_coords = vertices[:, :2].copy()
            
            # Handle case where all vertices have same coordinate
            if uv_coords[:, 0].max() == uv_coords[:, 0].min():
                uv_coords[:, 0] = 0.5
            else:
                uv_coords[:, 0] = (uv_coords[:, 0] - uv_coords[:, 0].min()) / (uv_coords[:, 0].max() - uv_coords[:, 0].min())
            
            if uv_coords[:, 1].max() == uv_coords[:, 1].min():
                uv_coords[:, 1] = 0.5
            else:
                uv_coords[:, 1] = (uv_coords[:, 1] - uv_coords[:, 1].min()) / (uv_coords[:, 1].max() - uv_coords[:, 1].min())
            
            # Sample colors from the original image using UV coordinates
            image_h, image_w = image_np.shape[:2]
            pixel_coords = np.stack([
                (uv_coords[:, 0] * (image_w - 1)).astype(int),
                (uv_coords[:, 1] * (image_h - 1)).astype(int)
            ], axis=1)
            
            # Clamp coordinates to valid range
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_w - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_h - 1)
            
            vertex_colors = image_np[pixel_coords[:, 1], pixel_coords[:, 0]]
            
            return {
                'uv_coords': uv_coords.tolist(),
                'vertex_colors': vertex_colors.tolist()
            }
        except Exception as e:
            print(f"Error generating textures: {e}")
            import traceback
            traceback.print_exc()
            return {'uv_coords': [], 'vertex_colors': []}
        
    def export_mesh(self, mesh_data: Dict[str, Any], output_path: str):
        """Export the mesh to various 3D file formats"""
        vertices = np.array(mesh_data['vertices'])
        faces = np.array(mesh_data['faces'])
        
        # Create meshio mesh
        mesh = meshio.Mesh(
            points=vertices,
            cells=[("triangle", faces)]
        )
        
        # Export to different formats
        meshio.write(output_path, mesh)
    
    async def process_with_prompt(self, prompt: str, base_mesh: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Process text prompt for 3D generation or modification"""
        # Placeholder - text-to-3D features can be implemented later
        print(f"Prompt processing requested: {prompt}")
        return None