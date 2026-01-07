"""
Semantic segmentation for background removal and object isolation.
Uses AI-based segmentation to extract the main object from background.
"""

import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import cv2

logger = logging.getLogger(__name__)


class SemanticSegmenter:
    """Segment objects from background using semantic segmentation."""
    
    def __init__(self):
        """Initialize segmentation model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load semantic segmentation model with extensive fallback options."""
        logger.info("=" * 70)
        logger.info("SEGMENTATION: Starting model initialization...")
        logger.info("=" * 70)
        
        # List of models to try in order
        # Using VALID models from Hugging Face Hub
        # Verified: These models exist and work with transformers library
        model_candidates = [
            # Tier 1: Best quality models (slower)
            ("nvidia/segformer-b5-finetuned-ade-640-640", "segformer"),      # Largest, best quality
            ("nvidia/segformer-b4-finetuned-ade-512-512", "segformer"),      # Large, good balance
            
            # Tier 2: Medium quality models (balanced)
            ("nvidia/segformer-b2-finetuned-ade-512-512", "segformer"),      # Medium size
            ("nvidia/segformer-b1-finetuned-ade-512-512", "segformer"),      # Smaller, faster
            
            # Tier 3: Fast models (lowest quality)
            ("nvidia/segformer-b0-finetuned-ade-512-512", "segformer"),      # Smallest, fastest
            ("facebook/detr-resnet50-panoptic", "detr"),                      # Alternative fast option
            
            # Tier 4: Fallback (OpenCV-based, no neural network needed)
            (None, "simple_cv2"),  # OpenCV-based fallback
        ]
        
        for model_info in model_candidates:
            try:
                model_name, model_type = model_info
                
                if model_name is None:
                    # Skip neural models, will use CV2
                    continue
                
                logger.info(f"[Tier] Attempting to load: {model_name}")
                
                # Try with different trust settings, using offline mode
                try:
                    self.processor = AutoImageProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=False,
                        local_files_only=True  # Use cached models only
                    )
                    self.model = AutoModelForSemanticSegmentation.from_pretrained(
                        model_name,
                        trust_remote_code=False,
                        local_files_only=True  # Use cached models only
                    )
                except Exception as e1:
                    logger.warning(f"  Failed without trust_remote_code, trying with trust_remote_code=True")
                    self.processor = AutoImageProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=True  # Use cached models only
                    )
                    self.model = AutoModelForSemanticSegmentation.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=True  # Use cached models only
                    )
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"✓ SUCCESS: Loaded segmentation model: {model_name}")
                return  # Success!
                
            except Exception as e:
                error_msg = str(e)[:150]
                logger.warning(f"  ✗ Failed: {type(e).__name__}: {error_msg}")
                continue
        
        # All neural models failed - will use OpenCV fallback
        logger.warning("=" * 70)
        logger.warning("All neural segmentation models failed to load!")
        logger.warning("Falling back to depth-based + color-variance segmentation")
        logger.warning("Quality will be lower but reconstruction will still work")
        logger.warning("=" * 70)
        self.model = None
        self.processor = None
    
    def segment_image(self, rgb_image: np.ndarray, 
                      depth_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment image to identify foreground object with improved quality.
        
        Falls back to depth-based segmentation if neural model fails.
        
        Args:
            rgb_image: RGB image as numpy array (H x W x 3), values 0-255
            depth_map: Optional depth map for better segmentation
        
        Returns:
            Tuple of (segmentation_mask, confidence_scores)
            - segmentation_mask: Binary mask where 1 = foreground, 0 = background
            - confidence_scores: Confidence per pixel
        """
        
        # If no neural model available, use hybrid fallback
        if self.model is None:
            logger.warning("No neural model available, using hybrid fallback segmentation")
            
            # Use combination of depth + color variance
            if depth_map is not None and np.count_nonzero(depth_map) > 0:
                depth_mask = SimpleSegmenter.segment_by_depth(depth_map)
                logger.info(f"  Depth-based mask: {np.count_nonzero(depth_mask)} pixels")
            else:
                depth_mask = None
            
            color_mask = SimpleSegmenter.segment_by_color_variance(rgb_image)
            logger.info(f"  Color-based mask: {np.count_nonzero(color_mask)} pixels")
            
            # Combine masks
            if depth_mask is not None:
                segmentation_mask = np.logical_or(depth_mask, color_mask).astype(np.uint8)
            else:
                segmentation_mask = color_mask
            
            # Post-process
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            logger.info(f"✓ Fallback segmentation complete: {np.count_nonzero(segmentation_mask)} foreground pixels")
            return segmentation_mask, None
        
        try:
            logger.info("Performing neural semantic segmentation...")
            
            # Convert to PIL
            pil_image = Image.fromarray(rgb_image.astype(np.uint8))
            
            # Inference
            with torch.no_grad():
                inputs = self.processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
            
            # Get segmentation logits
            logits = outputs.logits.cpu()
            
            # Upsample logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(rgb_image.shape[0], rgb_image.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            
            # Get predictions
            predictions = upsampled_logits.argmax(dim=1)
            pred_mask = predictions[0].numpy()
            
            # Prefer explicit object classes (better than treating every non-zero label as foreground)
            label_map = getattr(self.model.config, "id2label", {})
            target_terms = {"chair", "sofa", "armchair", "bench", "seat", "cushion", "swivel chair"}
            target_indices = [int(k) for k, v in label_map.items() if any(t in v.lower() for t in target_terms)]

            if target_indices:
                segmentation_mask = np.isin(pred_mask, target_indices).astype(np.uint8)
            else:
                # Fallback to previous behavior if labels unavailable
                segmentation_mask = (pred_mask > 0).astype(np.uint8)

            # Post-process mask for better quality (tighter)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel, iterations=2)

            # If depth map available, prefer components that are closer to camera
            if depth_map is not None and np.count_nonzero(depth_map) > 0 and segmentation_mask.sum() > 0:
                # compute component stats
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(segmentation_mask.astype('uint8'), connectivity=8)
                overall_median = np.median(depth_map[depth_map > 0]) if np.count_nonzero(depth_map) > 0 else None
                best_mask = np.zeros_like(segmentation_mask)

                # consider components except background (label 0)
                H, W = segmentation_mask.shape
                total_pixels = H * W
                # adaptive minimum component area (0.1% of image or at least 500 px)
                min_comp_area = max(500, int(0.001 * total_pixels))
                for label in range(1, num_labels):
                    comp_mask = (labels_im == label)
                    comp_area = stats[label, cv2.CC_STAT_AREA]
                    # ignore tiny components
                    if comp_area < min_comp_area:
                        continue
                    # compute median depth for component
                    comp_depth_vals = depth_map[comp_mask]
                    if comp_depth_vals.size == 0:
                        continue
                    comp_median = np.median(comp_depth_vals)
                    # prefer components noticeably closer than overall median
                    if overall_median is None or comp_median <= (overall_median * 1.05):
                        best_mask = np.logical_or(best_mask, comp_mask)

                # if we found good components, use them, else fall back to largest component
                if best_mask.sum() > 0:
                    segmentation_mask = best_mask.astype(np.uint8)
                else:
                    # choose largest connected component (avoid components that touch image border)
                    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(segmentation_mask.astype('uint8'), connectivity=8)
                    largest_area = 0
                    largest_mask = np.zeros_like(segmentation_mask)
                    H, W = segmentation_mask.shape
                    for label in range(1, num_labels):
                        x = stats[label, cv2.CC_STAT_LEFT]
                        y = stats[label, cv2.CC_STAT_TOP]
                        w_box = stats[label, cv2.CC_STAT_WIDTH]
                        h_box = stats[label, cv2.CC_STAT_HEIGHT]
                        comp_area = stats[label, cv2.CC_STAT_AREA]
                        # skip components that touch border (likely background) unless it's the only reasonable one
                        touches_border = (x == 0 or y == 0 or (x + w_box) >= W or (y + h_box) >= H)
                        if touches_border:
                            # deprioritize border-touching components
                            continue
                        if comp_area > largest_area:
                            largest_area = comp_area
                            largest_mask = (labels_im == label)
                    # If no non-border component found, try removing border-touching components and pick largest
                    if largest_area == 0:
                        # fallback: remove border-touching components from mask
                        non_border_mask = segmentation_mask.copy()
                        for label in range(1, num_labels):
                            x = stats[label, cv2.CC_STAT_LEFT]
                            y = stats[label, cv2.CC_STAT_TOP]
                            w_box = stats[label, cv2.CC_STAT_WIDTH]
                            h_box = stats[label, cv2.CC_STAT_HEIGHT]
                            if x == 0 or y == 0 or (x + w_box) >= W or (y + h_box) >= H:
                                non_border_mask[labels_im == label] = 0
                        # if removing border components reduces coverage sufficiently, use it
                        if non_border_mask.sum() < 0.5 * segmentation_mask.sum() and non_border_mask.sum() > 0:
                            segmentation_mask = non_border_mask.astype(np.uint8)
                        else:
                            # choose largest component even if border-touching
                            largest_area = 0
                            largest_mask = np.zeros_like(segmentation_mask)
                            for label in range(1, num_labels):
                                comp_area = stats[label, cv2.CC_STAT_AREA]
                                if comp_area > largest_area:
                                    largest_area = comp_area
                                    largest_mask = (labels_im == label)
                            if largest_area > 0:
                                segmentation_mask = largest_mask.astype(np.uint8)
                            else:
                                segmentation_mask = (segmentation_mask > 0).astype(np.uint8)

            # Smooth boundaries with dilation then erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            segmentation_mask = cv2.dilate(segmentation_mask, kernel, iterations=1)
            segmentation_mask = cv2.erode(segmentation_mask, kernel, iterations=1)

            logger.info(f"✓ Neural segmentation complete: {np.sum(segmentation_mask)} foreground pixels")

            return segmentation_mask, None
            
        except Exception as e:
            logger.error(f"Neural segmentation failed: {e}")
            logger.warning("Falling back to hybrid depth+color segmentation...")
            
            # Use fallback
            if depth_map is not None and np.count_nonzero(depth_map) > 0:
                depth_mask = SimpleSegmenter.segment_by_depth(depth_map)
            else:
                depth_mask = None
            
            color_mask = SimpleSegmenter.segment_by_color_variance(rgb_image)
            
            if depth_mask is not None:
                segmentation_mask = np.logical_or(depth_mask, color_mask).astype(np.uint8)
            else:
                segmentation_mask = color_mask
            
            return segmentation_mask, None
    
    def apply_mask_to_depth(self, depth_map: np.ndarray, 
                           segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Apply segmentation mask to depth map (zero out background).
        
        Args:
            depth_map: Depth map (H x W)
            segmentation_mask: Binary mask (H x W) where 1 = keep, 0 = remove
        
        Returns:
            Masked depth map
        """
        masked_depth = depth_map.copy()
        masked_depth[segmentation_mask == 0] = 0  # Zero out background
        
        logger.info(f"Masked depth map applied: {np.count_nonzero(masked_depth)} non-zero pixels")
        
        return masked_depth
    
    def apply_mask_to_image(self, rgb_image: np.ndarray,
                           segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Apply segmentation mask to RGB image.
        
        Args:
            rgb_image: RGB image (H x W x 3)
            segmentation_mask: Binary mask (H x W)
        
        Returns:
            Masked RGB image with background removed
        """
        masked_image = rgb_image.copy()
        masked_image[segmentation_mask == 0] = 0  # Zero out background
        
        return masked_image


class SimpleSegmenter:
    """Simple depth-based segmentation when neural model is unavailable."""
    
    @staticmethod
    def segment_by_depth(depth_map: np.ndarray, 
                        percentile: float = 25.0) -> np.ndarray:
        """
        Simple segmentation based on depth values.
        Assumes foreground objects are closer to camera.
        
        Args:
            depth_map: Depth map (H x W)
            percentile: Depth percentile threshold (lower = keep closer objects)
        
        Returns:
            Binary segmentation mask
        """
        logger.info("Using depth-based segmentation")
        
        # Get valid depth values
        valid_depths = depth_map[depth_map > 0]
        
        if len(valid_depths) == 0:
            logger.warning("No valid depth values found")
            return np.ones_like(depth_map, dtype=np.uint8)
        
        # Find threshold depth (closest percentile)
        threshold = np.percentile(valid_depths, percentile)
        
        # Create mask
        mask = ((depth_map > 0) & (depth_map < threshold * 1.5)).astype(np.uint8)
        
        logger.info(f"Depth-based segmentation: threshold={threshold:.4f}, masked pixels={np.count_nonzero(mask)}")
        
        return mask
    
    @staticmethod
    def segment_by_color_variance(rgb_image: np.ndarray) -> np.ndarray:
        """
        Simple color variance-based segmentation.
        Removes uniform-color backgrounds.
        
        Uses OpenCV only (no scikit-image dependency).
        
        Args:
            rgb_image: RGB image (H x W x 3)
        
        Returns:
            Binary segmentation mask
        """
        logger.info("Using color variance segmentation (OpenCV)")
        
        try:
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # Calculate color variation using Laplacian (gradient magnitude)
            # High values = high variation (likely object)
            # Low values = uniform color (likely background)
            
            # Split channels
            h, s, v = cv2.split(hsv)
            
            # Calculate gradients in saturation and value
            grad_s = cv2.Laplacian(s.astype(np.float32), cv2.CV_32F)
            grad_v = cv2.Laplacian(v.astype(np.float32), cv2.CV_32F)
            
            # Combine gradients
            var_map = np.abs(grad_s) + np.abs(grad_v)
            
            # Normalize
            if var_map.max() > 0:
                var_map = var_map / var_map.max() * 255
            
            # Apply blur for smoothing
            var_map = cv2.GaussianBlur(var_map.astype(np.uint8), (5, 5), 1.0)
            
            # Threshold - keep pixels with high variance
            threshold = np.percentile(var_map[var_map > 0], 30) if np.any(var_map > 0) else 50
            mask = (var_map > threshold).astype(np.uint8)
            
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            logger.info(f"Color variance segmentation: threshold={threshold:.1f}, masked pixels={np.count_nonzero(mask)}")
            
            return mask
            
        except Exception as e:
            logger.error(f"Color variance segmentation failed: {e}, using fallback")
            # Fallback: use grayscale edge detection
            gray = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(edges, kernel, iterations=3)
            
            return mask.astype(np.uint8)
