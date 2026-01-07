"""
Quality Diagnostics Tool
Analyzes current reconstruction quality and identifies bottlenecks.

Usage:
    python diagnose_quality.py --video <path> --output <dir>
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityDiagnostics:
    """Comprehensive quality diagnostics for 3D reconstruction."""
    
    def __init__(self, output_dir: str = "diagnostics_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {}
    
    def diagnose_video_quality(self, video_path: str) -> Dict:
        """
        Analyze video file for reconstruction suitability.
        
        Checks:
        - Duration and frame count
        - Resolution
        - Frame rate (FPS)
        - Motion quality (jitter)
        - Lighting consistency
        - Focus quality (sharpness)
        - Background complexity
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of video quality metrics
        """
        
        logger.info("=" * 60)
        logger.info("VIDEO QUALITY DIAGNOSTICS")
        logger.info("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return {}
        
        metrics = {}
        
        # Basic video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0
        
        logger.info(f"\n📹 VIDEO PROPERTIES")
        logger.info(f"  Duration:     {duration_sec:.1f} seconds")
        logger.info(f"  Frames:       {frame_count}")
        logger.info(f"  FPS:          {fps:.1f}")
        logger.info(f"  Resolution:   {width}x{height}")
        logger.info(f"  Format:       {width}x{height}@{fps:.0f}fps")
        
        metrics['duration_sec'] = duration_sec
        metrics['frame_count'] = frame_count
        metrics['fps'] = fps
        metrics['resolution'] = (width, height)
        
        # Analyze frame quality
        logger.info(f"\n📊 ANALYZING {frame_count} FRAMES...")
        
        sharpness_scores = []
        brightness_values = []
        motion_magnitudes = []
        prev_frame = None
        
        sample_rate = max(1, frame_count // 20)  # Sample ~20 frames
        
        for frame_idx in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(sharpness)
            
            # Brightness
            brightness = np.mean(gray)
            brightness_values.append(brightness)
            
            # Motion (frame difference)
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                motion_magnitudes.append(motion)
            
            prev_frame = gray
        
        # Analyze results
        sharpness_mean = np.mean(sharpness_scores)
        sharpness_std = np.std(sharpness_scores)
        brightness_mean = np.mean(brightness_values)
        brightness_std = np.std(brightness_values)
        motion_mean = np.mean(motion_magnitudes) if motion_magnitudes else 0
        
        logger.info(f"\n🔍 QUALITY METRICS")
        logger.info(f"  Sharpness:    {sharpness_mean:.1f} ± {sharpness_std:.1f}")
        logger.info(f"  Brightness:   {brightness_mean:.1f} ± {brightness_std:.1f}/255")
        logger.info(f"  Motion:       {motion_mean:.2f} pixels/frame")
        
        # Scoring
        logger.info(f"\n✅ QUALITY SCORES")
        
        # Duration score
        if duration_sec < 10:
            duration_score = 1
            logger.info(f"  Duration:     ⚠️  {duration_sec:.1f}s (TOO SHORT - need 15-30s)")
        elif duration_sec < 15:
            duration_score = 2
            logger.info(f"  Duration:     ⚠️  {duration_sec:.1f}s (SHORT - 15-30s better)")
        elif duration_sec < 60:
            duration_score = 4
            logger.info(f"  Duration:     ✓ {duration_sec:.1f}s (GOOD)")
        else:
            duration_score = 5
            logger.info(f"  Duration:     ✓✓ {duration_sec:.1f}s (EXCELLENT)")
        
        # Resolution score
        pixels = width * height
        if pixels < 720*480:
            resolution_score = 2
            logger.info(f"  Resolution:   ⚠️  {width}x{height} (LOW - use 1080p+)")
        elif pixels < 1920*1080:
            resolution_score = 3
            logger.info(f"  Resolution:   ✓ {width}x{height} (OK - 1080p better)")
        else:
            resolution_score = 5
            logger.info(f"  Resolution:   ✓✓ {width}x{height} (EXCELLENT)")
        
        # FPS score
        if fps < 25:
            fps_score = 2
            logger.info(f"  FPS:          ⚠️  {fps:.1f} (LOW - use 30fps)")
        elif fps < 30:
            fps_score = 3
            logger.info(f"  FPS:          ✓ {fps:.1f} (OK - 30fps better)")
        else:
            fps_score = 5
            logger.info(f"  FPS:          ✓✓ {fps:.1f}fps (EXCELLENT)")
        
        # Sharpness score
        if sharpness_mean < 100:
            sharpness_score = 1
            logger.info(f"  Sharpness:    ✗ {sharpness_mean:.1f} (BLURRY - improve focus)")
        elif sharpness_mean < 500:
            sharpness_score = 2
            logger.info(f"  Sharpness:    ⚠️  {sharpness_mean:.1f} (SOFT - check focus)")
        elif sharpness_mean < 2000:
            sharpness_score = 3
            logger.info(f"  Sharpness:    ✓ {sharpness_mean:.1f} (FAIR - ok)")
        else:
            sharpness_score = 5
            logger.info(f"  Sharpness:    ✓✓ {sharpness_mean:.1f} (EXCELLENT - sharp)")
        
        # Lighting score
        if brightness_mean < 50 or brightness_mean > 230:
            lighting_score = 1
            logger.info(f"  Lighting:     ✗ TOO DARK or TOO BRIGHT (brightness {brightness_mean:.0f})")
        elif brightness_std > 50:
            lighting_score = 2
            logger.info(f"  Lighting:     ⚠️  INCONSISTENT (variance {brightness_std:.1f})")
        elif brightness_std > 30:
            lighting_score = 3
            logger.info(f"  Lighting:     ✓ FAIR (variance {brightness_std:.1f})")
        else:
            lighting_score = 5
            logger.info(f"  Lighting:     ✓✓ EXCELLENT (smooth, variance {brightness_std:.1f})")
        
        # Motion score
        if motion_mean > 10:
            motion_score = 1
            logger.info(f"  Motion:       ✗ TOO FAST ({motion_mean:.2f} px/frame - move slower)")
        elif motion_mean > 5:
            motion_score = 2
            logger.info(f"  Motion:       ⚠️  FAST ({motion_mean:.2f} px/frame - slow down)")
        elif motion_mean > 2:
            motion_score = 3
            logger.info(f"  Motion:       ✓ GOOD ({motion_mean:.2f} px/frame)")
        else:
            motion_score = 5
            logger.info(f"  Motion:       ✓✓ EXCELLENT smooth ({motion_mean:.2f} px/frame)")
        
        # Overall score
        overall_score = (duration_score + resolution_score + fps_score + 
                        sharpness_score + lighting_score + motion_score) / 6
        
        logger.info(f"\n🎯 OVERALL VIDEO QUALITY SCORE: {overall_score:.1f}/5.0")
        
        if overall_score < 2:
            logger.info("   ✗✗ POOR - Video quality needs significant improvement")
            recommendation = "POOR"
        elif overall_score < 3:
            logger.info("   ⚠️  FAIR - Video quality could be better")
            recommendation = "FAIR"
        elif overall_score < 4:
            logger.info("   ✓ GOOD - Video quality acceptable")
            recommendation = "GOOD"
        else:
            logger.info("   ✓✓ EXCELLENT - Video quality is great")
            recommendation = "EXCELLENT"
        
        metrics['sharpness'] = sharpness_mean
        metrics['brightness'] = brightness_mean
        metrics['brightness_std'] = brightness_std
        metrics['motion'] = motion_mean
        metrics['overall_score'] = overall_score
        metrics['recommendation'] = recommendation
        
        cap.release()
        
        # Save analysis to file
        self._save_video_analysis(metrics, video_path)
        
        return metrics
    
    def diagnose_depth_quality(self, depth_map: np.ndarray) -> Dict:
        """
        Analyze depth map quality.
        
        Args:
            depth_map: Depth map array
            
        Returns:
            Dictionary of depth metrics
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("DEPTH MAP QUALITY DIAGNOSTICS")
        logger.info("=" * 60)
        
        metrics = {}
        
        # Basic stats
        valid_mask = depth_map > 0
        valid_depths = depth_map[valid_mask]
        
        if len(valid_depths) == 0:
            logger.warning("No valid depth values!")
            return metrics
        
        coverage = np.sum(valid_mask) / depth_map.size * 100
        
        logger.info(f"\n📊 DEPTH STATISTICS")
        logger.info(f"  Coverage:     {coverage:.1f}%")
        logger.info(f"  Min depth:    {valid_depths.min():.3f}")
        logger.info(f"  Max depth:    {valid_depths.max():.3f}")
        logger.info(f"  Mean depth:   {valid_depths.mean():.3f}")
        logger.info(f"  Std dev:      {valid_depths.std():.3f}")
        
        # Artifact detection
        logger.info(f"\n🔍 ARTIFACT DETECTION")
        
        # Holes (zero depth)
        hole_ratio = (depth_map == 0).sum() / depth_map.size * 100
        logger.info(f"  Holes:        {hole_ratio:.1f}%", end="")
        if hole_ratio > 20:
            logger.info(" ⚠️  (TOO MANY - refinement needed)")
        else:
            logger.info(" ✓")
        
        # Noise (high variance)
        laplacian = cv2.Laplacian(depth_map.astype(np.float32), cv2.CV_32F)
        noise_level = laplacian.std()
        logger.info(f"  Noise level:  {noise_level:.3f}", end="")
        if noise_level > 0.5:
            logger.info(" ⚠️  (NOISY - refinement recommended)")
        else:
            logger.info(" ✓")
        
        # Outliers (statistical)
        median = np.median(valid_depths)
        q1 = np.percentile(valid_depths, 25)
        q3 = np.percentile(valid_depths, 75)
        iqr = q3 - q1
        outlier_mask = (valid_depths < q1 - 1.5*iqr) | (valid_depths > q3 + 1.5*iqr)
        outlier_ratio = outlier_mask.sum() / len(valid_depths) * 100
        logger.info(f"  Outliers:     {outlier_ratio:.1f}%", end="")
        if outlier_ratio > 10:
            logger.info(" ⚠️  (MANY - improve depth estimation)")
        else:
            logger.info(" ✓")
        
        metrics['coverage'] = coverage
        metrics['noise_level'] = noise_level
        metrics['outlier_ratio'] = outlier_ratio
        
        return metrics
    
    def diagnose_segmentation_quality(self, mask: np.ndarray, rgb_image: np.ndarray) -> Dict:
        """
        Analyze segmentation mask quality.
        
        Args:
            mask: Binary segmentation mask
            rgb_image: Original RGB image
            
        Returns:
            Dictionary of segmentation metrics
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("SEGMENTATION QUALITY DIAGNOSTICS")
        logger.info("=" * 60)
        
        metrics = {}
        
        # Coverage
        object_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        object_ratio = object_pixels / total_pixels * 100
        
        logger.info(f"\n📊 SEGMENTATION STATISTICS")
        logger.info(f"  Object pixels: {object_ratio:.1f}%")
        
        if object_ratio < 5:
            logger.info("    ⚠️  TOO SMALL - may be only detecting part of object")
        elif object_ratio > 80:
            logger.info("    ⚠️  TOO LARGE - may be including background")
        else:
            logger.info("    ✓ REASONABLE")
        
        # Boundary quality
        logger.info(f"\n🔍 BOUNDARY QUALITY")
        
        # Edges in mask
        edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
        edge_roughness = edges.sum() / object_pixels if object_pixels > 0 else 0
        logger.info(f"  Edge roughness: {edge_roughness:.3f}", end="")
        if edge_roughness > 0.1:
            logger.info(" ⚠️  (ROUGH - smooth boundaries)")
        else:
            logger.info(" ✓")
        
        # Connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        logger.info(f"  Components:    {num_labels - 1} (excluding background)")
        if num_labels > 3:
            logger.info("    ⚠️  FRAGMENTED - object appears disconnected")
        elif num_labels == 2:
            logger.info("    ✓ CLEAN - single connected object")
        
        metrics['object_ratio'] = object_ratio
        metrics['edge_roughness'] = edge_roughness
        metrics['components'] = num_labels - 1
        
        return metrics
    
    def generate_report(self, video_metrics: Dict, depth_metrics: Dict, 
                       seg_metrics: Dict) -> str:
        """
        Generate comprehensive quality report.
        
        Returns:
            Formatted report string
        """
        
        report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    3D RECONSTRUCTION QUALITY REPORT                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

📹 VIDEO QUALITY
"""
        
        if video_metrics:
            report += f"""
  Duration:        {video_metrics.get('duration_sec', 0):.1f} seconds
  Resolution:      {video_metrics.get('resolution', (0, 0))[0]}x{video_metrics.get('resolution', (0, 0))[1]}
  FPS:             {video_metrics.get('fps', 0):.1f}
  Sharpness:       {video_metrics.get('sharpness', 0):.1f}
  Brightness:      {video_metrics.get('brightness', 0):.1f} ± {video_metrics.get('brightness_std', 0):.1f}
  Motion Smoothness: {video_metrics.get('motion', 0):.2f} pixels/frame
  
  Overall Score:   {video_metrics.get('overall_score', 0):.1f}/5.0 ({video_metrics.get('recommendation', 'UNKNOWN')})
"""
        
        if depth_metrics:
            report += f"""
📊 DEPTH QUALITY
  
  Coverage:        {depth_metrics.get('coverage', 0):.1f}%
  Noise Level:     {depth_metrics.get('noise_level', 0):.3f}
  Outliers:        {depth_metrics.get('outlier_ratio', 0):.1f}%
"""
        
        if seg_metrics:
            report += f"""
🎯 SEGMENTATION QUALITY
  
  Object Size:     {seg_metrics.get('object_ratio', 0):.1f}%
  Edge Roughness:  {seg_metrics.get('edge_roughness', 0):.3f}
  Components:      {seg_metrics.get('components', 0)}
"""
        
        report += """
═══════════════════════════════════════════════════════════════════════════════

🎯 RECOMMENDATIONS FOR IMPROVEMENT

Priority 1 (MOST CRITICAL):
  ✓ Re-record video with optimal settings:
    - Duration: 20-30 seconds (full 360° rotation)
    - Resolution: 1080p or 4K
    - FPS: 30fps constant
    - Lighting: Diffuse, even, bright (500+ lux)
    - Background: Plain white/gray
    - Motion: Smooth, consistent speed

Priority 2 (HIGH IMPACT):
  ✓ Apply depth refinement (reduces noise, fills holes)
  ✓ Implement multi-pass TSDF (improves geometry)
  ✓ Add GrabCut segmentation refinement

Priority 3 (MEDIUM IMPACT):
  ✓ Enhance normal estimation
  ✓ Apply texture refinement
  ✓ Tune TSDF parameters per object

═══════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def _save_video_analysis(self, metrics: Dict, video_path: str):
        """Save analysis results to file."""
        output_file = self.output_dir / "video_analysis.txt"
        with open(output_file, 'w') as f:
            f.write(f"Video: {video_path}\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"\n✓ Analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Quality Diagnostics for 3D Reconstruction')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='diagnostics_output', help='Output directory')
    
    args = parser.parse_args()
    
    diagnostics = QualityDiagnostics(output_dir=args.output)
    
    # Analyze video
    video_metrics = diagnostics.diagnose_video_quality(args.video)
    
    # Generate report
    report = diagnostics.generate_report(video_metrics, {}, {})
    print(report)
    
    # Save report
    report_file = Path(args.output) / "quality_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
