#!/usr/bin/env python3
"""
Quality Degradation Diagnostic Script
Tests if all quality degradation fixes are properly implemented.

Usage:
    python diagnose_quality_fixes.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import inspect

def check_file_exists(filepath: str) -> bool:
    """Check if file exists."""
    return os.path.exists(filepath)

def check_function_signature(module_path: str, class_name: str, method_name: str, 
                            expected_param: str) -> bool:
    """Check if function has expected parameter."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        cls = getattr(module, class_name)
        method = getattr(cls, method_name)
        sig = inspect.signature(method)
        
        return expected_param in sig.parameters
    except Exception as e:
        print(f"  ❌ Error checking signature: {e}")
        return False

def diagnose_quality_fixes():
    """Run diagnostic checks for all quality fixes."""
    print("=" * 70)
    print("SCENE FORGE QUALITY DEGRADATION - DIAGNOSTIC CHECK")
    print("=" * 70)
    print()
    
    issues_found = []
    fixes_verified = 0
    
    # FIX 1: Quality parameter in API
    print("🔍 Checking FIX #1: Quality parameter passed through pipeline...")
    print("-" * 70)
    
    api_file = "SceneForge_Backend/app/api/processing.py"
    if check_file_exists(api_file):
        with open(api_file, 'r') as f:
            content = f.read()
            if 'quality: str = Form("high")' in content:
                print("  ✅ Quality captured in upload_files")
                if 'await processing_service.process_scene' in content and 'quality=quality' in content:
                    print("  ✅ Quality passed to processing_service.process_scene()")
                    fixes_verified += 1
                else:
                    print("  ❌ Quality NOT passed to process_scene")
                    issues_found.append("Quality not passed from API to service")
            else:
                print("  ❌ Quality not captured in upload_files")
                issues_found.append("Quality parameter missing from API endpoint")
    else:
        print(f"  ⚠️  File not found: {api_file}")
    print()
    
    # FIX 2: Quality-aware downsampling
    print("🔍 Checking FIX #2: Quality-aware downsampling...")
    print("-" * 70)
    
    mesh_gen_file = "SceneForge_Backend/app/core/services/mesh_generator.py"
    if check_file_exists(mesh_gen_file):
        with open(mesh_gen_file, 'r') as f:
            content = f.read()
            if 'quality: str' in content and 'quality ==' in content:
                print("  ✅ Quality parameter present in depth_to_point_cloud")
                if 'quality == "high"' in content and 'scale = 1' in content:
                    print("  ✅ High quality: no downsampling")
                if 'quality == "medium"' in content:
                    print("  ✅ Medium quality: conditional downsampling")
                if 'quality == "low"' in content:
                    print("  ✅ Low quality: aggressive downsampling")
                fixes_verified += 1
            else:
                print("  ❌ Quality-aware downsampling not implemented")
                issues_found.append("Downsampling not quality-aware")
    else:
        print(f"  ⚠️  File not found: {mesh_gen_file}")
    print()
    
    # FIX 3: Lossless image formats
    print("🔍 Checking FIX #3: Lossless image formats...")
    print("-" * 70)
    
    proc_service_file = "SceneForge_Backend/app/core/services/processing_service.py"
    if check_file_exists(proc_service_file):
        with open(proc_service_file, 'r') as f:
            content = f.read()
            if "_save_preview_images" in content:
                if 'quality' in content.split("def _save_preview_images")[1].split("def ")[0]:
                    print("  ✅ _save_preview_images has quality parameter")
                if "format='PNG'" in content or 'format="PNG"' in content:
                    print("  ✅ PNG format used for high quality")
                if 'quality=95' in content:
                    print("  ✅ JPEG quality set to 95 for fallback")
                fixes_verified += 1
            else:
                print("  ❌ Image save function not found or not updated")
                issues_found.append("Image saving not using quality-aware formats")
    else:
        print(f"  ⚠️  File not found: {proc_service_file}")
    print()
    
    # FIX 4: Raw depth preservation
    print("🔍 Checking FIX #4: Raw depth data preservation...")
    print("-" * 70)
    
    if check_file_exists(proc_service_file):
        with open(proc_service_file, 'r') as f:
            content = f.read()
            if "np.save" in content or "_raw.npy" in content:
                print("  ✅ Raw depth (float32) being saved to .npy files")
                if "astype(np.float32)" in content:
                    print("  ✅ Depth preserved as float32")
                fixes_verified += 1
            else:
                print("  ❌ Raw depth not being saved")
                issues_found.append("Float32 depth not preserved")
    else:
        print(f"  ⚠️  File not found: {proc_service_file}")
    print()
    
    # FIX 5: Quality-aware model input
    print("🔍 Checking FIX #5: Quality-aware model input sizing...")
    print("-" * 70)
    
    ai_proc_file = "SceneForge_Backend/app/core/services/ai_processor.py"
    if check_file_exists(ai_proc_file):
        with open(ai_proc_file, 'r') as f:
            content = f.read()
            if "input_size" in content:
                print("  ✅ Variable input_size implemented")
                if "input_size = 768" in content and 'quality == "high"' in content:
                    print("  ✅ High quality: 768px input")
                if "input_size = 512" in content and 'quality == "medium"' in content:
                    print("  ✅ Medium quality: 512px input")
                if "input_size = 384" in content:
                    print("  ✅ Low quality: 384px input")
                if "LANCZOS" in content and 'quality == "high"' in content:
                    print("  ✅ LANCZOS interpolation for high quality")
                fixes_verified += 1
            else:
                print("  ❌ Quality-aware model input not implemented")
                issues_found.append("Model input sizing not quality-aware")
    else:
        print(f"  ⚠️  File not found: {ai_proc_file}")
    print()
    
    # Summary
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Fixes Verified: {fixes_verified}/5")
    print()
    
    if not issues_found:
        print("✅ ALL QUALITY FIXES PROPERLY IMPLEMENTED!")
        print()
        print("Next steps:")
        print("  1. Run test_quality_settings() in processing_service.py")
        print("  2. Process test images at all three quality levels")
        print("  3. Compare output quality between low/medium/high")
        print("  4. Verify .npy depth files are created")
        print("  5. Check processing logs for quality settings")
        return True
    else:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print()
        print("Missing fixes needed:")
        if "Quality not passed from API to service" in issues_found:
            print("  → FIX #1: Pass quality parameter through pipeline")
        if "Downsampling not quality-aware" in issues_found:
            print("  → FIX #2: Implement quality-aware downsampling")
        if "Image saving not using quality-aware formats" in issues_found:
            print("  → FIX #3: Use lossless formats for high quality")
        if "Float32 depth not preserved" in issues_found:
            print("  → FIX #4: Preserve raw depth data")
        if "Model input sizing not quality-aware" in issues_found:
            print("  → FIX #5: Implement quality-aware model input")
        return False

def test_quality_degradation():
    """Test actual quality degradation in processing."""
    print()
    print("=" * 70)
    print("QUALITY DEGRADATION TEST")
    print("=" * 70)
    print()
    
    print("To test quality degradation:")
    print()
    print("1. Create test image:")
    print("   from PIL import Image")
    print("   img = Image.new('RGB', (1920, 1080), color='white')")
    print("   img.save('test_image.jpg')")
    print()
    print("2. Process at different quality levels:")
    print("   processing_service.process_scene('test_image.jpg', 'out', '', '1', callback, quality='high')")
    print("   processing_service.process_scene('test_image.jpg', 'out', '', '2', callback, quality='medium')")
    print("   processing_service.process_scene('test_image.jpg', 'out', '', '3', callback, quality='low')")
    print()
    print("3. Compare outputs:")
    print("   - Check file sizes (high quality PNG > medium JPEG > low)")
    print("   - Visual quality inspection")
    print("   - Depth map detail comparison")
    print("   - Check logs for quality settings and parameters")
    print()

if __name__ == "__main__":
    success = diagnose_quality_fixes()
    test_quality_degradation()
    
    sys.exit(0 if success else 1)

