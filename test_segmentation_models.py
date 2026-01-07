#!/usr/bin/env python3
"""
Segmentation Model Diagnostic & Tester
Tests which segmentation models can be loaded successfully.

Usage:
    python test_segmentation_models.py
"""

import sys
import torch
from pathlib import Path

def check_system():
    """Check system capabilities."""
    print("=" * 70)
    print("SYSTEM DIAGNOSTICS")
    print("=" * 70)
    
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA Compute Capability: {props.major}.{props.minor}")
    else:
        print("⚠️  CUDA Not Available - Will use CPU (slower)")
    
    # Check transformers
    try:
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed - Install with: pip install transformers")
        return False
    
    # Check cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.glob("**/*") if f.is_file())
        print(f"Hugging Face Cache: {cache_size / 1e9:.2f} GB")
    else:
        print(f"Hugging Face Cache: 0.00 GB (will be created on first download)")
    
    print()
    return True

def test_models():
    """Test loading each model tier."""
    print("=" * 70)
    print("TESTING SEGMENTATION MODELS")
    print("=" * 70)
    print()
    
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    
    # Models in order of preference
    models = [
        {
            "name": "nvidia/segformer-b5-finetuned-ade-640-640",
            "tier": "1 (Best Quality)",
            "size": "1.2 GB",
            "vram": "12 GB",
            "speed": "Very Slow",
        },
        {
            "name": "nvidia/segformer-b4-finetuned-ade-512-512",
            "tier": "2 (Good Balance)",
            "size": "900 MB",
            "vram": "8 GB",
            "speed": "Slow",
        },
        {
            "name": "nvidia/segformer-b2-finetuned-ade-512-512",
            "tier": "3 (Medium)",
            "size": "600 MB",
            "vram": "4 GB",
            "speed": "Medium",
        },
        {
            "name": "nvidia/segformer-b1-finetuned-ade-512-512",
            "tier": "4 (Fast)",
            "size": "500 MB",
            "vram": "2 GB",
            "speed": "Fast",
        },
        {
            "name": "nvidia/segformer-b0-finetuned-ade-512-512",
            "tier": "5 (Fastest)",
            "size": "380 MB",
            "vram": "1 GB",
            "speed": "Very Fast",
        },
    ]
    
    successful_models = []
    failed_models = []
    
    for i, model_info in enumerate(models, 1):
        model_name = model_info["name"]
        print(f"[{i}/5] Testing Tier {model_info['tier']}")
        print(f"      Model: {model_name}")
        print(f"      Download Size: {model_info['size']}")
        print(f"      VRAM Required: {model_info['vram']}")
        print(f"      Speed: {model_info['speed']}")
        
        try:
            print(f"      Loading processor... ", end="", flush=True)
            processor = AutoImageProcessor.from_pretrained(model_name)
            print("✓")
            
            print(f"      Loading model... ", end="", flush=True)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            print("✓")
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            print(f"      Device: {device}")
            
            print(f"      ✅ SUCCESS - Can use this model!")
            successful_models.append(model_name)
            
        except Exception as e:
            error_msg = str(e)
            
            # Categorize error
            if "not a local folder" in error_msg or "not a valid model" in error_msg:
                print(f"      ❌ INVALID MODEL ID (not on Hugging Face)")
            elif "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
                print(f"      ⚠️  OUT OF MEMORY (need {model_info['vram']} VRAM)")
            elif "Connection" in error_msg or "timeout" in error_msg:
                print(f"      ⚠️  NETWORK ERROR (can't download)")
            else:
                print(f"      ❌ FAILED - {error_msg[:80]}")
            
            failed_models.append(model_name)
        
        print()
    
    return successful_models, failed_models

def test_opencv_fallback():
    """Test OpenCV fallback."""
    print("=" * 70)
    print("TESTING OPENCV FALLBACK")
    print("=" * 70)
    print()
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        print("OpenCV Version:", cv2.__version__)
        print("Testing color variance segmentation...", end="", flush=True)
        
        # Create dummy image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test segmentation
        hsv = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        grad_s = cv2.Laplacian(s.astype(np.float32), cv2.CV_32F)
        grad_v = cv2.Laplacian(v.astype(np.float32), cv2.CV_32F)
        var_map = np.abs(grad_s) + np.abs(grad_v)
        
        print(" ✓")
        print("✅ OpenCV fallback is WORKING")
        return True
        
    except Exception as e:
        print(f" ❌ FAILED - {e}")
        return False

def main():
    """Run all diagnostics."""
    print()
    print("🔍 SCENE FORGE SEGMENTATION MODEL DIAGNOSTIC")
    print()
    
    # Check system
    if not check_system():
        print("❌ System check failed - please install required packages")
        return 1
    
    # Test models
    successful, failed = test_models()
    
    # Test fallback
    fallback_ok = test_opencv_fallback()
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    if successful:
        print(f"✅ {len(successful)} model(s) can be loaded:")
        for model in successful:
            size = model.split("-")[-1]
            print(f"   • {model}")
        print()
        print(f"   Best option: {successful[0]}")
    else:
        print(f"❌ No neural models can be loaded")
        print(f"   Reason: Internet connection or insufficient VRAM")
        print()
    
    print(f"Fallback (OpenCV): {'✅ WORKING' if fallback_ok else '❌ FAILED'}")
    print()
    
    if successful:
        print("✅ DIAGNOSIS: System is working well!")
        print("   Your segmentation should work with high quality.")
    elif fallback_ok:
        print("⚠️  DIAGNOSIS: Fallback mode only")
        print("   Reconstruction will work but with lower segmentation quality.")
        print()
        print("OPTIONS TO FIX:")
        print("   1. Check internet connection (for model download)")
        print("   2. Free up disk space (~1.2 GB for largest model)")
        print("   3. Check CUDA/GPU (if using GPU)")
        print("   4. Try smaller models (b0 or b1 use less VRAM)")
    else:
        print("❌ DIAGNOSIS: System needs attention")
        print("   Please check OpenCV installation")
    
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())

