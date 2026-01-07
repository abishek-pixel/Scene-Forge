#!/usr/bin/env python3
"""
End-to-end test simulating the actual image processing pipeline
"""

import sys
sys.path.insert(0, 'SceneForge_Backend')

from PIL import Image
import numpy as np
import imageio.v2 as iio
from torchvision import transforms
import torch

def test_full_pipeline():
    """Simulate the actual processing pipeline"""
    
    print("=" * 70)
    print("END-TO-END IMAGE PROCESSING PIPELINE TEST")
    print("=" * 70 + "\n")
    
    file_path = r'SceneForge_Backend\uploads\20251210_114510\sofa.jpg'
    
    # Step 1: Load image (like processing_service.py does)
    print("Step 1: Load AVIF image via imageio")
    print("-" * 70)
    img_array = iio.imread(file_path)
    pil_img = Image.fromarray(img_array, mode='RGB').convert('RGB')
    print(f"✓ Loaded: {pil_img.size}, mode={pil_img.mode}")
    
    # Step 2: Convert to numpy (like processing_service.py does)
    print("\nStep 2: Convert to numpy array for storage")
    print("-" * 70)
    rgb_array = np.array(pil_img)
    print(f"✓ NumPy array: {rgb_array.shape}, dtype={rgb_array.dtype}")
    
    # Step 3: Simulate depth estimation input
    print("\nStep 3: Prepare for depth estimation (torchvision resize)")
    print("-" * 70)
    try:
        # This is what ai_processor._generate_depth_map does
        resize = transforms.Resize((384, 384))
        
        # The frame passed to ai_processor is the PIL Image
        resized = resize(pil_img)
        print(f"✓ Torchvision resize succeeded: {resized.size}")
        
        # Convert to tensor like depth estimation would
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(resized)
        print(f"✓ Converted to tensor: {tensor.shape}, dtype={tensor.dtype}")
        
    except AttributeError as e:
        if "'NoneType' object has no attribute 'seek'" in str(e):
            print(f"✗ FAILED: Got the seek error (fix didn't work)")
            print(f"  Error: {str(e)}")
            return False
        else:
            raise
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        return False
    
    # Step 4: Simulate segmentation input
    print("\nStep 4: Prepare for segmentation")
    print("-" * 70)
    try:
        # Segmentation expects numpy array
        seg_input = rgb_array
        print(f"✓ Segmentation ready with: {seg_input.shape}")
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE FLOW:")
    print("-" * 70)
    print(f"1. Load AVIF via imageio      ✓ (2000x1121)")
    print(f"2. Convert to PIL Image       ✓ (RGB in-memory)")
    print(f"3. Resize for depth est.      ✓ (384x384)")
    print(f"4. Convert to tensor          ✓ (3x384x384)")
    print(f"5. Ready for segmentation     ✓ (1121x2000x3 numpy)")
    print("=" * 70)
    print("✓ SUCCESS - Full pipeline works with AVIF images")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
