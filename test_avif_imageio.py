#!/usr/bin/env python3
"""Test updated image loading with AVIF support"""

import sys
sys.path.insert(0, 'SceneForge_Backend')

from PIL import Image
import imageio.v2 as iio
import os

file_path = r'SceneForge_Backend\uploads\20251210_114510\sofa.jpg'

print("Testing updated image loading logic\n")
print(f"File: {file_path}")
print(f"Exists: {os.path.exists(file_path)}\n")

frames = []

# Step 1: Try PIL
print("Step 1: Try PIL Image.open()")
try:
    img = Image.open(file_path)
    frames = [img]
    print(f"✓ PIL succeeded")
except Exception as pil_error:
    print(f"✗ PIL failed: {str(pil_error)}")
    
    # Step 2: Try imageio
    if not frames:
        print("\nStep 2: Try imageio (PyAV for AVIF)")
        try:
            img_array = iio.imread(file_path)
            frames = [Image.fromarray(img_array)]
            print(f"✓ imageio succeeded - loaded shape {img_array.shape}")
        except Exception as imageio_error:
            print(f"✗ imageio failed: {str(imageio_error)}")

if frames:
    frame = frames[0]
    print(f"\n✓ SUCCESS! Image loaded.")
    print(f"  Type: {type(frame)}")
    print(f"  Size: {frame.size}")
    print(f"  Mode: {frame.mode}")
    
    # Convert to RGB if needed
    if frame.mode != 'RGB':
        frame = frame.convert('RGB')
        print(f"  Converted to RGB")
    
    import numpy as np
    rgb_array = np.array(frame)
    print(f"  NumPy shape: {rgb_array.shape}")
    print(f"\n✓ Image is ready for processing!")
else:
    print(f"\n✗ FAILED: Could not load image with any method")
