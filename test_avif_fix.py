#!/usr/bin/env python3
"""Test AVIF loading fix"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, 'SceneForge_Backend')

from PIL import Image
import cv2
import numpy as np

def test_avif_loading():
    file_path = r'SceneForge_Backend\uploads\20251210_114510\sofa.jpg'
    
    print(f"Testing AVIF loading from: {file_path}\n")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path)} bytes\n")
    
    # Test 1: PIL (will fail)
    print("Test 1: PIL Image.open()")
    try:
        img = Image.open(file_path)
        print(f"✓ Success: {img.size}, format={img.format}, mode={img.mode}")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    # Test 2: OpenCV (should work)
    print("\nTest 2: OpenCV cv2.imread()")
    try:
        cv_img = cv2.imread(file_path)
        if cv_img is None:
            print(f"✗ Failed: cv2.imread returned None")
        else:
            # Convert BGR to RGB
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            print(f"✓ Success: shape={cv_img_rgb.shape}")
            
            # Convert to PIL
            pil_img = Image.fromarray(cv_img_rgb)
            print(f"  → Converted to PIL: size={pil_img.size}, mode={pil_img.mode}")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    # Test 3: Simulating the fixed code
    print("\nTest 3: Fixed code logic (PIL → fallback to OpenCV)")
    try:
        frames = []
        try:
            img = Image.open(file_path)
            frames = [img]
            print("✓ Loaded via PIL")
        except Exception as pil_error:
            print(f"PIL failed: {str(pil_error)}")
            print("Falling back to OpenCV...")
            cv_img = cv2.imread(file_path)
            if cv_img is None:
                raise Exception(f"OpenCV could not read image")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            frames = [Image.fromarray(cv_img)]
            print(f"✓ Loaded via OpenCV fallback")
        
        if frames:
            frame = frames[0]
            print(f"  → Frame type: {type(frame)}")
            print(f"  → Frame size: {frame.size}")
            print(f"  → Frame mode: {frame.mode}")
            
            # Convert to RGB if needed
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
                print(f"  → Converted to RGB")
            
            print(f"\n✓ FULL SUCCESS: Image is ready for processing!")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")

if __name__ == "__main__":
    test_avif_loading()
