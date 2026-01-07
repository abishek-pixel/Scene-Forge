#!/usr/bin/env python3
"""
Test that PIL images can be resized by torchvision after being loaded via imageio
"""

import sys
sys.path.insert(0, 'SceneForge_Backend')

from PIL import Image
import numpy as np
import imageio.v2 as iio
from torchvision import transforms

def test_imageio_resize():
    """Test the problematic resize operation that was failing"""
    
    print("=" * 70)
    print("PIL IMAGE RESIZE TEST (torchvision compatibility)")
    print("=" * 70 + "\n")
    
    file_path = r'SceneForge_Backend\uploads\20251210_114510\sofa.jpg'
    
    print("Step 1: Load AVIF via imageio")
    print("-" * 70)
    img_array = iio.imread(file_path)
    print(f"✓ Loaded array: shape {img_array.shape}, dtype {img_array.dtype}")
    
    print("\nStep 2: Convert to PIL Image")
    print("-" * 70)
    pil_img = Image.fromarray(img_array, mode='RGB')
    print(f"✓ Created PIL Image: {pil_img.size}, mode {pil_img.mode}")
    
    print("\nStep 3: Convert to in-memory RGB (important!)")
    print("-" * 70)
    pil_img = pil_img.convert('RGB')
    print(f"✓ Converted to RGB: {pil_img.size}, mode {pil_img.mode}")
    
    print("\nStep 4: Try to resize with torchvision (the failing operation)")
    print("-" * 70)
    try:
        # This is what was failing with NoneType seek error
        resize_transform = transforms.Resize((384, 384))
        resized = resize_transform(pil_img)
        print(f"✓ Resize succeeded: {resized.size}")
    except AttributeError as e:
        if "'NoneType' object has no attribute 'seek'" in str(e):
            print(f"✗ Got the seek error: {str(e)}")
            print("  This means PIL image is still file-backed")
            return False
        else:
            raise
    except Exception as e:
        print(f"✗ Resize failed with different error: {str(e)}")
        return False
    
    print("\nStep 5: Verify resized image can be used")
    print("-" * 70)
    resized_array = np.array(resized)
    print(f"✓ Converted to array: {resized_array.shape}")
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS - torchvision resize works with imageio-loaded images")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_imageio_resize()
    sys.exit(0 if success else 1)
