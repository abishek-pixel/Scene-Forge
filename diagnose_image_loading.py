#!/usr/bin/env python3
"""
Diagnostic script for image loading issues.
Tests PIL image opening with various paths and formats.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_loading():
    """Test image loading with absolute and relative paths."""
    
    print("\n" + "="*70)
    print("IMAGE LOADING DIAGNOSTIC")
    print("="*70 + "\n")
    
    # Get backend path
    backend_path = Path(__file__).parent / "SceneForge_Backend"
    
    # Test paths
    test_cases = [
        ("uploads/20251210_114510/sofa.jpg", "Relative path (current working dir)"),
        (os.path.join(os.getcwd(), "uploads/20251210_114510/sofa.jpg"), "Absolute from CWD"),
        (str(backend_path / "uploads/20251210_114510/sofa.jpg"), "Absolute from backend"),
    ]
    
    for test_path, description in test_cases:
        print(f"Test: {description}")
        print(f"Path: {test_path}")
        
        # Check if file exists
        if os.path.exists(test_path):
            print(f"✓ File exists")
            
            # Try to open with PIL
            try:
                img = Image.open(test_path)
                print(f"✓ PIL opened successfully")
                print(f"  Mode: {img.mode}, Size: {img.size}")
                img.close()
            except Exception as e:
                print(f"✗ PIL error: {str(e)}")
        else:
            abs_path = os.path.abspath(test_path)
            print(f"✗ File not found")
            print(f"  Absolute: {abs_path}")
            print(f"  Exists there: {os.path.exists(abs_path)}")
        
        print()
    
    # Show current working directory
    print("\nSystem Info:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Backend path: {backend_path}")
    
    # List uploads directory if it exists
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        print(f"\nUploads directory exists")
        recent_dirs = sorted(uploads_dir.iterdir())[-3:] if uploads_dir.iterdir() else []
        for d in recent_dirs:
            files = list(d.glob("*"))
            print(f"  {d.name}/: {len(files)} file(s)")
            for f in files[:3]:
                print(f"    - {f.name}")
    else:
        print(f"Uploads directory not found at {uploads_dir.absolute()}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("""
1. Always convert paths to absolute:
   if not os.path.isabs(path):
       path = os.path.abspath(path)

2. Verify file exists before opening:
   if not os.path.exists(path):
       raise Exception(f"File not found: {path}")

3. Catch PIL errors specifically:
   try:
       img = Image.open(path)
   except Exception as e:
       raise Exception(f"Cannot open {path}: {str(e)}")

4. Run from correct working directory:
   cd SceneForge_Backend
   python -m uvicorn app.main:app --reload
""")

if __name__ == "__main__":
    test_image_loading()
