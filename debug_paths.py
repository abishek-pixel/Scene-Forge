#!/usr/bin/env python3
"""Debug AVIF path issues"""

import os
import cv2
from pathlib import Path

test_paths = [
    r'SceneForge_Backend\uploads\20251210_114510\sofa.jpg',  # Backslash
    'SceneForge_Backend/uploads/20251210_114510/sofa.jpg',   # Forward slash
    os.path.join('SceneForge_Backend', 'uploads', '20251210_114510', 'sofa.jpg'),  # os.path.join
    str(Path('SceneForge_Backend/uploads/20251210_114510/sofa.jpg')),  # pathlib
]

for path in test_paths:
    exists = os.path.exists(path)
    print(f"Path: {path}")
    print(f"  Exists: {exists}")
    if exists:
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"  OpenCV: Returned None")
            else:
                print(f"  OpenCV: SUCCESS - shape {img.shape}")
        except Exception as e:
            print(f"  OpenCV: Error - {str(e)}")
    print()
