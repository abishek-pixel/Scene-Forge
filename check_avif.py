#!/usr/bin/env python3

# Check AVIF support
try:
    from PIL import features
    has_avif = features.check('avif')
    print(f"PIL AVIF support: {has_avif}")
except:
    print("PIL AVIF support: False")

# Try to import pillow_avif
try:
    import pillow_avif
    print("pillow_avif: installed")
except ImportError:
    print("pillow_avif: NOT installed")

# Try alternate: use opencv
try:
    import cv2
    print(f"OpenCV: installed (version {cv2.__version__})")
except:
    print("OpenCV: NOT installed")
