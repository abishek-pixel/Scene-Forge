#!/usr/bin/env python3
"""
Diagnostic script for segmentation model loading issues.
Run this to diagnose and verify the fix.

Usage:
  python diagnose_segmentation.py
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_internet_connectivity():
    """Check if HuggingFace is reachable."""
    logger.info("=" * 60)
    logger.info("1. CHECKING INTERNET CONNECTIVITY")
    logger.info("=" * 60)
    
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        logger.info("✓ HuggingFace.co is reachable")
        return True
    except Exception as e:
        logger.error(f"✗ Cannot reach HuggingFace.co: {e}")
        logger.error("  → Check internet connection or firewall settings")
        return False


def check_transformers_library():
    """Check if transformers library is installed."""
    logger.info("\n" + "=" * 60)
    logger.info("2. CHECKING TRANSFORMERS LIBRARY")
    logger.info("=" * 60)
    
    try:
        import transformers
        logger.info(f"✓ transformers library installed: v{transformers.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ transformers not installed: {e}")
        logger.error("  → Run: pip install transformers")
        return False


def check_torch_installation():
    """Check if PyTorch is installed."""
    logger.info("\n" + "=" * 60)
    logger.info("3. CHECKING PYTORCH")
    logger.info("=" * 60)
    
    try:
        import torch
        logger.info(f"✓ PyTorch installed: v{torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"  GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("  Running on CPU (no GPU detected)")
        return True
    except ImportError as e:
        logger.error(f"✗ PyTorch not installed: {e}")
        logger.error("  → Run: pip install torch")
        return False


def check_huggingface_cache():
    """Check HuggingFace cache status."""
    logger.info("\n" + "=" * 60)
    logger.info("4. CHECKING HUGGINGFACE CACHE")
    logger.info("=" * 60)
    
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    
    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        logger.info(f"✓ Cache directory exists: {cache_dir}")
        logger.info(f"  Cache size: {cache_size / (1024**2):.1f} MB")
        return True
    else:
        logger.info(f"ℹ Cache directory doesn't exist yet: {cache_dir}")
        logger.info("  (Will be created on first model download)")
        return True


def test_model_loading():
    """Test loading segmentation models."""
    logger.info("\n" + "=" * 60)
    logger.info("5. TESTING MODEL LOADING")
    logger.info("=" * 60)
    
    try:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        
        model_candidates = [
            "nvidia/segformer-b0-imagenet",
            "facebook/mask2former-swin-tiny-ade20k",
            "openmmlab/upernet-convnext-tiny",
            "nvidia/segformer-b3-cityscapes",
        ]
        
        for i, model_name in enumerate(model_candidates, 1):
            try:
                logger.info(f"\nAttempt {i}: Loading {model_name}...")
                
                processor = AutoImageProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                logger.info(f"✓ SUCCESS: {model_name}")
                logger.info(f"  Processor: {type(processor).__name__}")
                logger.info(f"  Model: {type(model).__name__}")
                return True
                
            except Exception as e:
                logger.warning(f"✗ Failed: {type(e).__name__}: {str(e)[:80]}")
                continue
        
        logger.error("✗ All model candidates failed to load!")
        return False
        
    except Exception as e:
        logger.error(f"✗ Could not test models: {e}")
        return False


def test_segmentation_inference():
    """Test actual segmentation on sample image."""
    logger.info("\n" + "=" * 60)
    logger.info("6. TESTING SEGMENTATION INFERENCE")
    logger.info("=" * 60)
    
    try:
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        from PIL import Image
        
        logger.info("Loading model...")
        try:
            processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-imagenet")
            model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-imagenet")
        except:
            logger.warning("Could not load primary model, skipping inference test")
            return False
        
        logger.info("Creating test image...")
        # Create a simple test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        logger.info("Running segmentation...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
        
        logger.info("✓ Segmentation inference successful!")
        logger.info(f"  Output shape: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Segmentation inference failed: {e}")
        return False


def print_summary(results):
    """Print diagnostic summary."""
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    checks = [
        ("Internet Connectivity", results.get('internet', False)),
        ("Transformers Library", results.get('transformers', False)),
        ("PyTorch Installation", results.get('torch', False)),
        ("HuggingFace Cache", results.get('cache', False)),
        ("Model Loading", results.get('models', False)),
        ("Segmentation Inference", results.get('inference', False)),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
    
    logger.info(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n✓ All systems operational! Segmentation should work.")
    elif passed >= 4:
        logger.info("\n⚠ Most systems OK. Segmentation may work with fallback.")
    else:
        logger.error("\n✗ Critical issues detected. Segmentation may not work.")
        logger.error("\nRecommended actions:")
        if not results.get('internet'):
            logger.error("  1. Check internet connection to HuggingFace")
        if not results.get('transformers'):
            logger.error("  2. Install transformers: pip install transformers")
        if not results.get('torch'):
            logger.error("  3. Install PyTorch: pip install torch")
        if not results.get('cache'):
            logger.error("  4. Clear cache: rm -rf ~/.cache/huggingface/")


def main():
    """Run all diagnostics."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║" + "SEGMENTATION MODEL DIAGNOSTIC".center(58) + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # Run all checks
    results['internet'] = check_internet_connectivity()
    results['transformers'] = check_transformers_library()
    results['torch'] = check_torch_installation()
    results['cache'] = check_huggingface_cache()
    results['models'] = test_model_loading()
    results['inference'] = test_segmentation_inference()
    
    # Print summary
    print_summary(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("1. Review the checks above")
    logger.info("2. Fix any issues (see recommendations)")
    logger.info("3. Restart backend: python main.py")
    logger.info("4. Test with a video upload")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
