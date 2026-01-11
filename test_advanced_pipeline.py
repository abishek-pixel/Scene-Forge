#!/usr/bin/env python3
"""
Test advanced 3D reconstruction pipeline locally
Useful for debugging before deployment
"""

import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "SceneForge_Backend"))

from app.core.services.processing_service import ProcessingService


async def create_test_image(output_path: str):
    """Create a simple test image"""
    img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    # Add a gradient for more realistic depth
    for i in range(512):
        for j in range(512):
            brightness = int(255 * (1 - (i + j) / 1024))
            img_array[i, j] = [brightness, brightness // 2, brightness // 3]
    
    img = Image.fromarray(img_array)
    img.save(output_path)
    logger.info(f"✓ Test image created: {output_path}")
    return output_path


async def update_callback(job_id: str, progress: int, message: str):
    """Progress callback"""
    if progress >= 0:
        print(f"[{progress:3d}%] {message}")
    else:
        print(f"[ERROR] {message}")


async def test_advanced_pipeline():
    """Test the advanced 3D reconstruction pipeline"""
    
    logger.info("=" * 60)
    logger.info("Advanced 3D Reconstruction Pipeline Test")
    logger.info("=" * 60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_img = tmpdir / "test_image.png"
        output_dir = tmpdir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Create test image
        await create_test_image(str(input_img))
        
        # Initialize service
        logger.info("\n1. Initializing ProcessingService...")
        service = ProcessingService()
        logger.info(f"   Advanced reconstruction available: {service.advanced_3d is not None}")
        
        # Process
        logger.info("\n2. Starting advanced 3D reconstruction...")
        print()
        
        try:
            result = await service.process_scene(
                input_path=str(input_img),
                output_path=str(output_dir),
                prompt="Test 3D reconstruction",
                job_id="test-001",
                update_callback=update_callback
            )
            
            # Print results
            print()
            logger.info("\n3. Results:")
            logger.info(f"   Status: {result['status']}")
            logger.info(f"   Output: {result.get('output_path', 'N/A')}")
            
            metadata = result.get('metadata', {})
            logger.info(f"   Pipeline: {metadata.get('pipeline', 'Unknown')}")
            logger.info(f"   Accuracy: {metadata.get('expected_accuracy', 'Unknown')}")
            
            if 'mesh_stats' in metadata:
                stats = metadata['mesh_stats']
                logger.info(f"   Mesh vertices: {stats.get('vertices', 'N/A')}")
                logger.info(f"   Mesh faces: {stats.get('faces', 'N/A')}")
                logger.info(f"   File size: {stats.get('file_size', 'N/A')} bytes")
            
            # Verify output file
            output_file = result.get('output_path')
            if output_file and Path(output_file).exists():
                file_size = Path(output_file).stat().st_size
                logger.info(f"\n4. File Verification: ✓ SUCCESS")
                logger.info(f"   File: {output_file}")
                logger.info(f"   Size: {file_size} bytes ({file_size/1024:.1f} KB)")
                
                if file_size == 0:
                    logger.error("   ⚠ WARNING: File is empty (0 bytes)!")
                    return False
                else:
                    logger.info("   ✓ File is valid (non-empty)")
                    return True
            else:
                logger.error(f"\n4. File Verification: ✗ FAILED")
                logger.error(f"   Output file not found: {output_file}")
                return False
            
        except Exception as e:
            logger.error(f"\n3. Processing FAILED: {e}", exc_info=True)
            return False


async def main():
    """Main test runner"""
    success = await test_advanced_pipeline()
    
    print("\n" + "=" * 60)
    if success:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("Advanced pipeline is working correctly!")
    else:
        logger.error("✗ TESTS FAILED")
        logger.error("See errors above for details.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
