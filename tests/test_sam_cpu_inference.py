
print("DEBUG: Importing sys...")
import sys
print("DEBUG: Importing os...")
import os
print("DEBUG: Importing torch...")
import torch
print("DEBUG: Importing numpy...")
import numpy as np
print("DEBUG: Importing logging...")
import logging

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

print("DEBUG: Importing SamPersonSegmenter...")
from src.sam_segmenter import SamPersonSegmenter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cpu_inference():
    logger.info("Starting SAM 3.0 CPU inference test...")
    
    # Force CPU
    if torch.cuda.is_available():
        logger.info("CUDA is available, but forcing CPU for this test.")
    else:
        logger.info("CUDA is not available, running on CPU as expected.")
        
    try:
        # Initialize Segmenter (this should use the logic in src/sam_segmenter.py which we modified to check device)
        # We might need to mock torch.cuda.is_available() if we want to strictly test CPU path on a GPU machine,
        # but for now let's assume the user's machine has no CUDA or we can rely on the internal logic.
        # Actually, let's explicitly patch torch.cuda.is_available to return False to simulate the user's environment exactly.
        
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        logger.info("Mocked torch.cuda.is_available() to return False.")
        
        segmenter = SamPersonSegmenter()
        logger.info("Model initialized successfully on CPU.")
        
        # Create a dummy image (H, W, 3)
        dummy_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Create a dummy bounding box [x1, y1, x2, y2]
        dummy_bboxes = [[100, 100, 200, 200]]
        
        logger.info("Running inference on dummy image...")
        masks = segmenter.segment_persons(dummy_image, dummy_bboxes)
        
        logger.info(f"Inference successful. Generated {len(masks)} masks.")
        
        # Restore
        torch.cuda.is_available = original_is_available
        
    except Exception as e:
        logger.error(f"Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_cpu_inference()
