
import logging
import sys
import os
import time
import threading

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

import config
from src.detector import CampusSurveillance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_startup():
    logger.info("Starting verification...")
    
    # Initialize CampusSurveillance
    try:
        surveillance = CampusSurveillance(show_windows=False)
        logger.info("CampusSurveillance initialized successfully.")
        
        if config.SAM_ENABLED:
            if surveillance.sam_segmenter is not None:
                logger.info("SUCCESS: SAM3 person segmenter is initialized and active.")
            else:
                logger.error("FAILURE: SAM3 person segmenter is None despite being enabled.")
        else:
            logger.info("SAM is disabled in config.")
            
    except Exception as e:
        logger.error(f"CampusSurveillance initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_startup()
