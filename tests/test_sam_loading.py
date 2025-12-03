import sys
import os
import unittest
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sam_segmenter import SamPersonSegmenter
import config

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestSamLoading(unittest.TestCase):
    def test_sam_loading(self):
        print("Testing SAM3 loading...")
        try:
            segmenter = SamPersonSegmenter()
            print("Successfully initialized SamPersonSegmenter")
            self.assertIsNotNone(segmenter.model)
            self.assertIsNotNone(segmenter.processor)
        except Exception as e:
            self.fail(f"Failed to initialize SamPersonSegmenter: {e}")

if __name__ == "__main__":
    unittest.main()
