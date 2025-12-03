
import sys
import os
sys.path.insert(0, os.getcwd())
print("Starting import...")
import sam3
print("Imported sam3")
from sam3.model_builder import build_sam3_image_model
print("Imported build_sam3_image_model")
