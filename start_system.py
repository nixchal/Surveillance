#!/usr/bin/env python3
"""
Easy start script for the Campus Surveillance System.
Handles dependency checks, model downloads, and system startup.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

# Define paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SAM3_DIR = MODELS_DIR / "sam3"
SAM3_CHECKPOINT = SAM3_DIR / "sam3.pt"
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"

def check_python_version():
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)

def check_dependencies():
    print("Checking dependencies...")
    try:
        import ultralytics
        import cv2
        import sam3
        print("✅ Dependencies look good.")
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e.name}")
        print("Please run: pip install -r requirements.txt")
        # Optional: ask to install
        response = input("Do you want to install requirements now? (y/n): ").strip().lower()
        if response == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            print("Proceeding, but errors may occur.")

def check_models():
    print("Checking models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SAM3_DIR.mkdir(parents=True, exist_ok=True)

    # Check SAM 3.0
    if not SAM3_CHECKPOINT.exists():
        print(f"⚠️  SAM 3.0 model not found at {SAM3_CHECKPOINT}")
        print("You can download it using the helper tool.")
        response = input("Run download tool now? (y/n): ").strip().lower()
        if response == 'y':
            subprocess.call([sys.executable, "tools/download_sam3.py"])
        else:
            print("Skipping SAM 3.0 download. System may fail if SAM is enabled.")
    else:
        print("✅ SAM 3.0 model found.")

    # Check YOLO (Ultralytics usually handles this, but good to check)
    # We don't strictly need to download it here as YOLO auto-downloads, 
    # but we can ensure the dir exists.

def run_system(video_path=None):
    print("\n🚀 Starting Campus Surveillance System...")
    
    env = os.environ.copy()
    if video_path:
        print(f"🎥 Using video source: {video_path}")
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        env["CAMERA_SOURCE"] = str(video_path)
    else:
        print("📷 Using default camera source (Webcam)")

    try:
        subprocess.run([sys.executable, "main.py"], env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ System crashed with exit code {e.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Start the Campus Surveillance System")
    parser.add_argument("--video", type=str, help="Path to a video file to use as input")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and model checks")
    
    args = parser.parse_args()

    check_python_version()
    
    if not args.skip_checks:
        check_dependencies()
        check_models()

    run_system(args.video)

if __name__ == "__main__":
    main()
