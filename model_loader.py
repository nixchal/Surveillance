"""
Model Loader and Manager
Handles all downloaded pre-trained models

Project Structure:
campus_surveillance/
├── models/
│   ├── yolov8n.pt              (auto-downloaded)
│   ├── yolov8n-seg.pt          (auto-downloaded when segmentation enabled)
│   ├── fire_yolov8n.pt         (download from links above)
│   ├── smoking_yolov8n.pt      (download from links above)
│   ├── weapon_yolov8n.pt       (download from links above)
│   └── violence_model.pt       (optional)
"""

import os
from ultralytics import YOLO

class ModelManager:
    """Manages all detection models"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available models"""
        print("Loading detection models...\n")
        
        # 1. Base object detection (auto-downloads if not exists)
        self.models['objects'] = self._load_model('yolov8n.pt', 'Object Detection')
        
        # 2. Segmentation enhanced model (optional)
        self.models['objects_seg'] = self._load_model('yolov8n-seg.pt', 'Object Segmentation', optional=True)

        # 3. Fire detection (optional)
        self.models['fire'] = self._load_model('fire_yolov8n.pt', 'Fire Detection', optional=True)
        
        # 4. Smoking detection (optional)
        self.models['smoking'] = self._load_model('smoking_yolov8n.pt', 'Smoking Detection', optional=True)
        
        # 5. Weapon detection (optional)
        self.models['weapon'] = self._load_model('weapon_yolov8n.pt', 'Weapon Detection', optional=True)
        
        # 6. Violence detection (optional)
        self.models['violence'] = self._load_model('violence_model.pt', 'Violence Detection', optional=True)
        
        print("\n" + "="*60)
        print(f"✅ Loaded {len([m for m in self.models.values() if m])} models")
        print("="*60 + "\n")
    
    def _load_model(self, filename, model_name, optional=False):
        """Load individual model"""
        model_path = os.path.join(self.models_dir, filename)
        
        # If file doesn't exist
        if not os.path.exists(model_path):
            if optional:
                print(f"⚠️  {model_name}: Not found (optional)")
                print(f"   Download from instructions and place in: {model_path}")
                print(f"   System will use fallback detection method.\n")
                return None
            else:
                # For yolov8n.pt, ultralytics will auto-download
                if filename == 'yolov8n.pt':
                    print(f"📥 {model_name}: Downloading...")
                    model = YOLO(filename)
                    print(f"✅ {model_name}: Ready\n")
                    return model
                else:
                    print(f"❌ {model_name}: REQUIRED but not found!")
                    print(f"   Please place model at: {model_path}\n")
                    return None
        
        # Load existing model
        try:
            model = YOLO(model_path)
            print(f"✅ {model_name}: Loaded from {filename}")
            return model
        except Exception as e:
            print(f"❌ {model_name}: Failed to load - {e}\n")
            return None
    
    def get_model(self, model_type):
        """Get specific model"""
        return self.models.get(model_type)
    
    def is_available(self, model_type):
        """Check if model is loaded"""
        return self.models.get(model_type) is not None
    
    def get_available_models(self):
        """List all available models"""
        return [name for name, model in self.models.items() if model is not None]


# Download helper functions
class ModelDownloader:
    """Helper to download models programmatically"""
    
    @staticmethod
    def download_from_roboflow(api_key, workspace, project, version, output_path):
        """
        Download model from Roboflow
        
        Example:
        download_from_roboflow(
            api_key="your_api_key",
            workspace="fire-detection",
            project="fire-and-smoke",
            version=1,
            output_path="models/fire_yolov8n.pt"
        )
        """
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project)
            dataset = project.version(version).download("yolov8")
            
            print(f"✅ Downloaded to: {dataset.location}")
            print(f"   Weights file: {dataset.location}/weights/best.pt")
            print(f"   Copy to: {output_path}")
            
        except ImportError:
            print("Install roboflow: pip install roboflow")
        except Exception as e:
            print(f"Download failed: {e}")
    
    @staticmethod
    def download_from_url(url, output_path):
        """Download model from direct URL"""
        import urllib.request
        
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"✅ Downloaded to: {output_path}")
        except Exception as e:
            print(f"Download failed: {e}")


# Setup instructions generator
def generate_setup_instructions():
    """Generate setup instructions for downloaded models"""
    
    instructions = """
╔══════════════════════════════════════════════════════════════════╗
║          MODEL DOWNLOAD & SETUP INSTRUCTIONS                      ║
╚══════════════════════════════════════════════════════════════════╝

STEP 1: Create Project Structure
---------------------------------
campus_surveillance/
├── models/                  ← Create this folder
│   ├── yolov8n.pt          ← Auto-downloads
│   ├── fire_yolov8n.pt     ← Download manually
│   ├── smoking_yolov8n.pt  ← Download manually
│   └── weapon_yolov8n.pt   ← Download manually


STEP 2: Download Base Model (Automatic)
----------------------------------------
✅ yolov8n.pt will download automatically on first run
   No action needed!

Optional: ✅ yolov8n-seg.pt downloads automatically when segmentation is enabled.
           Place a copy in models/ if you want to ship it with the project.


STEP 3: Download Fire Detection Model (Optional but Recommended)
-----------------------------------------------------------------

Method A: Roboflow (Easiest)
1. Go to: https://universe.roboflow.com/
2. Search: "fire and smoke detection"
3. Choose: "Fire and Smoke Detection by My Workspace"
4. Click: "Download Dataset"
5. Select: "YOLOv8" format
6. Sign up (free) and download
7. Extract and find: weights/best.pt
8. Rename to: fire_yolov8n.pt
9. Move to: campus_surveillance/models/

Method B: Direct GitHub
1. Download from: https://github.com/spacewalk01/yolov8-fire-detection
2. Get: fire_yolov8n.pt
3. Place in: campus_surveillance/models/


STEP 4: Download Smoking Detection Model (Optional)
----------------------------------------------------
1. Visit: https://universe.roboflow.com/detectingsmokingobject/smoking-gmpbb
2. Download as YOLOv8 format
3. Rename weights/best.pt to: smoking_yolov8n.pt
4. Place in: campus_surveillance/models/

Alternative: Use pose-based detection (no model needed)
   ✅ Already included in code!


STEP 5: Download Weapon Detection Model (Optional)
---------------------------------------------------
1. Visit: https://universe.roboflow.com/weapon-vzapd/weapon-zw8kv
2. Download as YOLOv8
3. Rename to: weapon_yolov8n.pt
4. Place in: campus_surveillance/models/


STEP 6: Verify Installation
----------------------------
Run this Python script:

```python
from model_loader import ModelManager

manager = ModelManager()
print("Available models:", manager.get_available_models())
```

Expected output:
✅ Object Detection: Ready
✅ Fire Detection: Loaded from fire_yolov8n.pt
⚠️  Smoking Detection: Not found (optional)
⚠️  Weapon Detection: Not found (optional)


QUICK START (Minimum Setup)
----------------------------
To run the system with just basic detection:

1. Install dependencies:
   pip install ultralytics opencv-python mediapipe

2. Run system (yolov8n.pt auto-downloads):
   python main_surveillance.py

3. Optional models will be skipped, system uses fallback methods


FALLBACK DETECTION METHODS (When models not available)
-------------------------------------------------------
✅ Fire Detection → Color-based analysis (built-in)
✅ Smoking → Pose analysis (hand-to-mouth detection)
✅ Fight → Pose + proximity analysis
✅ Weapons → Not detected (model required)


TROUBLESHOOTING
---------------
Issue: "Model not found"
Fix: Check file is named exactly as shown and in models/ folder

Issue: "CUDA out of memory"
Fix: Use smaller model (yolov8n.pt instead of yolov8l.pt)

Issue: "Slow detection"
Fix: Reduce frame processing rate or use GPU


RECOMMENDED SETUP FOR BEST RESULTS
-----------------------------------
Priority 1 (Essential):
✅ yolov8n.pt (auto-downloads)

Priority 2 (Important):
✅ fire_yolov8n.pt (fire safety)

Priority 3 (Nice to have):
⚪ smoking_yolov8n.pt
⚪ weapon_yolov8n.pt

All optional models can be added later without code changes!
"""
    
    return instructions


# Save instructions to file
def save_instructions():
    instructions = generate_setup_instructions()
    
    with open('MODEL_SETUP_INSTRUCTIONS.txt', 'w') as f:
        f.write(instructions)
    
    print("✅ Instructions saved to: MODEL_SETUP_INSTRUCTIONS.txt")
    print(instructions)


if __name__ == "__main__":
    # Test model manager
    print("Testing Model Manager...\n")
    
    manager = ModelManager()
    
    print("\nAvailable Models:")
    for model_name in manager.get_available_models():
        print(f"  ✅ {model_name}")
    
    print("\n" + "="*60)
    print("Run save_instructions() to generate setup guide")
    print("="*60)
    
    # Uncomment to save instructions
    # save_instructions()
