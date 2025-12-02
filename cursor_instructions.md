# 🚀 Cursor Instructions - AI Campus Surveillance

## Quick Setup (5 minutes)

```bash
# 1. Create project
mkdir campus_surveillance && cd campus_surveillance

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install ultralytics opencv-python mediapipe flask numpy

# 4. Create structure
mkdir -p models data/{database,events} src templates static
```

---

## Project Structure

```
campus_surveillance/
├── src/
│   ├── detector.py          # Main detection engine
│   ├── database.py          # SQLite operations
│   └── alert_manager.py     # Alert system
├── models/                   # AI models (auto-download)
├── data/                     # Database & captured events
├── templates/                # HTML files
├── main.py                   # Run this
├── config.py                 # Settings
└── requirements.txt
```

---

## Step 1: Create config.py

```python
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Camera
CAMERA_SOURCE = 0  # 0=webcam, or RTSP URL, or video file path

# Detection
CONFIDENCE_THRESHOLD = 0.5
CROWD_THRESHOLD = 5
ALERT_COOLDOWN = 10

# Create directories
for d in [MODELS_DIR, DATA_DIR, f"{DATA_DIR}/database", f"{DATA_DIR}/events"]:
    os.makedirs(d, exist_ok=True)
```

---

## Step 2: Create src/detector.py

**Prompt for Cursor AI:**
```
Create a Python class called CampusSurveillance that:
1. Uses YOLOv8 for object detection
2. Detects people, fire (color-based), fights (pose-based), smoking
3. Manages restricted zones
4. Generates alerts with priority levels
5. Saves incident snapshots
```

Or use the code from the earlier artifact (SimpleSurveillance class).

---

## Step 3: Create src/database.py

```python
import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_database()
    
    def init_database(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                priority TEXT,
                confidence REAL,
                description TEXT,
                image_path TEXT
            )
        ''')
        self.conn.commit()
    
    def save_event(self, alert):
        self.conn.execute('''
            INSERT INTO events VALUES (NULL, ?, ?, ?, ?, ?, ?)
        ''', (alert['timestamp'], alert['type'], alert['priority'],
              alert['confidence'], alert['description'], alert['image_path']))
        self.conn.commit()
    
    def get_recent_events(self, limit=50):
        return self.conn.execute(
            'SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', 
            (limit,)
        ).fetchall()
```

---

## Step 4: Create main.py

```python
from src.detector import CampusSurveillance

if __name__ == "__main__":
    print("Starting Campus Surveillance System...")
    system = CampusSurveillance()
    system.run()
```

---

## Step 5: Run It

```bash
python main.py
```

**Expected output:**
- Opens camera feed window
- Detects people with bounding boxes
- Prints alerts to console
- Saves snapshots to `data/events/`

---

## Optional: Add Web Dashboard

**Create dashboard.py:**

```python
from flask import Flask, render_template, Response, jsonify
from src.detector import CampusSurveillance
from src.database import DatabaseManager
import cv2

app = Flask(__name__)
surveillance = CampusSurveillance()
db = DatabaseManager('data/database/surveillance.db')

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/alerts')
def get_alerts():
    events = db.get_recent_events(20)
    return jsonify({'alerts': events})

def generate_frames():
    while True:
        frame = surveillance.get_current_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

**Run:** `python dashboard.py` → Open `http://localhost:5000`

---

## Download Optional Models

### Fire Detection
```bash
# Visit: https://universe.roboflow.com/fire-detection-ysq6h/fire-detection-vvysp
# Download as YOLOv8 → Rename to fire_yolov8n.pt → Place in models/
```

### Smoking Detection
```bash
# Visit: https://universe.roboflow.com/detectingsmokingobject/smoking-gmpbb
# Download → Rename to smoking_yolov8n.pt → Place in models/
```

**Note:** System works without these (uses fallback methods)

---

## Using Cursor AI

### Quick Commands
- `Ctrl+K` - Quick edit with AI
- `Ctrl+L` - Chat with AI
- `Ctrl+I` - Composer (multi-file edits)

### Good Prompts
```
"Add error handling to database operations"
"Optimize frame processing for better FPS"
"Create unit tests for detector.py"
"Add logging to all functions"
"Fix this camera connection error"
```

---

## Common Issues

**Camera won't open:**
```python
CAMERA_SOURCE = 1  # Try different index (0, 1, 2...)
```

**Slow performance:**
```python
CONFIDENCE_THRESHOLD = 0.6  # Higher = faster
# Or resize frames: cv2.resize(frame, (320, 240))
```

**Too many alerts:**
```python
ALERT_COOLDOWN = 30  # Increase cooldown
CONFIDENCE_THRESHOLD = 0.7  # Higher confidence
```

---

## Testing

```bash
# Run system
python main.py

# With dashboard
python dashboard.py

# Check installation
python -c "import cv2, ultralytics, mediapipe; print('OK')"
```

---

## What Gets Detected

| Detection | Method | Model Needed? |
|-----------|--------|---------------|
| People | YOLO | ❌ Auto-downloads |
| Fire | Color analysis | ⚪ Optional |
| Fight | Pose + proximity | ❌ Built-in |
| Smoking | Hand-to-mouth | ⚪ Optional |
| Fall | Pose analysis | ❌ Built-in |
| Crowd | Count people | ❌ Built-in |

---

## Quick Reference

```bash
# Start
python main.py

# Stop
Ctrl+C

# View database
sqlite3 data/database/surveillance.db

# Check logs
ls -lh data/events/

# Backup
cp -r data backups/data_$(date +%Y%m%d)
```

---

## Development Phases

1. **Phase 1:** Basic person detection (Day 1)
2. **Phase 2:** Add anomaly detection (Day 2-3)
3. **Phase 3:** Alert system (Day 4)
4. **Phase 4:** Database logging (Day 5)
5. **Phase 5:** Web dashboard (Day 6-7)

---

## That's It! 🎉

Start with `python main.py` and build from there.

Use Cursor AI (`Ctrl+L`) to ask questions as you code.

For full details, see PRD document.