# 🚀 Quick Reference Guide
## Campus Surveillance System - Command Cheatsheet

---

## ⚡ Installation (One-Time Setup)

```bash
# 1. Create project
mkdir campus_surveillance && cd campus_surveillance

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install everything
pip install ultralytics opencv-python mediapipe flask numpy pillow

# 4. Create folders
mkdir -p models data/{database,events,logs} static templates src tests

# Done! ✅
```

---

## 🎬 Running the System

```bash
# Basic detection (console only)
python main.py

# With web dashboard
python dashboard.py
# Then open: http://localhost:5000

# Background mode (Linux/Mac)
nohup python dashboard.py &

# Stop background process
pkill -f dashboard.py
```

---

## 📥 Model Downloads

### Required (Auto-downloads)
```bash
# yolov8n.pt - Downloads automatically on first run
# Just run the code!
```

### Optional Models

**Fire Detection:**
```bash
# Method 1: Roboflow
Visit: https://universe.roboflow.com/fire-detection-ysq6h/fire-detection-vvysp
Download as YOLOv8 → Rename to: fire_yolov8n.pt → Place in models/

# Method 2: Direct download
wget https://github.com/spacewalk01/yolov8-fire-detection/releases/download/v1.0/fire_yolov8n.pt
mv fire_yolov8n.pt models/
```

**Smoking Detection:**
```bash
Visit: https://universe.roboflow.com/detectingsmokingobject/smoking-gmpbb
Download → Rename to: smoking_yolov8n.pt → Place in models/
```

**Weapon Detection:**
```bash
Visit: https://universe.roboflow.com/weapon-vzapd/weapon-zw8kv
Download → Rename to: weapon_yolov8n.pt → Place in models/
```

---

## 🔧 Configuration

### Edit `config.py`

```python
# Camera source
CAMERA_SOURCE = 0  # Webcam
CAMERA_SOURCE = "rtsp://192.168.1.100:554/stream"  # IP camera
CAMERA_SOURCE = "videos/test.mp4"  # Video file

# Detection sensitivity
CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections (0.3-0.7)
CROWD_THRESHOLD = 5  # Number of people = crowd

# Alert frequency
ALERT_COOLDOWN = 10  # Seconds between same alerts

# Enable/Disable features
ENABLE_FIRE_DETECTION = True
ENABLE_FIGHT_DETECTION = True
ENABLE_SMOKING_DETECTION = False  # Set True when model available
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_detector.py

# Run with coverage
python -m pytest --cov=src tests/

# Quick verification
python verify_installation.py
```

---

## 🐛 Debugging

### Check what's running
```bash
# See Python processes
ps aux | grep python

# Check port usage
netstat -an | grep 5000  # Linux/Mac
netstat -an | findstr 5000  # Windows
```

### View logs
```bash
# Real-time log viewing
tail -f data/logs/system.log

# Last 50 lines
tail -n 50 data/logs/system.log

# Search logs for errors
grep ERROR data/logs/system.log
```

### Common fixes
```bash
# Camera not working? Try different index
CAMERA_SOURCE = 1  # or 2, 3...

# Port already in use?
DASHBOARD_PORT = 5001  # Change port in config.py

# Slow performance?
# Use smaller model or reduce resolution in config.py
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
```

---

## 📊 Database Operations

```bash
# View database
sqlite3 data/database/surveillance.db

# Common queries (in SQLite shell):
SELECT COUNT(*) FROM events;
SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;
SELECT event_type, COUNT(*) FROM events GROUP BY event_type;

# Exit SQLite
.quit
```

### Python database access
```python
from src.database import DatabaseManager

db = DatabaseManager('data/database/surveillance.db')
recent = db.get_recent_events(20)
print(f"Found {len(recent)} recent events")
```

---

## 🎯 Cursor IDE Commands

### AI Assistant
```
Ctrl+K (Cmd+K) - Quick edit with AI
Ctrl+L (Cmd+L) - Chat with AI
Ctrl+I (Cmd+I) - Open Composer
```

### Useful Prompts
```
"Fix this error"
"Explain this code"
"Add error handling"
"Optimize for performance"
"Write tests for this function"
"Add logging statements"
```

---

## 📁 Project Structure Reference

```
campus_surveillance/
├── models/              # AI model files (.pt)
├── data/
│   ├── database/       # SQLite database
│   ├── events/         # Saved incident images
│   └── logs/           # System logs
├── src/                # Source code
│   ├── detector.py     # Main detection
│   ├── alert_manager.py
│   ├── database.py
│   └── model_manager.py
├── static/             # Web assets (CSS/JS)
├── templates/          # HTML files
├── tests/              # Unit tests
├── main.py             # Entry point
├── dashboard.py        # Web server
├── config.py           # Configuration
└── requirements.txt    # Dependencies
```

---

## 🔍 Detection Types & Fallbacks

| Detection | Primary Method | Fallback | Model Needed? |
|-----------|---------------|----------|---------------|
| **People** | YOLOv8 | None | ❌ Auto-downloads |
| **Fire** | Fire YOLO | Color analysis | ⚪ Optional |
| **Smoke** | Fire YOLO | None | ⚪ Optional |
| **Fight** | Pose + Proximity | None | ❌ Built-in |
| **Smoking** | Smoking YOLO | Hand-to-mouth pose | ⚪ Optional |
| **Fall** | Pose analysis | Bounding box ratio | ❌ Built-in |
| **Crowd** | Person count | None | ❌ Built-in |
| **Weapons** | Weapon YOLO | None | ⚪ Optional |

**Legend:**
- ❌ No download needed
- ⚪ Optional download for better accuracy

---

## 🚨 Alert Priority Levels

```
🔴 CRITICAL → Fire, Weapon, Fall, Intrusion
🟡 HIGH     → Fight, Unauthorized entry
🟠 MEDIUM   → Smoking, Unattended object
⚪ LOW      → Loitering, Parking violation
```

### Default cooldown periods:
- CRITICAL: 5 seconds
- HIGH: 10 seconds  
- MEDIUM: 20 seconds
- LOW: 30 seconds

---

## 📈 Performance Tuning

### If FPS is low (<10):

```python
# Option 1: Reduce resolution
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Option 2: Skip frames
if frame_count % 3 == 0:  # Process every 3rd frame
    results = model(frame)

# Option 3: Use nano model
model = YOLO('yolov8n.pt')  # Fastest

# Option 4: Reduce confidence threshold
CONFIDENCE_THRESHOLD = 0.6  # Higher = faster
```

### If too many false alerts:

```python
# Increase confidence
CONFIDENCE_THRESHOLD = 0.7

# Increase cooldown
ALERT_COOLDOWN = 30

# Adjust thresholds
CROWD_THRESHOLD = 10  # More people needed
FIRE_COLOR_THRESHOLD = 0.10  # More fire color needed
```

---

## 🛠️ Maintenance Tasks

### Daily
```bash
# Check system is running
ps aux | grep dashboard.py

# Quick log check
tail data/logs/system.log
```

### Weekly
```bash
# Backup database
cp data/database/surveillance.db backups/surveillance_$(date +%Y%m%d).db

# Check disk space
df -h

# Review error logs
grep ERROR data/logs/system.log | tail -20
```

### Monthly
```bash
# Clean old events (>90 days)
python -c "from src.database import DatabaseManager; db = DatabaseManager('data/database/surveillance.db'); db.cleanup_old_events(90)"

# Clean old images
find data/events -type f -mtime +90 -delete

# Update dependencies
pip install --upgrade ultralytics opencv-python
```

---

## 🆘 Troubleshooting Quick Fixes

### Problem: "Cannot open camera"
```python
# Try different camera indices
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
```

### Problem: "Model not found"
```bash
# Force re-download
rm models/yolov8n.pt
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Problem: "Port already in use"
```bash
# Find what's using port 5000
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Kill the process or change port in config.py
```

### Problem: "Out of memory"
```python
# In config.py:
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
# Use yolov8n.pt (smallest model)
```

### Problem: "Too slow"
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA or accept CPU performance
```

---

## 📞 Support Resources

### Documentation
- **This Project:** README.md, CURSOR_INSTRUCTIONS.md
- **YOLOv8:** https://docs.ultralytics.com/
- **OpenCV:** https://docs.opencv.org/
- **Flask:** https://flask.palletsprojects.com/

### Community
- YOLOv8 GitHub: https://github.com/ultralytics/ultralytics
- OpenCV Forum: https://forum.opencv.org/
- Stack Overflow: Tag with `yolo`, `opencv-python`

---

## ✅ Pre-Deployment Checklist

```
[ ] Python 3.8+ installed
[ ] All dependencies installed (pip list)
[ ] Camera accessible (test with main.py)
[ ] Models downloaded (check models/ folder)
[ ] Database initialized (check data/database/)
[ ] Config.py customized for your setup
[ ] Restricted zones defined
[ ] Alert thresholds tested
[ ] Dashboard accessible (http://localhost:5000)
[ ] Runs for 1+ hour without crash
[ ] Documentation reviewed
[ ] Team trained on usage
```

---

## 🎓 Quick Start Tutorial

### 5-Minute Demo

```bash
# 1. Setup (2 min)
git clone your-repo OR create folder
cd campus_surveillance
python -m venv venv && source venv/bin/activate
pip install ultralytics opencv-python mediapipe

# 2. Create main.py (1 min)
# Copy the SimpleSurveillance code from artifacts

# 3. Run (1 min)
python main.py

# 4. Test (1 min)
# Move in front of camera
# Wait for person detection
# Check console for alerts
```

### 30-Minute Full Setup

Follow CURSOR_INSTRUCTIONS.md Phase 1-3

---

## 🎯 Key Files to Edit

### For basic customization:
1. `config.py` - All settings
2. `main.py` - Entry point
3. `src/detector.py` - Detection logic

### For dashboard:
1. `dashboard.py` - Backend API
2. `templates/dashboard.html` - Frontend UI
3. `static/css/style.css` - Styling

### For alerts:
1. `src/alert_manager.py` - Alert logic
2. `config.py` - Alert thresholds

---

## 💾 Backup Commands

```bash
# Backup everything
tar -czf backup_$(date +%Y%m%d).tar.gz \
    data/database/ \
    data/events/ \
    models/ \
    config.py

# Restore backup
tar -xzf backup_20241107.tar.gz

# Backup database only
cp data/database/surveillance.db backups/
```

---

## 🔑 Important Variables

```python
# In config.py

CONFIDENCE_THRESHOLD = 0.5   # 0.3-0.7 typical
CROWD_THRESHOLD = 5          # People count
ALERT_COOLDOWN = 10          # Seconds
FIRE_COLOR_THRESHOLD = 0.05  # Percentage
DB_RETENTION_DAYS = 90       # Data lifecycle
FPS_TARGET = 15              # Performance goal
```

---

## 📱 Quick Commands Summary

```bash
# Start
python main.py                    # Console mode
python dashboard.py               # Web mode

# Stop
Ctrl+C                           # Graceful stop
pkill -f python                  # Force stop

# Test
python -m pytest tests/          # Run tests
python verify_installation.py    # Check setup

# Monitor
tail -f data/logs/system.log    # Watch logs
htop                             # System resources

# Maintain
sqlite3 data/database/surveillance.db  # Query DB
du -sh data/events/              # Check disk usage
```

---

**Keep this file handy for quick reference! 📌**

*Last updated: November 2025*
