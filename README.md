# AI-Powered Smart Campus Surveillance System

A comprehensive, real-time surveillance solution designed for smart campuses. This system leverages state-of-the-art computer vision models, including **YOLOv8** for object detection and **Meta's SAM 3.0 (Segment Anything Model)** for precise person segmentation, to detect anomalies and enhance security.

## 🚀 Features

*   **Advanced Anomaly Detection**:
    *   **Fire & Smoke Detection**: Real-time visual analysis to identify fire hazards.
    *   **Fight/Violence Detection**: Motion and pose-based analysis to detect aggressive behavior.
    *   **Weapon Detection**: Identifies potential threats (knives, guns) using object detection.
    *   **Crowd Analysis**: Monitors density and detects overcrowding.
    *   **Smoking Detection**: Identifies smoking behavior in restricted areas.
*   **Precision Segmentation with SAM 3.0**:
    *   Integrated **SAM 3.0** for high-fidelity, pixel-perfect segmentation of detected persons.
    *   Robust **CPU/GPU compatibility**: Optimized to run on CUDA if available, with automatic CPU fallback for SAM 3.0 components.
*   **Interactive Web Dashboard**:
    *   **Live Video Feed**: Low-latency streaming of processed footage.
    *   **Real-time Alerts**: Instant visual notifications for detected anomalies.
    *   **Premium UI**: Modern, dark-mode interface with glassmorphism effects.
*   **Modular Architecture**:
    *   Extensible detector framework (`src/detectors/`) allowing easy addition of new detection logic.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Computer Vision**:
    *   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Object Detection)
    *   [Meta SAM 3.0](https://github.com/facebookresearch/segment-anything) (Segmentation)
    *   OpenCV (Image Processing)
    *   MediaPipe (Pose Estimation)
*   **Web Framework**: Flask (Backend), HTML5/CSS3/JS (Frontend)
*   **Database**: SQLite (Alert logging)

## 📦 Installation

### Prerequisites
*   Python 3.8 or higher
*   (Optional) NVIDIA GPU with CUDA toolkit installed for accelerated inference.

### Quick Start

The easiest way to get started is using the helper script:

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the system (uses webcam by default)
python start_system.py
```

### 🎥 Testing with Video

To test the system with a video file (e.g., for fight detection):

```bash
python start_system.py --video fight.mp4
```

### Manual Installation

If you prefer to set up manually:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Models:**
    *   **SAM 3.0:** Run `python tools/download_sam3.py` or download `sam3.pt` manually to `models/sam3/`.
    *   **YOLO:** Automatically downloaded on first run.

3.  **Run Application:**
    ```bash
    python main.py
    ```

## 🚀 Usage

1.  **Start the System:**
    ```bash
    python start_system.py
    ```
    *   The system will initialize the camera feed, load YOLO and SAM models, and start the web server.

2.  **Access the Dashboard:**
    Open your web browser and navigate to:
    ```
    http://localhost:5000
    ```

## ⚙️ Configuration

The system is highly configurable via `config.py`. Key settings include:

*   **Feature Flags**: Toggle specific detectors (e.g., `SAM_ENABLED`, `ENABLE_FIRE_DETECTION`).
*   **Thresholds**: Adjust confidence levels for detections (`CONFIDENCE_THRESHOLD`, `FIGHT_CONFIDENCE_THRESHOLD`).
*   **Camera Source**: Set `CAMERA_SOURCE` to a webcam ID (e.g., `0`) or a video file path.
*   **Alerts**: Configure email notifications and alert priorities.

## 📂 Project Structure

```
Surveillance/
├── main.py                 # Entry point
├── config.py               # Global configuration
├── requirements.txt        # Python dependencies
├── src/
│   ├── detector.py         # Main orchestration logic
│   ├── sam_segmenter.py    # SAM 3.0 integration wrapper
│   ├── video_stream.py     # Video capture handling
│   └── detectors/          # Modular detector implementations
│       ├── fire_detector.py
│       └── fight_detector.py
├── sam3/                   # Local SAM 3.0 package source
├── templates/              # HTML templates for dashboard
├── static/                 # CSS/JS assets
└── models/                 # Directory for model weights
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
