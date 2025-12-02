# Product Requirements Document (PRD)
## AI-Powered Smart Campus Surveillance System

---

## 📋 Document Information

**Version:** 1.0  
**Last Updated:** November 2025  
**Project Status:** Development Phase  
**Target Platform:** Local Desktop/Server Application  
**Technology Stack:** Python, YOLOv8, OpenCV, Flask, SQLite

---

## 🎯 Executive Summary

### Vision Statement
Create an intelligent, AI-powered surveillance system that transforms traditional passive CCTV monitoring into a proactive, automated security solution for educational campuses. The system will detect anomalies in real-time, generate instant alerts, and provide actionable insights to security personnel without requiring continuous human monitoring.

### Problem Statement
Current campus surveillance systems are:
- **Passive & Reactive** - Only record, don't analyze
- **Labor Intensive** - Require constant human monitoring
- **Prone to Human Error** - Fatigue leads to missed incidents
- **Delayed Response** - Manual intervention causes critical time loss
- **Disconnected** - No integration between security components

### Solution Overview
An AI-powered surveillance system that:
- Automatically detects anomalies (unauthorized entry, fights, fire, etc.)
- Generates real-time alerts with priority classification
- Runs locally (no cloud dependency, cost-effective)
- Provides a unified dashboard for monitoring and analytics
- Works with existing camera infrastructure

---

## 🎯 Project Objectives

### Primary Goals
1. **Real-time Anomaly Detection** - Detect 8+ types of security incidents automatically
2. **Instant Alert System** - Notify security within 3 seconds of detection
3. **Local Deployment** - Run entirely on campus infrastructure
4. **High Accuracy** - Achieve >85% detection accuracy with <15% false positives
5. **User-Friendly Interface** - Enable non-technical security staff to operate system

### Success Metrics
| Metric | Target | Priority |
|--------|--------|----------|
| Detection Accuracy | >85% | Critical |
| Alert Response Time | <3 seconds | Critical |
| False Positive Rate | <15% | High |
| System Uptime | >95% | High |
| User Satisfaction | >80% | Medium |
| Frame Processing Rate | >15 FPS | Medium |

---

## 👥 Target Users

### Primary Users
1. **Security Personnel** (Guards, Officers)
   - Monitor dashboard
   - Respond to alerts
   - Review incident recordings

2. **Security Supervisors** (Head of Security)
   - View analytics and reports
   - Configure system settings
   - Manage user access

3. **Campus Administration** (Deans, Admin Staff)
   - Review security reports
   - Make policy decisions
   - Budget allocation

### User Personas

**Persona 1: Campus Security Guard**
- Name: Officer Rajesh
- Age: 35
- Tech Proficiency: Basic
- Needs: Simple interface, clear alerts, quick incident response
- Pain Points: Too many camera feeds, alert fatigue, missed incidents

**Persona 2: Security Supervisor**
- Name: Mr. Sharma
- Age: 45
- Tech Proficiency: Intermediate
- Needs: Analytics, reporting, system configuration
- Pain Points: Lack of data, manual report generation, no trends analysis

---

## 🔍 Functional Requirements

### Core Features (MVP - Must Have)

#### 1. Video Input & Processing
- **FR-1.1** System shall accept input from:
  - Webcam (USB camera)
  - IP cameras (RTSP stream)
  - Video files (MP4, AVI for testing)
- **FR-1.2** Process video at minimum 15 FPS
- **FR-1.3** Support multiple camera feeds (1-10 cameras)
- **FR-1.4** Handle camera connection failures gracefully

#### 2. AI Detection Capabilities
- **FR-2.1** Detect persons with >85% accuracy
- **FR-2.2** Identify the following anomalies:
  - Unauthorized entry in restricted zones
  - Unattended objects (bags, packages)
  - Crowd gathering (>5 people)
  - Fire/smoke detection
  - Fighting/violence
  - Smoking behavior
  - Person falling
  - Vehicle in restricted area

#### 3. Zone Management
- **FR-3.1** Allow defining virtual restricted zones
- **FR-3.2** Enable/disable zones dynamically
- **FR-3.3** Support multiple zones per camera
- **FR-3.4** Visual zone editing via mouse clicks

#### 4. Alert System
- **FR-4.1** Generate alerts with priority levels:
  - CRITICAL (Fire, weapon, fall)
  - HIGH (Fight, unauthorized entry)
  - MEDIUM (Smoking, unattended object)
  - LOW (Loitering)
- **FR-4.2** Alert delivery methods:
  - Dashboard notification
  - Console output
  - Audio alert (local)
  - Email (optional)
  - SMS (optional)
- **FR-4.3** Alert cooldown to prevent spam (10-30 seconds)
- **FR-4.4** Save snapshot image with each alert

#### 5. Data Storage
- **FR-5.1** Store alert records in SQLite database
- **FR-5.2** Save incident snapshots locally
- **FR-5.3** Store detection metadata (timestamp, confidence, location)
- **FR-5.4** Implement data retention policy (30-90 days)

#### 6. Web Dashboard
- **FR-6.1** Live video feed display
- **FR-6.2** Real-time alert notifications
- **FR-6.3** Alert history view
- **FR-6.4** Basic statistics (daily incident count)
- **FR-6.5** System status indicators

### Advanced Features (Future Enhancements)

#### Phase 2 Features
- Face recognition for authorized personnel
- License plate recognition
- Object tracking across cameras
- Heatmap generation
- Advanced analytics dashboard
- Mobile app integration
- Multi-user access with roles
- Cloud backup option

---

## 🛠️ Technical Requirements

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User Interface Layer                │
│         (Flask Web Dashboard + Live Display)         │
├─────────────────────────────────────────────────────┤
│                 Application Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │   Detector   │  │ Alert Manager│  │  Database  ││
│  │   Engine     │  │              │  │  Manager   ││
│  └──────────────┘  └──────────────┘  └────────────┘│
├─────────────────────────────────────────────────────┤
│                   AI/ML Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐│
│  │  YOLO    │ │ MediaPipe│ │ Color/Pattern        ││
│  │  Models  │ │  Pose    │ │ Analysis             ││
│  └──────────┘ └──────────┘ └──────────────────────┘│
├─────────────────────────────────────────────────────┤
│                  Data Layer                          │
│  ┌──────────────┐  ┌──────────────┐                │
│  │   SQLite DB  │  │  File Storage│                │
│  │  (Metadata)  │  │  (Images/Video)│              │
│  └──────────────┘  └──────────────┘                │
├─────────────────────────────────────────────────────┤
│                Hardware Interface                    │
│         (Camera Drivers, Video Capture)              │
└─────────────────────────────────────────────────────┘
```

### Technology Stack

#### Core Technologies
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Programming Language | Python | 3.8+ | Core development |
| AI Framework | Ultralytics YOLO | v8 | Object detection |
| Computer Vision | OpenCV | 4.8+ | Video processing |
| Pose Estimation | MediaPipe | 0.10+ | Body pose analysis |
| Web Framework | Flask | 2.3+ | Dashboard backend |
| Database | SQLite | 3.40+ | Data storage |
| Frontend | HTML/CSS/JS | - | Dashboard UI |

#### Python Dependencies
```
ultralytics>=8.0.0
opencv-python>=4.8.0
mediapipe>=0.10.0
flask>=2.3.0
numpy>=1.24.0
pillow>=10.0.0
```

### Hardware Requirements

#### Minimum Specifications
- **CPU:** Intel i5 / AMD Ryzen 5 (4 cores)
- **RAM:** 8 GB
- **Storage:** 100 GB free space
- **GPU:** Optional (runs on CPU)
- **Camera:** Any USB webcam or IP camera with RTSP

#### Recommended Specifications
- **CPU:** Intel i7 / AMD Ryzen 7 (8 cores)
- **RAM:** 16 GB
- **Storage:** 500 GB SSD
- **GPU:** NVIDIA GTX 1660 or better (CUDA enabled)
- **Camera:** 1080p IP cameras with night vision

#### Network Requirements
- **Bandwidth:** 10 Mbps per camera
- **Local Network:** Gigabit Ethernet recommended
- **Internet:** Not required (fully local system)

### Performance Requirements

- **PR-1** Process minimum 15 frames per second
- **PR-2** Detect anomalies within 500ms of occurrence
- **PR-3** Deliver alerts within 3 seconds
- **PR-4** Support up to 10 concurrent camera streams
- **PR-5** System startup time < 30 seconds
- **PR-6** Memory usage < 4GB for 4 camera streams

---

## 🎨 User Interface Requirements

### Dashboard Layout

```
┌────────────────────────────────────────────────────────┐
│  🎥 Campus Surveillance Dashboard           🔴 LIVE   │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│
│  │              │  │              │  │              ││
│  │   Camera 1   │  │   Camera 2   │  │   Camera 3   ││
│  │              │  │              │  │              ││
│  └──────────────┘  └──────────────┘  └──────────────┘│
│                                                         │
├────────────────────────────────────────────────────────┤
│  🚨 ACTIVE ALERTS                                      │
│  ┌──────────────────────────────────────────────────┐ │
│  │ 🔥 FIRE DETECTED - Camera 2 - 10:23 AM   [VIEW]  │ │
│  │ ⚠️  FIGHT SUSPECTED - Camera 1 - 10:20 AM [VIEW]  │ │
│  └──────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────┤
│  📊 STATISTICS                                         │
│  Alerts Today: 12  |  People Count: 45  |  Status: ✅ │
└────────────────────────────────────────────────────────┘
```

### UI Components

#### 1. Live Video Display
- Grid layout (1x1, 2x2, 3x3, 4x4)
- Click to expand camera view
- Overlay detection boxes
- Show zone boundaries
- FPS counter

#### 2. Alert Panel
- List of active alerts
- Color-coded by priority
- Timestamp and camera ID
- Quick action buttons (View, Dismiss, Acknowledge)
- Alert sound toggle

#### 3. Controls
- Start/Stop detection
- Add/Edit zones
- Configure alert thresholds
- View logs
- Export reports

#### 4. Statistics
- Real-time metrics
- Daily/weekly trends
- Detection accuracy
- System health

---

## 📊 Data Requirements

### Database Schema

#### Table: events
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    camera_id INTEGER,
    event_type TEXT NOT NULL,
    priority TEXT,
    confidence REAL,
    location TEXT,
    description TEXT,
    image_path TEXT,
    acknowledged BOOLEAN DEFAULT 0,
    acknowledged_by TEXT,
    acknowledged_at TEXT
);
```

#### Table: cameras
```sql
CREATE TABLE cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    location TEXT,
    status TEXT DEFAULT 'active',
    created_at TEXT
);
```

#### Table: zones
```sql
CREATE TABLE zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER,
    name TEXT,
    coordinates TEXT,
    zone_type TEXT,
    enabled BOOLEAN DEFAULT 1,
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);
```

### File Storage Structure
```
campus_surveillance/
├── data/
│   ├── database/
│   │   └── surveillance.db
│   ├── events/
│   │   ├── 2024-11-07/
│   │   │   ├── fire_10-23-15.jpg
│   │   │   └── fight_10-20-30.jpg
│   │   └── 2024-11-08/
│   ├── videos/ (optional recordings)
│   └── logs/
│       └── system.log
```

---

## 🔒 Security & Privacy Requirements

### Data Security
- **SR-1** Encrypt sensitive data at rest
- **SR-2** Secure database access with authentication
- **SR-3** Implement user roles (Admin, Operator, Viewer)
- **SR-4** Log all system access and changes
- **SR-5** Auto-delete old footage per retention policy

### Privacy Compliance
- **PR-1** No facial recognition (unless legally permitted)
- **PR-2** Blur faces in saved images (optional feature)
- **PR-3** Restrict access to authorized personnel only
- **PR-4** Comply with campus privacy policies
- **PR-5** Provide data access logs for auditing

### Network Security
- **NS-1** Run on isolated campus network
- **NS-2** No external internet access required
- **NS-3** Encrypted communication between components
- **NS-4** Regular security updates

---

## 🧪 Testing Requirements

### Test Categories

#### 1. Unit Testing
- Individual detection algorithms
- Database operations
- Alert generation
- Configuration management

#### 2. Integration Testing
- Camera input → Detection → Alert flow
- Dashboard → Backend API
- Database → File storage

#### 3. Performance Testing
- Frame processing speed
- Multi-camera load testing
- Memory usage monitoring
- Alert latency measurement

#### 4. Accuracy Testing
- Detection precision/recall
- False positive rate
- False negative rate
- Edge case handling

### Test Scenarios

| Test ID | Scenario | Expected Result |
|---------|----------|-----------------|
| T-001 | Person enters restricted zone | Alert generated within 3s |
| T-002 | Fire detected in frame | CRITICAL alert + snapshot saved |
| T-003 | Multiple people fighting | HIGH alert with bounding boxes |
| T-004 | Unattended bag stationary >30s | MEDIUM alert triggered |
| T-005 | Camera disconnected | System logs error, continues with other cameras |
| T-006 | 10 cameras active simultaneously | All process at >10 FPS |
| T-007 | Same alert within cooldown period | No duplicate alert |
| T-008 | Database full | Implements retention policy |

---

## 📅 Development Timeline

### Phase 1: Foundation (Weeks 1-4)
- **Week 1:** Project setup, environment configuration
- **Week 2:** Basic video capture and display
- **Week 3:** YOLO integration and object detection
- **Week 4:** Database setup and basic UI

### Phase 2: Core Detection (Weeks 5-8)
- **Week 5:** Implement person detection and zone management
- **Week 6:** Fire and smoke detection
- **Week 7:** Fight and pose-based detection
- **Week 8:** Alert system integration

### Phase 3: Dashboard & Integration (Weeks 9-12)
- **Week 9:** Web dashboard development
- **Week 10:** Multi-camera support
- **Week 11:** Analytics and reporting
- **Week 12:** Testing and bug fixes

### Phase 4: Deployment & Training (Weeks 13-16)
- **Week 13:** Documentation and user manual
- **Week 14:** Pilot deployment
- **Week 15:** User training
- **Week 16:** Full rollout and monitoring

---

## 📝 Acceptance Criteria

### MVP Release Criteria
✅ Detects at least 5 anomaly types (person, fire, fight, unattended object, crowd)  
✅ Processes single camera feed at >15 FPS  
✅ Generates alerts within 3 seconds  
✅ False positive rate <20%  
✅ Dashboard displays live feed and alerts  
✅ Saves incident snapshots and metadata  
✅ Runs stably for 24 hours without crash  
✅ User can define and edit restricted zones  
✅ Complete user documentation provided  

### Production Release Criteria
✅ All MVP criteria met  
✅ Supports 4+ cameras simultaneously  
✅ False positive rate <15%  
✅ Detection accuracy >85%  
✅ Web dashboard fully functional  
✅ Email/SMS alerts working (if enabled)  
✅ User training completed  
✅ 1 week of stable pilot testing  

---

## 🚀 Deployment Plan

### Pre-Deployment Checklist
- [ ] Hardware installed and tested
- [ ] Software dependencies installed
- [ ] Models downloaded and loaded
- [ ] Database initialized
- [ ] Configuration file created
- [ ] Backup system in place
- [ ] User accounts created
- [ ] Training materials prepared

### Deployment Steps
1. Install system on target hardware
2. Configure camera connections
3. Define restricted zones
4. Set alert thresholds
5. Test end-to-end workflow
6. Train security personnel
7. Run pilot for 1 week
8. Gather feedback and adjust
9. Full production rollout
10. Ongoing monitoring and support

### Rollback Plan
- Maintain backup of previous system
- Document rollback procedures
- Keep traditional CCTV active during transition
- Quick disable switch for AI system

---

## 📞 Support & Maintenance

### Support Channels
- **Technical Issues:** Email support or helpdesk
- **User Training:** In-person sessions + video tutorials
- **Documentation:** User manual + FAQ
- **Updates:** Monthly security patches

### Maintenance Schedule
- **Daily:** System health check
- **Weekly:** Database backup, log review
- **Monthly:** Performance optimization, model updates
- **Quarterly:** Security audit, user feedback review

---

## 📚 Documentation Requirements

### Required Documents
1. **User Manual** - For security personnel
2. **Admin Guide** - For system administrators
3. **Technical Documentation** - Architecture and API docs
4. **Installation Guide** - Setup instructions
5. **Troubleshooting Guide** - Common issues and solutions
6. **Training Materials** - Slides and videos

---

## 🎓 Training Requirements

### Training Modules
1. **Basic Operation** (2 hours)
   - Dashboard navigation
   - Reading alerts
   - Basic troubleshooting

2. **Advanced Features** (2 hours)
   - Zone configuration
   - Report generation
   - System settings

3. **Emergency Response** (1 hour)
   - Critical alert protocols
   - Communication procedures
   - Escalation paths

---

## ⚠️ Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| False positives overload | High | Medium | Implement alert cooldown, tune thresholds |
| Hardware failure | High | Low | Redundant systems, backup cameras |
| Poor lighting affects detection | Medium | High | Use cameras with night vision, train on varied lighting |
| Network connectivity issues | Medium | Medium | Local processing, buffer alerts |
| User resistance to new system | Medium | Medium | Comprehensive training, easy UI |
| Privacy concerns | High | Low | No facial recognition, clear policies |

---

## 📄 Appendices

### Appendix A: Glossary
- **Anomaly:** Unusual event requiring attention
- **RTSP:** Real-Time Streaming Protocol
- **FPS:** Frames Per Second
- **YOLO:** You Only Look Once (detection algorithm)
- **Cooldown:** Period to prevent duplicate alerts

### Appendix B: References
- YOLOv8 Documentation: https://docs.ultralytics.com/
- OpenCV Documentation: https://docs.opencv.org/
- MediaPipe: https://developers.google.com/mediapipe

### Appendix C: Change Log
- v1.0 (Nov 2025): Initial PRD

---

**Document Approval:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Owner | | | |
| Technical Lead | | | |
| Security Lead | | | |

---

*End of Document*
