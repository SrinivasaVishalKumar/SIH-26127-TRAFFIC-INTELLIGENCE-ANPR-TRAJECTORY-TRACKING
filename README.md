# SIH-26127: Traffic Intelligence — ANPR & Vehicle Trajectory Tracking

Real-time Automatic Number Plate Recognition (ANPR), vehicle trajectory tracking, and traffic intelligence system developed for Smart India Hackathon (SIH).

---

## 📌 Project Overview

This project provides an end-to-end intelligent traffic monitoring and vehicle tracking solution:
- **Vehicle & License Plate Detection**: YOLOv8 model optimized and accelerated with Intel OpenVINO for real-time edge inference.
- **Number Plate Recognition (OCR)**: Enhanced EasyOCR pipeline tailored for Indian license plates (Standard and BH series) with regex validation and sanitization.
- **Batch Export & Cloud Sync**: Automated batch aggregation and transmission to remote cloud backend endpoints.
- **Interactive Analytics Dashboard**: Streamlit-based web dashboard integrating Folium maps for spatial-temporal vehicle trajectory tracking and live log feeds.

---

## 📂 Repository Structure

`	ext
SIH-26127-Traffic-Intelligence/
├── ai_engine/
│   ├── license_plate_model_openvino_model/  # OpenVINO IR model for license plate detection
│   ├── yolov8n_openvino_model/              # OpenVINO IR model for YOLOv8
│   ├── tracker.py                           # Core real-time video stream & tracking engine
│   ├── ocr_utils.py                         # OCR extraction, preprocessing & plate parsing
│   ├── api_sender.py                        # Cloud sync service sending logs to backend
│   └── __init__.py
├── backend/                                 # Backend service endpoints and API logic
├── data/                                    # Local sample videos and test feeds
├── database/                                # Vehicle logs, CSV records, and counts
│   ├── log.json
│   ├── plate_log.csv
│   └── traffic_counts.json
├── frontend/
│   └── dashboard.py                         # Streamlit & Folium vehicle tracking UI
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
`

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed
- Virtual environment (recommended)

### 2. Installation

Clone the repository and install the dependencies:
`ash
git clone https://github.com/SrinivasaVishalKumar/SIH-26127-TRAFFIC-INTELLIGENCE-ANPR-TRAJECTORY-TRACKING.git
cd SIH-26127-TRAFFIC-INTELLIGENCE-ANPR-TRAJECTORY-TRACKING

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
`

### 3. Running the AI Engine & Tracker
To start the live detection and ANPR tracking:
`ash
cd ai_engine
python tracker.py
`

### 4. Running the Dashboard
Launch the interactive Streamlit dashboard:
`ash
cd frontend
streamlit run dashboard.py
`

### 5. Running Cloud Log Dispatcher (Optional)
To send batch detection logs to the remote cloud server:
`ash
cd ai_engine
python api_sender.py
`

---

## 🛠️ Tech Stack
- **Deep Learning / Vision**: Ultralytics YOLOv8, OpenVINO, OpenCV
- **OCR Engine**: EasyOCR, Regex (Indian Standard & BH Series Plate Validation)
- **Frontend & Mapping**: Streamlit, Folium, Streamlit-Folium
- **Data & APIs**: Pandas, Requests, JSON
