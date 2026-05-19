# Sponsor Logo Detection

Bachelor thesis project at OsloMet – Oslo Metropolitan University, Spring 2026.  
Developed in collaboration with Forzasys AS.

## About

Sponsors invest significant resources in sports broadcasting, yet measuring the actual visibility of their logos during matches and interviews remains a manual and time-consuming process. This project addresses that challenge by developing an automated sponsor logo detection system for Allsvenskan football pre-game interview footage.

The system extracts frames from live video streams, uses a trained YOLOv8 object detection model to identify sponsor logos, and presents the results through a web interface showing each sponsor's exposure time and detection frequency.

## Pipeline
HLS/M3U8 Video Stream → FFmpeg Frame Extraction → Label Studio Annotation → YOLO Training → Streamlit Web Interface

## Installation

```bash
pip install ultralytics streamlit opencv-python pandas
```

## Usage

**Train the model:**
```bash
python train.py
```

**Run the application:**
```bash
python -m streamlit run app.py
```

## Streamlit

After running, choose between uploading a video or pasting URL
You can choose threshold with the slider
When the model is finshed running, you will get a table with metrics and data

 
DATA3900 · OsloMet · Spring 2026