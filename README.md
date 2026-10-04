# 🎯 YOLOv8 Real-Time AI Object Detection & Tracking

A real-time computer vision dashboard for **Object Detection and Multi-Object Tracking** built for **CodeAlpha AI Internship (Task 3)** using **YOLOv8**, **OpenCV**, and **Streamlit**.

## 🚀 Features
- 🎥 **Real-time Live Webcam Stream**: Perform object detection and tracking directly through webcam feed.
- 🧠 **Multi-Model Selector**: Switch dynamically between `YOLOv8n` (Fast) and `YOLOv8s` (Accurate) models.
- 🎛️ **Custom Confidence Control**: Interactive sidebar slider to fine-tune detection sensitivity.
- 🎨 **Modern Dark UI**: SaaS-style dark glassmorphism web interface built with custom CSS in Streamlit.
- 🆔 **Object Tracking**: Assigns unique Tracking IDs (using ByteTrack/BoT-SORT) to track individual objects across frames.

## 🛠️ Tech Stack
- **Python** 🐍
- **Streamlit** 🎈 (Web Interface)
- **Ultralytics YOLOv8** 🧠 (Detection & Tracking)
- **OpenCV** 📹 (Computer Vision Frame Processing)

## 💻 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt