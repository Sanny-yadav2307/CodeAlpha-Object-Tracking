import cv2
import streamlit as st
from ultralytics import YOLO

# Page setup ⚙️
st.set_page_config(
    page_title="YOLOv8 AI Object Tracker",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for Full Dark Theme & High Contrast Text 🎨
st.markdown("""
    <style>
    /* Top Header Bar Transparent/Dark */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }
    
    /* App Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* All Labels & Text High Visibility */
    label, .stMarkdown, p, span {
        color: #F0F2F6 !important;
        font-weight: 500;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1E1E2F 0%, #2A2A40 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3A3A55;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    .header-title {
        color: #00E676 !important;
        font-size: 26px;
        font-weight: 700;
    }
    
    .header-subtitle {
        color: #B0B3C6 !important;
        font-size: 14px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #262935;
    }
    </style>
""", unsafe_allow_html=True)

# Custom Banner 🏷️
st.markdown("""
    <div class="header-card">
        <div class="header-title">🎯 YOLOv8 Real-Time AI Object Tracker</div>
        <div class="header-subtitle">CodeAlpha AI Internship — Task 3 | High Performance Computer Vision Dashboard</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls 🎛️
st.sidebar.title("🎛️ Control Panel")

model_type = st.sidebar.selectbox(
    "Choose YOLO Model 🧠",
    ["yolov8n.pt (Fast)", "yolov8s.pt (Accurate)"],
    index=0
)
model_name = "yolov8n.pt" if "yolov8n" in model_type else "yolov8s.pt"

conf_threshold = st.sidebar.slider("Confidence Threshold 📊", 0.1, 1.0, 0.35, 0.05)

run_tracker = st.sidebar.checkbox("▶️ Start Live Camera Feed", value=False)

# Safe Model Loading Function 🧠
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"❌ Model load nahi ho pa raha: {e}. Internet connection check karein.")
        return None

model = load_model(model_name)

# Video Display Frame 🖥️️
frame_placeholder = st.empty()

if run_tracker and model is not None:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Camera access nahi ho pa raha hai.")
    
    while cap.isOpened() and run_tracker:
        success, frame = cap.read()
        if not success:
            st.warning("⚠️ Frame capture karne me dikkat aa rahi hai.")
            break

        # YOLO Tracking 🎯
        results = model.track(frame, conf=conf_threshold, persist=True)

        # Draw Annotations 🎨
        annotated_frame = results[0].plot()

        # BGR to RGB Conversion
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display Video Feed
        frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
else:
    if model is not None:
        st.info("👈 Live Tracking start karne ke liye Sidebar me **'▶️ Start Live Camera Feed'** ko check karein.")