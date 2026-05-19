import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import pandas as pd

# Load the trained YOLO model, cached to avoid reloading on every rerun
@st.cache_resource
def load_model():
    return YOLO('weights/best.pt')

model = load_model()

# Page title and description
st.title("Sponsor Logo Detection")
st.write("Detect and measure sponsor logo exposure in football interview footage")

# Let user choose between uploading a file or pasting a URL
input_method = st.radio("Choose input method:", ["Upload Video", "Paste Video URL"])

video_source = None

if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(uploaded_file.read())
            video_source = f.name
        st.video(uploaded_file)

elif input_method == "Paste Video URL":
    # Supports HLS/M3U8 streams from Forzasys
    url = st.text_input("Paste video URL (supports HLS/M3U8):")
    if url:
        video_source = url
        st.info(f"URL loaded: {url}")

# Confidence threshold - detections below this are marked as unknown
confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.5)

if video_source and st.button("Run Detection"):
    
    logo_total_detections = {}  # Total bounding box detections per logo
    logo_unique_frames = {}     # Frames where logo appears at least once
    unknown_frames = 0          # Frames with low confidence detections
    total_frames = 0
    fps = 25  # Default FPS

    status = st.empty()

    with st.spinner("Detecting logos..."):
        try:
            # Get FPS from video to convert frames to seconds
            cap = cv2.VideoCapture(video_source)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                cap.release()

            # Run YOLO inference frame by frame
            results = model.predict(
                source=video_source,
                stream=True,       # Memory efficient streaming
                conf=confidence,
                verbose=False
            )

            for i, result in enumerate(results):
                total_frames += 1
                logos_this_frame = set()  # Unique logos in this frame

                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if conf >= confidence:
                        logo_name = model.names[cls]
                        
                        # Count every detection including duplicates
                        if logo_name not in logo_total_detections:
                            logo_total_detections[logo_name] = 0
                        logo_total_detections[logo_name] += 1

                        logos_this_frame.add(logo_name)
                    else:
                        unknown_frames += 1

                # Count unique frames per logo
                for logo_name in logos_this_frame:
                    if logo_name not in logo_unique_frames:
                        logo_unique_frames[logo_name] = 0
                    logo_unique_frames[logo_name] += 1

                if i % 10 == 0:
                    status.text(f"Processing frame {i}...")

        except Exception as e:
            st.error(f"Error: {e}")

    st.success("Detection complete!")
    st.subheader("Results")

    # Build results table sorted by exposure time
    results_data = []
    for logo in sorted(logo_unique_frames.keys(),
                      key=lambda x: logo_unique_frames[x],
                      reverse=True):

        unique_seconds = round(logo_unique_frames[logo] / fps, 2)
        total_detections = logo_total_detections.get(logo, 0)

        results_data.append({
            "Logo": logo,
            "Unique Frames": logo_unique_frames[logo],
            "Exposure Time (seconds)": unique_seconds,
            "Total Detections": total_detections,
        })

    # Add unknown detections at the bottom
    if unknown_frames > 0:
        results_data.append({
            "Logo": "Unknown / Out of bounds",
            "Unique Frames": unknown_frames,
            "Exposure Time (seconds)": round(unknown_frames / fps, 2),
            "Total Detections": unknown_frames,
        })

    if results_data:
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Frames", total_frames)
        col2.metric("Logos Detected", len(logo_unique_frames))
        col3.metric("Total Duration", f"{round(total_frames / fps, 1)}s")
    else:
        st.warning("No logos detected. Try lowering the confidence threshold.")