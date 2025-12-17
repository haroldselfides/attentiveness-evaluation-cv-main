import streamlit as st
import cv2
import tempfile
import numpy as np
import time
import os
import pandas as pd
import altair as alt

from gaze import analyze_gaze
from concentration import get_engagement_prob
from fusion import AttentionFusion, summarize_session

# ---------------------------
# CONFIGURATION & STATE
# ---------------------------
st.set_page_config(page_title="Student Engagement Analysis", layout="wide")

if "logs" not in st.session_state:
    st.session_state.logs = []
if "behavior_history" not in st.session_state:
    st.session_state.behavior_history = []
if "fusion" not in st.session_state:
    st.session_state.fusion = AttentionFusion()

# Colors (BGR)
COLOR_FOCUSED = (0, 255, 0)      # Green
COLOR_CONFUSED = (0, 165, 255)   # Orange
COLOR_FRUSTRATED = (0, 0, 255)   # Red
COLOR_BORED = (255, 255, 0)      # Cyan
COLOR_DROWSY = (128, 0, 128)     # Purple
COLOR_AWAY = (0, 0, 0)           # Black

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def determine_state_and_color(gaze, prob):
    """
    Determines the label and color based on AI Score and Gaze.
    Priority: Drowsy (Low Score) -> Looking Away (Gaze) -> Engaged States
    """
    # 1. Check for Drowsy/Sleeping FIRST (Low score = eyes closed/head drop)
    if prob < 0.40:
        return "Not Engaged: Drowsy", COLOR_DROWSY

    # 2. Check for Distraction (Good score but looking away)
    if gaze == "OFF_SCREEN":
        return "Not Engaged: Looking Away", COLOR_AWAY

    # 3. Classified Engaged States based on intensity
    if prob >= 0.85: 
        return "Engaged: Focused", COLOR_FOCUSED
    elif prob >= 0.65: 
        return "Engaged: Confused", COLOR_CONFUSED
    elif prob >= 0.40: 
        return "Engaged: Frustrated", COLOR_FRUSTRATED
    
    return "Not Engaged: Bored", COLOR_BORED

def draw_indicator(frame, box, label, color):
    """Draws the bounding box and text label on the frame."""
    if box is None: return
    
    (x, y, w, h) = box
    # Draw Rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw Label Background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - 25), (x + text_w, y), color, -1)
    
    # Draw Text
    cv2.putText(frame, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ---------------------------
# MAIN APPLICATION
# ---------------------------
st.title("Student Engagement Analyzer")

mode = st.sidebar.radio("Mode", ["Upload Video", "Live Camera"])

# ===========================
# 1. LIVE CAMERA MODE
# ===========================
if mode == "Live Camera":
    st.write("### 🎥 Live Analysis")
    st.info("Check the box to start. Uncheck to stop and view report.")

    run_live = st.checkbox("Start/Stop Live Camera")
    
    frame_window = st.image([])
    metrics_placeholder = st.empty()

    if run_live:
        # Reset session if starting fresh
        if st.session_state.get("just_started", True):
             st.session_state.logs = []
             st.session_state.behavior_history = []
             st.session_state.fusion = AttentionFusion()
             st.session_state.just_started = False

        cap = cv2.VideoCapture(0)
        
        while run_live:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not found.")
                break
            
            # Resolution safety check
            if frame.shape[:2] != (480, 640): pass 

            # Analysis
            gaze = analyze_gaze(frame)
            prob, box = get_engagement_prob(frame)

            # Update Logic
            if box:
                label, color = determine_state_and_color(gaze, prob)
                draw_indicator(frame, box, label, color)
                st.session_state.fusion.update(gaze, prob)
            else:
                st.session_state.fusion.update("ABSENT", None)

            # Metrics
            metrics = st.session_state.fusion.compute_metrics()
            if metrics:
                st.session_state.logs.append(metrics)
                st.session_state.behavior_history = st.session_state.fusion.behavior_history
                metrics_placeholder.markdown(f"**Status:** {metrics['BEHAVIOR']} | **Score:** {metrics['CAS']}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

        cap.release()

    else:
        # STOPPED -> GENERATE REPORT
        if len(st.session_state.logs) > 0:
            st.divider()
            st.session_state.just_started = True # Prepare reset for next run
            
            summary = summarize_session(st.session_state.logs, st.session_state.behavior_history)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Final Engagement Score", summary["Average_CAS"])
                st.metric("Overall Verdict", summary["Final_Verdict"])
            with c2:
                st.dataframe(summary["Behavior_Breakdown"])

            st.subheader("📈 Engagement Over Time")
            st.line_chart([l['CAS'] for l in st.session_state.logs])

# ===========================
# 2. UPLOAD VIDEO MODE
# ===========================
elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload Lecture Video (MP4)", type=["mp4", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.info("Video Uploaded.")
        
        if st.button("🚀 Process Video Now"):
            cap = cv2.VideoCapture(video_path)
            
            # Video Properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Output Setup (H.264 for Browser Support)
            output_path = os.path.join(tempfile.gettempdir(), "processed_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Analysis Init
            fusion = AttentionFusion(window_seconds=5, fps=fps)
            logs = []
            
            # Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # OPTIMIZATION VARS
            frame_count = 0
            SKIP_FRAMES = 3 
            
            # State Memory (for skipped frames)
            last_box = None
            last_label = ""
            last_color = (0, 255, 0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Resize for Fast AI Detection
                small_frame = cv2.resize(frame, (640, 360)) 
                scale_x = width / 640
                scale_y = height / 360

                # --- AI PROCESSING (Every 3rd frame) ---
                if frame_count % SKIP_FRAMES == 0:
                    gaze = analyze_gaze(small_frame)
                    prob, small_box = get_engagement_prob(small_frame)
                    
                    # Update Fusion Stats
                    if small_box:
                        fusion.update(gaze, prob)
                    else:
                        fusion.update("ABSENT", None)

                    metrics = fusion.compute_metrics()
                    if metrics: logs.append(metrics)

                    # Update Drawing Coordinates
                    if small_box is not None:
                        (sx, sy, sw, sh) = small_box
                        # Scale back to original resolution
                        real_box = (int(sx*scale_x), int(sy*scale_y), int(sw*scale_x), int(sh*scale_y))
                        
                        last_box = real_box
                        last_label, last_color = determine_state_and_color(gaze, prob)
                    else:
                        last_box = None
                
                # --- DRAWING (Every frame) ---
                if last_box is not None:
                    draw_indicator(frame, last_box, last_label, last_color)
                
                out.write(frame)
                
                # UI Update
                frame_count += 1
                if frame_count % 50 == 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            
            st.success("Processing Complete!")
            
            # Results
            st.subheader("📽️ Processed Video")
            st.video(output_path)

            st.divider()
            
            # Report
            summary = summarize_session(logs, fusion.behavior_history)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Overall Score", summary["Average_CAS"])
                st.metric("Verdict", summary["Final_Verdict"])
            with c2:
                st.dataframe(summary["Behavior_Breakdown"])
                
            st.subheader("📈 Engagement Timeline")
            st.line_chart([l['CAS'] for l in logs])