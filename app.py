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

# Page Config
st.set_page_config(page_title="Student Engagement Analysis", layout="wide")

# Initialize Session State for Live Reporting
if "logs" not in st.session_state:
    st.session_state.logs = []
if "behavior_history" not in st.session_state:
    st.session_state.behavior_history = []
if "fusion" not in st.session_state:
    st.session_state.fusion = AttentionFusion()

# Colors for Indicators
COLOR_FOCUSED = (0, 255, 0)      # Green
COLOR_CONFUSED = (0, 165, 255)   # Orange
COLOR_FRUSTRATED = (0, 0, 255)   # Red
COLOR_BORED = (255, 255, 0)      # Cyan
COLOR_DROWSY = (128, 0, 128)     # Purple
COLOR_AWAY = (0, 0, 0)           # Black

def determine_state_and_color(gaze, prob):
    """Maps Gaze + Score to your specific categories."""
    if gaze == "OFF_SCREEN":
        return "Not Engaged: Looking Away", COLOR_AWAY

    if prob >= 0.85: return "Engaged: Focused", COLOR_FOCUSED
    elif prob >= 0.65: return "Engaged: Confused", COLOR_CONFUSED
    elif prob >= 0.50: return "Engaged: Frustrated", COLOR_FRUSTRATED
    elif prob >= 0.30: return "Not Engaged: Bored", COLOR_BORED
    else: return "Not Engaged: Drowsy", COLOR_DROWSY

def draw_indicator(frame, box, label, color):
    """Draws the box and label on the frame."""
    (x, y, w, h) = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - 25), (x + text_w, y), color, -1)
    cv2.putText(frame, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ---------------------------
# MAIN APP
# ---------------------------
st.title("Student Engagement Analyzer")

mode = st.sidebar.radio("Mode", ["Upload Video", "Live Camera"])

# ---------------------------
# LIVE CAMERA MODE
# ---------------------------
if mode == "Live Camera":
    st.write("### 🎥 Live Analysis")
    st.info("Check the box below to start. Uncheck it to Stop and generate the report.")

    # CONTROL CHECKBOX
    run_live = st.checkbox("Start/Stop Live Camera")
    
    # Placeholders for Layout
    frame_window = st.image([])
    metrics_placeholder = st.empty()

    if run_live:
        # --- STARTING ---
        # If we just started, clear previous logs to start fresh
        if len(st.session_state.logs) > 0 and st.session_state.get("just_started", True):
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
            
            # Crash Fix: Reset gaze if resolution changes
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
                # Handle Absent
                st.session_state.fusion.update("ABSENT", None)

            # Store Metrics in Session State
            metrics = st.session_state.fusion.compute_metrics()
            if metrics:
                st.session_state.logs.append(metrics)
                st.session_state.behavior_history = st.session_state.fusion.behavior_history
                
                # Show simple live stat
                metrics_placeholder.markdown(f"**Status:** {metrics['BEHAVIOR']} | **Score:** {metrics['CAS']}")

            # Convert for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

        cap.release()

    else:
        # --- STOPPED: GENERATE REPORT ---
        # This block runs when the checkbox is UNCHECKED
        if len(st.session_state.logs) > 0:
            st.divider()
            st.success("✅ Session Stopped. Generating Report...")
            
            # Prepare Summary
            summary = summarize_session(st.session_state.logs, st.session_state.behavior_history)
            
            # Columns for Layout
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("📊 Session Summary")
                st.metric("Final Engagement Score", summary["Average_CAS"])
                st.metric("Overall Verdict", summary["Final_Verdict"])
            
            with c2:
                st.subheader("🧠 Behavior Breakdown")
                st.dataframe(summary["Behavior_Breakdown"])

            # Timeline Graph
            st.subheader("📈 Engagement Over Time")
            if len(st.session_state.logs) > 0:
                df = pd.DataFrame(st.session_state.logs)
                df['Time'] = df.index
                st.line_chart(df, x='Time', y='CAS')
            
            # Reset flag so next time we click start, it clears data
            st.session_state.just_started = True

# ---------------------------
# UPLOAD VIDEO MODE
# ---------------------------
elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload Lecture Video (MP4)", type=["mp4", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.info("Video Uploaded.")
        
        if st.button("🚀 Process Video Now"):
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Codec setup (H.264)
            output_path = os.path.join(tempfile.gettempdir(), "processed_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            fusion = AttentionFusion(window_seconds=5, fps=fps)
            logs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            SKIP_FRAMES = 3 
            last_box = None
            last_label = ""
            last_color = (0, 255, 0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Resize for AI
                small_frame = cv2.resize(frame, (640, 360)) 
                scale_x = width / 640
                scale_y = height / 360

                if frame_count % SKIP_FRAMES == 0:
                    gaze = analyze_gaze(small_frame)
                    prob, small_box = get_engagement_prob(small_frame)
                    
                    if small_box:
                        fusion.update(gaze, prob)
                    else:
                        fusion.update("ABSENT", None)

                    metrics = fusion.compute_metrics()
                    if metrics: logs.append(metrics)

                    if small_box is not None:
                        (sx, sy, sw, sh) = small_box
                        real_box = (int(sx*scale_x), int(sy*scale_y), int(sw*scale_x), int(sh*scale_y))
                        last_box = real_box
                        last_label, last_color = determine_state_and_color(gaze, prob)
                    else:
                        last_box = None
                
                if last_box is not None:
                    draw_indicator(frame, last_box, last_label, last_color)
                
                out.write(frame)
                frame_count += 1
                if frame_count % 50 == 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            
            st.success("Processing Complete!")
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
                
            st.line_chart([l['CAS'] for l in logs])