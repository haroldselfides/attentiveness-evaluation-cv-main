# --- START OF FILE fusion.py ---
# ---------------------------
# FUSION MODULE (UPDATED)
# ---------------------------
from collections import deque
import numpy as np

# --- 6. SPECIFIC THRESHOLDS ---
THRESH_HIGH_ATTENTION = 0.70  # Above this is "High Engagement"
THRESH_LOW_ATTENTION = 0.40   # Below this is "Low Engagement"
THRESH_DROWSY_MODEL = 0.50    # If model prob is low but eyes are on screen

class AttentionFusion:
    def __init__(self, window_seconds=5, fps=30):
        self.window_size = window_seconds * fps
        
        # Buffers
        self.gaze_buffer = deque(maxlen=self.window_size)
        self.engagement_buffer = deque(maxlen=self.window_size)
        
        # To track specific behavior counts for the final report
        self.behavior_history = [] 

    def update(self, gaze_state, engagement_prob):
        """
        gaze_state: "ON_SCREEN", "OFF_SCREEN", or "ABSENT" (New logic handles None)
        engagement_prob: float 0-1 or None
        """
        
        # 1. SPECIFIC INDICATOR OF ENGAGEMENT (Data Normalization)
        if gaze_state == "ON_SCREEN":
            g_val = 1.0
        else:
            g_val = 0.0 # Off screen or Absent

        if engagement_prob is None:
            e_val = 0.0
        else:
            e_val = float(engagement_prob)

        self.gaze_buffer.append(g_val)
        self.engagement_buffer.append(e_val)

        # 4. BEHAVIOR DETERMINATION (Instantaneous)
        # Determine specific behavior for this specific frame
        current_behavior = "Unknown"
        
        if engagement_prob is None:
            current_behavior = "ABSENT" # No face detected
        elif gaze_state == "OFF_SCREEN":
            current_behavior = "DISTRACTED" # Face present, but looking away
        elif gaze_state == "ON_SCREEN" and e_val < THRESH_DROWSY_MODEL:
            current_behavior = "DROWSY/BORED" # Looking at screen, but facial features look disengaged
        elif gaze_state == "ON_SCREEN" and e_val >= THRESH_DROWSY_MODEL:
            current_behavior = "FOCUSED" # Ideally engaged

        self.behavior_history.append(current_behavior)

    def compute_metrics(self):
        if len(self.gaze_buffer) == 0:
            return None

        ogr = np.mean(self.gaze_buffer) # On-Gaze Ratio
        ep = np.mean(self.engagement_buffer) # Engagement Probability

        # 3. SPECIFIC CRITERION OF ENGAGEMENT
        # Formula: 60% Facial Feature Score + 40% Gaze Direction
        cas = (0.6 * ep) + (0.4 * ogr)

        # Determine State based on Thresholds
        if cas >= THRESH_HIGH_ATTENTION:
            state = "HIGHLY ENGAGED"
        elif cas >= THRESH_LOW_ATTENTION:
            state = "PARTIALLY ENGAGED"
        else:
            state = "DISENGAGED"

        # Get the most recent behavior added
        current_behavior = self.behavior_history[-1] if self.behavior_history else "Unknown"

        return {
            "OGR": round(ogr, 2),
            "EP": round(ep, 2),
            "CAS": round(cas, 2),
            "STATE": state,
            "BEHAVIOR": current_behavior
        }

def summarize_session(logs, behavior_history):
    """
    5. FACTORS ON SAYING VIDEO IS ENGAGED OR NOT
    Generates a final report based on the % of time spent in each state.
    """
    if not logs:
        return None

    # Calculate Average Scores
    avg_cas = sum(l["CAS"] for l in logs) / len(logs)
    
    # Calculate Behavior Percentages
    total_frames = len(behavior_history)
    counts = {
        "FOCUSED": behavior_history.count("FOCUSED"),
        "DISTRACTED": behavior_history.count("DISTRACTED"),
        "DROWSY/BORED": behavior_history.count("DROWSY/BORED"),
        "ABSENT": behavior_history.count("ABSENT")
    }
    
    percentages = {k: round((v/total_frames)*100, 1) for k,v in counts.items()}

    # Final Verdict Logic
    # If Focused > 60%, the session is considered "Effective"
    if percentages["FOCUSED"] > 60:
        verdict = "EFFECTIVE LECTURE ENGAGEMENT"
    elif percentages["FOCUSED"] > 30:
        verdict = "MODERATE ENGAGEMENT (Needs Improvement)"
    else:
        verdict = "INEFFECTIVE / DISENGAGED"

    return {
        "Average_CAS": round(avg_cas, 2),
        "Behavior_Breakdown": percentages,
        "Final_Verdict": verdict
    }