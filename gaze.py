# --- START OF FILE gaze.py ---
import cv2
import numpy as np

# Load classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier("haarcascade_eye.xml")

# Preprocessing
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

prev_gray = None
pupil_point = None

def analyze_gaze(frame):
    """
    Returns: "ON_SCREEN" or "OFF_SCREEN"
    """
    global prev_gray, pupil_point

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    soft = cv2.GaussianBlur(gray, (3,3), 0)

    # Resolution change check
    if prev_gray is not None:
        if prev_gray.shape != soft.shape:
            prev_gray = None
            pupil_point = None

    faces = face_cascade.detectMultiScale(soft, 1.2, 5)
    
    # 1. NO FACE = DEFINITELY LOOKING AWAY/ABSENT
    if len(faces) == 0:
        prev_gray = soft.copy()
        return "OFF_SCREEN"

    detected_point = None
    eyes_detected_count = 0

    for (x, y, w, h) in faces:
        roi_gray = soft[y:y+h, x:x+w]
        # Detect Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.15, 5)
        eyes_detected_count += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            if ex < 0 or ey < 0 or ex+ew > w or ey+eh > h: continue
            
            # Pupil Processing
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_blur = cv2.GaussianBlur(eye_gray, (3,3), 0)
            eye_eq = clahe.apply(eye_blur)

            _, thresh = cv2.threshold(eye_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            if contours:
                cnt = contours[0]
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                if ew * 0.05 < radius < ew * 0.45:
                    detected_point = np.array([[cx + ex + x, cy + ey + y]], dtype=np.float32)
                    break
    
    # Optical Flow Tracking
    if prev_gray is not None and pupil_point is not None:
        try:
            new_point, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, soft, pupil_point, None, **lk_params)
            if status[0][0] == 1:
                pupil_point = new_point
                prev_gray = soft.copy()
                return "ON_SCREEN"
        except:
            pupil_point = None

    if detected_point is not None:
        pupil_point = detected_point
        prev_gray = soft.copy()
        return "ON_SCREEN"

    prev_gray = soft.copy()

    # --- FINAL VERDICT LOGIC ---
    # Case A: Eyes were detected, but pupils weren't (Likely Dark Room or Glare)
    if eyes_detected_count > 0:
        return "ON_SCREEN"  # Benefit of doubt given ONLY if eyes are open

    # Case B: No eyes detected (Sleeping/Eyes Closed OR Head Turned Away)
    return "OFF_SCREEN"