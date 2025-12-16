# --- START OF FILE gaze.py ---

# ------------------------------------------------
# GAZE DETECTION USING CLASSICAL OBJECT DETECTION
# ------------------------------------------------
# determines if person is looking at screen or not (ON_SCREEN or OFF_SCREEN)
import cv2
import numpy as np

# load pre-trained haar cascade classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier("haarcascade_eye.xml")

# preprocessing
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# uses lucas-kanade optical flow to track pupil movement across frames
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

prev_gray = None
pupil_point = None


def analyze_gaze(frame):
    """
    Returns:
        "ON_SCREEN" or "OFF_SCREEN"
    """

    global prev_gray, pupil_point

    # convert frame to grayscale and applied gaussian blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    soft = cv2.GaussianBlur(gray, (3,3), 0)

    # --- FIX FOR CRASH: Check if resolution changed ---
    if prev_gray is not None:
        if prev_gray.shape != soft.shape:
            # Resolution changed (e.g., switched from Cam to Video), reset tracker
            prev_gray = None
            pupil_point = None
    # --------------------------------------------------

    # detects the face in the frame (only detects one person)
    faces = face_cascade.detectMultiScale(soft, 1.2, 5)
    detected_point = None
    
    for (x, y, w, h) in faces:
        # extract face region
        roi_gray = soft[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.15, 5)

        for (ex, ey, ew, eh) in eyes:
            # detect eye region
            if ex < 0 or ey < 0 or ex+ew > w or ey+eh > h:
                continue

            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            # preprocessing of eye region for pupil detection
            eye_blur = cv2.GaussianBlur(eye_gray, (3,3), 0)
            eye_eq = clahe.apply(eye_blur)

            # applied thresholding to isolate dark pupil region
            _, thresh = cv2.threshold(
                eye_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel_small, iterations=1
            )

            # finds contours detecting the pupil region
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            if contours:
                cnt = contours[0]
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                if ew * 0.05 < radius < ew * 0.45:
                    detected_point = np.array(
                        [[cx + ex + x, cy + ey + y]],
                        dtype=np.float32
                    )
                    break
    
    # still uses LK for pupil tracking (avoids repeated detection every frame)
    if prev_gray is not None and pupil_point is not None:
        try:
            new_point, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, soft, pupil_point, None, **lk_params
            )
            if status[0][0] == 1:
                pupil_point = new_point
                prev_gray = soft.copy()
                return "ON_SCREEN"
        except Exception:
            # If tracking fails for any other reason, reset and continue
            pupil_point = None

    if detected_point is not None:
        pupil_point = detected_point
        prev_gray = soft.copy()
        return "ON_SCREEN"

    prev_gray = soft.copy()
    return "OFF_SCREEN"