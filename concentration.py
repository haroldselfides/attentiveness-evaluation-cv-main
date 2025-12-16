# --- START OF FILE concentration.py ---
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "concentration_model.keras")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

def get_engagement_prob(frame):
    """
    Returns:
        (prob, box)
        prob: float 0-1
        box: tuple (x, y, w, h) or None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return None, None

    # Get largest face
    (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
    face_roi = frame[y:y+h, x:x+w]

    if face_roi.size == 0:
        return None, None

    # Preprocess
    face_roi = cv2.resize(face_roi, (224, 224))
    face_roi = face_roi.astype("float32")
    face_roi = preprocess_input(face_roi)
    face_roi = np.expand_dims(face_roi, axis=0)

    prob = model.predict(face_roi, verbose=0)[0][0]
    
    # RETURN BOTH PROBABILITY AND BOX COORDINATES
    return float(prob), (x, y, w, h)