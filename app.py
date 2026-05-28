import streamlit as st
import cv2
import numpy as np
import threading
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model
eye_model = load_model("eye_cnn.h5")

st.title("🚗 Driver Drowsiness Detection System")

st.write("Real-time AI system using CNN + Eye Aspect Ratio")

# Alarm
def play_alarm():
    try:
        from playsound import playsound
        playsound("alarm.wav")
    except:
        st.warning("Alarm failed")

alarm_thread = None

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye):
    p1, p2 = landmarks[eye[1]], landmarks[eye[5]]
    p3, p4 = landmarks[eye[2]], landmarks[eye[4]]
    p5, p6 = landmarks[eye[0]], landmarks[eye[3]]
    return (euclidean(p2, p4) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p6))

class Detector(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.frame_count = 0

    def transform(self, frame):
        global alarm_thread

        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                h, w, _ = img.shape
                landmarks = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

                left_eye = [362, 385, 387, 263, 373, 380]
                right_eye = [33, 160, 158, 133, 153, 144]

                left_ear = calculate_ear(landmarks, left_eye)
                right_ear = calculate_ear(landmarks, right_eye)

                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    self.frame_count += 1
                else:
                    self.frame_count = 0

                if self.frame_count >= CONSEC_FRAMES:
                    cv2.putText(img, "DROWSY!", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if alarm_thread is None or not alarm_thread.is_alive():
                        alarm_thread = threading.Thread(target=play_alarm)
                        alarm_thread.start()

                cv2.putText(img, f"EAR: {ear:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

webrtc_streamer(key="drowsiness", video_transformer_factory=Detector)
