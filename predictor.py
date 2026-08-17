"""
predictor.py
------------
STEP 12a : MediaPipe Integration + Inference Wrapper

Loads the saved model, scaler, and label encoder, and exposes a clean
`GesturePredictor` class used by app.py for both:
  - Webcam frame prediction
  - Uploaded image prediction

Keeping this logic separate from app.py keeps the Streamlit UI code clean
and makes the prediction logic independently testable/reusable.
"""

import numpy as np
import cv2
import mediapipe as mp
import joblib
import json
from tensorflow import keras

MODEL_PATH = "models/gesture_ann_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURES_PATH = "models/feature_columns.json"


class GesturePredictor:
    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoder = joblib.load(ENCODER_PATH)
        with open(FEATURES_PATH) as f:
            self.feature_columns = json.load(f)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
        )

    def extract_landmarks(self, image_bgr):
        """
        Runs MediaPipe Hands on a BGR image (as returned by cv2 / uploaded file).
        Returns (feature_vector, annotated_image, hand_landmarks) or
        (None, original_image, None) if no hand is detected.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        annotated = image_bgr.copy()

        if not result.multi_hand_landmarks:
            return None, annotated, None

        hand_landmarks = result.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        return np.array(features).reshape(1, -1), annotated, hand_landmarks

    def predict(self, image_bgr):
        """
        Full pipeline: detect hand -> extract landmarks -> scale -> predict.
        Returns dict with gesture name, confidence, full probability breakdown,
        and the annotated image (with landmarks drawn) for display.
        """
        features, annotated, _ = self.extract_landmarks(image_bgr)

        if features is None:
            return {
                "success": False,
                "message": "No hand detected. Please show your hand clearly to the camera.",
                "annotated_image": annotated,
            }

        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict(features_scaled, verbose=0)[0]

        pred_idx = int(np.argmax(probabilities))
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probabilities[pred_idx])

        prob_breakdown = {
            self.label_encoder.classes_[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        return {
            "success": True,
            "gesture": pred_label,
            "confidence": confidence,
            "probabilities": prob_breakdown,
            "annotated_image": annotated,
        }
