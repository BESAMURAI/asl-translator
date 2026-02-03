
"""
Simple Flask API that takes an image frame, extracts MediaPipe hand landmarks,
and returns the predicted ASL character using the trained RandomForest model.
Serves the frontend when deployed (e.g. on Render).
"""

import io
import joblib
import numpy as np
import os
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp


# Paths relative to this file (backend/)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "asl_model.pkl")
FRONTEND_DIR = os.path.join(_BASE_DIR, "..", "frontend")

app = Flask(__name__)
CORS(app)

# Load model once at startup
model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist (landmark 0) to make them scale and position invariant."""
    wrist = landmarks[0]
    normalized = []
    for lm in landmarks:
        normalized.append([lm.x - wrist.x, lm.y - wrist.y])
    return normalized


def extract_landmarks(img_bgr):
    """Return flattened (x,y) landmarks for a single hand or None if not found."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    
    # Extract raw landmarks (same format as training: x0, y0, x1, y1, ..., x20, y20)
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y])
    
    # Convert to DataFrame with proper feature names to match training
    feature_names = []
    for i in range(21):
        feature_names.extend([f"x{i}", f"y{i}"])
    df = pd.DataFrame([coords], columns=feature_names)
    
    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    # Expect multipart/form-data with field name "frame"
    if "frame" not in request.files:
        return jsonify({"error": "missing frame"}), 400

    file = request.files["frame"]
    img_bytes = file.read()

    # Decode image bytes to OpenCV BGR
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid image"}), 400

    landmarks = extract_landmarks(img)
    if landmarks is None:
        return jsonify({"prediction": None, "message": "no hand detected"}), 200

    try:
        probs = model.predict_proba(landmarks)[0]
        best_idx = int(np.argmax(probs))
        best_label = model.classes_[best_idx]
        best_prob = float(probs[best_idx])
        
        # Only return prediction if confidence is above threshold
        confidence_threshold = 0.3  # Adjust this based on your needs
        
        if best_prob < confidence_threshold:
            return jsonify({
                "prediction": None,
                "message": f"low confidence ({best_prob:.2f})",
                "top_predictions": [
                    {"label": model.classes_[i], "confidence": float(probs[i])}
                    for i in np.argsort(probs)[-3:][::-1]
                ]
            }), 200

        return jsonify({
            "prediction": best_label,
            "confidence": best_prob,
            "top_predictions": [
                {"label": model.classes_[i], "confidence": float(probs[i])}
                for i in np.argsort(probs)[-3:][::-1]
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve frontend (for Render / single-service deploy)
@app.route("/")
def index():
    if os.path.isdir(FRONTEND_DIR):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"message": "ASL Translator API", "docs": "/health, POST /predict"})


@app.route("/<path:path>")
def frontend_static(path):
    # Serve known frontend files + anything under assets/ (e.g. assets/asl_gifs/A.jpg)
    if not os.path.isdir(FRONTEND_DIR):
        return jsonify({"error": "Not found"}), 404
    allowed = ("index.html", "script.js", "style.css")
    if path in allowed:
        return send_from_directory(FRONTEND_DIR, path)
    # Allow assets/ folder (for Text-to-Sign letter images)
    if path.startswith("assets/") and ".." not in path:
        return send_from_directory(FRONTEND_DIR, path)
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

