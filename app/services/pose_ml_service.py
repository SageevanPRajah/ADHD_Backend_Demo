import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
import tempfile

# -------------------------
# Load Model & Scaler
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Pose_model", "rf_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "Pose_model", "scaler.joblib")

rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

mp_pose = mp.solutions.pose


# -------------------------
# Landmark Selection (Same as Training)
# -------------------------

LANDMARK_IDS = [
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24   # right_hip
]


# -------------------------
# Feature Extraction
# -------------------------

def extract_features_from_sequence(seq):
    """
    seq: numpy array (T, 16) → 8 landmarks * (x,y)
    """

    T = seq.shape[0]
    if T < 5:
        raise ValueError("Not enough pose frames")

    # velocity
    velocities = np.linalg.norm(np.diff(seq, axis=0), axis=1)

    mean_velocity = float(np.mean(velocities))
    max_velocity = float(np.max(velocities))
    std_velocity = float(np.std(velocities))

    # fidget (wrist + elbow movement)
    fidget_score = float(np.mean(np.abs(np.diff(seq[:, 4:12], axis=0))))

    # sway (shoulder midpoint)
    mid_shoulders = (seq[:, 0:2] + seq[:, 2:4]) / 2
    sway_x = float(np.std(mid_shoulders[:, 0]))
    sway_y = float(np.std(mid_shoulders[:, 1]))

    # mean absolute displacement (average frame-to-frame movement across all landmarks)
    displacements = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    mean_abs_displacement = float(np.mean(displacements))

    # stability (inverse of movement)
    stability_score = float(1.0 / (1.0 + mean_velocity))

    return {
        "fidget_score": fidget_score,
        "sway_x": sway_x,
        "sway_y": sway_y,
        "mean_velocity": mean_velocity,
        "max_velocity": max_velocity,
        "std_velocity": std_velocity,
        "mean_abs_displacement": mean_abs_displacement,
        "stability_score": stability_score,
    }


# -------------------------
# Video → Pose Sequence
# -------------------------

def extract_pose_sequence(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    sequence = []

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                row = []
                for idx in LANDMARK_IDS:
                    lm = results.pose_landmarks.landmark[idx]
                    row.extend([lm.x, lm.y])
                sequence.append(row)

    cap.release()

    if len(sequence) < 10:
        raise ValueError("Pose detection failed or insufficient frames")

    return np.array(sequence, dtype=np.float32)


# -------------------------
# Main Prediction Function
# -------------------------

def predict_from_video_file(video_file_bytes):

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(video_file_bytes)
        tmp_path = tmp.name

    try:
        seq = extract_pose_sequence(tmp_path)
        features = extract_features_from_sequence(seq)

        # Order must match the scaler/model training order exactly
        X = np.array([[
            features["fidget_score"],
            features["sway_x"],
            features["sway_y"],
            features["mean_velocity"],
            features["max_velocity"],
            features["std_velocity"],
            features["mean_abs_displacement"],
            features["stability_score"],
        ]])

        X_scaled = scaler.transform(X)
        prob = rf_model.predict_proba(X_scaled)[0][1]

        # Subtype Categorization Logic
        if prob < 0.4:
            subtype = "Low Risk (Control)"
        elif prob > 0.8:
            subtype = "Combined (High Risk)"
        elif features["fidget_score"] > 0.05:  # Arbitrary threshold based on current training data range
            subtype = "Hyperactive-Impulsive"
        else:
            subtype = "Inattentive"

        result = {
            "adhd_score": round(float(prob * 10), 2),
            "adhd_probability": round(float(prob), 4),
            "subtype": subtype,
            "derived_features": features
        }

        return result

    finally:
        os.remove(tmp_path)