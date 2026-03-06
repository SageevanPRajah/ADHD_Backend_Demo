"""
Handwriting ADHD Detection Service
-----------------------------------
Loads the trained RandomForest model, StandardScaler, and feature list,
then extracts behavioral features from raw stroke JSON and predicts
ADHD probability.
"""

import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

# ──────────────────────────── paths ────────────────────────────

_MODEL_DIR = Path(__file__).resolve().parent.parent / "Handwriting_ml_model"

# ──────────────────────────── load artefacts ───────────────────

_model = joblib.load(_MODEL_DIR / "adhd_model.pkl")
_scaler = joblib.load(_MODEL_DIR / "scaler.pkl")

with open(_MODEL_DIR / "features.pkl", "rb") as _f:
    _feature_order: List[str] = pickle.load(_f)


# ──────────────────────────── helpers ──────────────────────────

def _group_strokes(strokes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split flat stroke-point list into individual stroke groups
    (each starting with a 'start' point followed by 'move' points)."""
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] | None = None

    for pt in strokes:
        if pt["type"] == "start":
            if current and len(current) > 1:
                groups.append(current)
            current = [pt]
        elif pt["type"] == "move" and current is not None:
            current.append(pt)

    if current and len(current) > 1:
        groups.append(current)

    return groups


def _distance(p1: Dict, p2: Dict) -> float:
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    return math.sqrt(dx * dx + dy * dy)


def _stroke_length(points: List[Dict]) -> float:
    return sum(_distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def _stroke_duration(points: List[Dict]) -> float:
    """Duration in seconds."""
    if len(points) < 2:
        return 0.0
    return max(0.001, (points[-1]["timestamp"] - points[0]["timestamp"]) / 1000.0)


def _direction_angle(p1: Dict, p2: Dict) -> float:
    return math.atan2(p2["y"] - p1["y"], p2["x"] - p1["x"])


# ──────────────────────────── feature extraction ───────────────

def extract_features(strokes: List[Dict[str, Any]], pen_size: float = 8.0) -> Dict[str, float]:
    """
    Extract the 11 behavioural handwriting features that the model expects.

    Because the HTML canvas does not provide real pressure data, we approximate
    pressure from point-density within each stroke (points per pixel of stroke length).
    """

    groups = _group_strokes(strokes)
    num_strokes = len(groups)

    if num_strokes == 0:
        # Fallback: return all zeros (model will still predict)
        return {f: 0.0 for f in _feature_order}

    # ---- per-stroke metrics ----
    lengths: List[float] = []
    durations: List[float] = []
    pressures: List[float] = []

    for g in groups:
        sl = _stroke_length(g)
        sd = _stroke_duration(g)
        lengths.append(sl)
        durations.append(sd)

        # Approximate pressure as point-density (points per pixel).
        # More points in a shorter distance → higher "pressure" / slower drawing.
        density = len(g) / max(sl, 1.0)
        # Normalise density to a 0-1-ish range (pen_size acts as a baseline)
        pressure = min(1.0, density * (pen_size / 8.0))
        pressures.append(pressure)

    # ---- aggregate metrics ----
    total_strokes = num_strokes

    # Activity duration (seconds) – first point to last point in the session
    first_ts = strokes[0]["timestamp"]
    last_ts = strokes[-1]["timestamp"]
    activity_duration = max(0.1, (last_ts - first_ts) / 1000.0)

    avg_stroke_length = float(np.mean(lengths))
    max_stroke_length = float(np.max(lengths))
    avg_stroke_duration = float(np.mean(durations))

    completion_speed = avg_stroke_length / activity_duration if activity_duration > 0 else 0.0

    # Pause count: gaps > 200 ms between consecutive strokes
    pause_count = 0
    for i in range(len(groups) - 1):
        end_ts = groups[i][-1]["timestamp"]
        start_ts = groups[i + 1][0]["timestamp"]
        gap_s = (start_ts - end_ts) / 1000.0
        if gap_s > 0.2:
            pause_count += 1

    avg_pressure = float(np.mean(pressures))
    max_pressure = float(np.max(pressures))
    min_pressure = float(np.min(pressures))
    pressure_variation = max_pressure - min_pressure

    return {
        "total_strokes": total_strokes,
        "activity_duration": activity_duration,
        "completion_speed": completion_speed,
        "pause_count": pause_count,
        "pressure_variation": pressure_variation,
        "avg_pressure": avg_pressure,
        "max_pressure": max_pressure,
        "min_pressure": min_pressure,
        "avg_stroke_length": avg_stroke_length,
        "max_stroke_length": max_stroke_length,
        "avg_stroke_duration": avg_stroke_duration,
    }


# ──────────────────────────── prediction ───────────────────────

def predict_adhd(strokes: List[Dict[str, Any]], pen_size: float = 8.0) -> Dict[str, Any]:
    """
    End-to-end prediction: extract features → scale → predict.

    Returns dict with:
      prediction      – "ADHD Risk" | "No ADHD Risk"
      probability     – float  (ADHD class probability 0-1)
      risk_level      – "High" | "Moderate" | "Low"
      adhd_probability – same as probability (for convenience)
      risk_score      – human-readable percentage string e.g. "67%"
    """
    features = extract_features(strokes, pen_size)

    # Build feature vector in the exact order the model was trained on
    feature_vector = [features[f] for f in _feature_order]
    X = np.array([feature_vector])

    # Scale
    X_scaled = _scaler.transform(X)

    # Predict – class 1 = ADHD
    proba = _model.predict_proba(X_scaled)[0]
    adhd_prob = float(proba[1])  # probability of class 1 (ADHD)

    # Classify risk level
    if adhd_prob >= 0.7:
        risk_level = "High"
    elif adhd_prob >= 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    prediction = "ADHD Risk" if adhd_prob >= 0.5 else "No ADHD Risk"

    return {
        "prediction": prediction,
        "probability": adhd_prob,
        "risk_level": risk_level,
        "adhd_probability": adhd_prob,
        "risk_score": f"{round(adhd_prob * 100)}%",
    }
