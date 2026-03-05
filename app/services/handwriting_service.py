import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict

from app.api.schemas import HandwritingSessionIn

logger = logging.getLogger(__name__)

class HandwritingMLService:
    def __init__(self):
        # Path to models directory
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_model")
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Try loading models on startup
        self.load_models()

    def load_models(self):
        try:
            model_path = os.path.join(self.models_dir, "adhd_model.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            columns_path = os.path.join(self.models_dir, "feature_columns.json")
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                logger.warning(f"Model file not found at {model_path}")
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                logger.warning(f"Scaler file not found at {scaler_path}")
                
            if os.path.exists(columns_path):
                with open(columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
            else:
                logger.warning(f"Feature columns file not found at {columns_path}")
                
            logger.info("Handwriting ML models loaded successfully (if present).")
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")

    def extract_features(self, session: HandwritingSessionIn) -> Dict[str, float]:
        """
        Extract behavioral features from raw stroke data.
        """
        strokes = session.strokes
        
        if not strokes:
            return {}

        total_points = len(strokes)
        
        # Group strokes into individual continuous lines (start -> end)
        continuous_strokes = []
        current_stroke = []
        
        for p in strokes:
            if p.type == 'start':
                current_stroke = [p]
            elif p.type == 'move':
                current_stroke.append(p)
            elif p.type == 'end':
                current_stroke.append(p)
                continuous_strokes.append(current_stroke)
                current_stroke = []
        
        if current_stroke and current_stroke[0].type == 'start':
            continuous_strokes.append(current_stroke)
            
        total_strokes = len(continuous_strokes)
        
        # Calculate features
        total_distance = 0.0
        stroke_lengths = []
        stroke_durations = []
        points_per_stroke = []
        pauses = []
        
        for i, stroke_pts in enumerate(continuous_strokes):
            if len(stroke_pts) < 1:
                continue
            
            points_per_stroke.append(len(stroke_pts))
                
            # Distance and duration of this stroke
            stroke_dist = 0.0
            for j in range(1, len(stroke_pts)):
                p1, p2 = stroke_pts[j-1], stroke_pts[j]
                dist = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                stroke_dist += dist
            
            total_distance += stroke_dist
            stroke_lengths.append(stroke_dist)
            
            duration = (stroke_pts[-1].timestamp - stroke_pts[0].timestamp) / 1000.0 # seconds
            stroke_durations.append(duration)
            
            # Pause before next stroke
            if i < len(continuous_strokes) - 1:
                next_stroke_start = continuous_strokes[i+1][0]
                pause = (next_stroke_start.timestamp - stroke_pts[-1].timestamp) / 1000.0 # seconds
                if pause > 0:
                    pauses.append(pause)
                    
        activity_duration = (strokes[-1].timestamp - strokes[0].timestamp) / 1000.0 if total_points > 1 else 0.1
        
        # 20 Features Identification
        # 1. totalStrokes
        # 2. dataPoints
        # 3. activityDuration
        # 4. avgStrokeLength
        # 5. completionSpeed
        # 6. pauseCount
        # 7. pressureVariation
        # 8. meanStrokeLength
        # 9. stdStrokeLength
        # 10. meanStrokeDuration
        # 11. stdStrokeDuration
        # 12. meanPressure
        # 13. stdPressure
        # 14. pressureRange
        # 15. meanKeyPoints
        # 16. pauseRate
        # 17. pointsPerSecond
        # 18. mode
        # 19. age
        # 20. gender

        # Pressure proxy from penSize
        pen_size = getattr(session, 'penSize', 8.0)
        
        # Mode detection
        mode = 1 # Default to letters
        if hasattr(session, 'activity') and session.activity:
            if len(session.activity) > 1:
                mode = 2 # Words

        features = {
            "totalStrokes": total_strokes,
            "dataPoints": total_points,
            "activityDuration": activity_duration,
            "avgStrokeLength": np.mean(stroke_lengths) if stroke_lengths else 0.0,
            "completionSpeed": total_distance / activity_duration if activity_duration > 0 else 0.0,
            "pauseCount": len(pauses),
            "pressureVariation": 0.0, # Placeholder
            "meanStrokeLength": np.mean(stroke_lengths) if stroke_lengths else 0.0,
            "stdStrokeLength": np.std(stroke_lengths) if stroke_lengths else 0.0,
            "meanStrokeDuration": np.mean(stroke_durations) if stroke_durations else 0.0,
            "stdStrokeDuration": np.std(stroke_durations) if stroke_durations else 0.0,
            "meanPressure": pen_size,
            "stdPressure": 0.0,
            "pressureRange": 0.0,
            "meanKeyPoints": np.mean(points_per_stroke) if points_per_stroke else 0.0,
            "pauseRate": len(pauses) / activity_duration if activity_duration > 0 else 0.0,
            "pointsPerSecond": total_points / activity_duration if activity_duration > 0 else 0.0,
            "mode": mode,
            "age": 8.0, # Default age for screening if not provided
            "gender": 1.0, # 1 for Male, 2 for Female? Usually 1/0 or 1/2.
        }
        
        return features

    def predict_risk(self, session: HandwritingSessionIn):
        """
        Run feature extraction and ML model prediction.
        """
        if not self.model:
            # Fallback if model isn't loaded (e.g., during development without files)
            logger.warning("Model not loaded. Returning dummy prediction.")
            return {
                "prediction": "No model loaded",
                "probability": 0.0,
                "risk_level": "Unknown"
            }
            
        try:
            # 1. Extract features
            features_dict = self.extract_features(session)
            
            # 5. Predict probability
            expected_cols = [
                'totalStrokes', 'dataPoints', 'activityDuration', 'avgStrokeLength', 
                'completionSpeed', 'pauseCount', 'pressureVariation', 'meanStrokeLength', 
                'stdStrokeLength', 'meanStrokeDuration', 'stdStrokeDuration', 'meanPressure', 
                'stdPressure', 'pressureRange', 'meanKeyPoints', 'pauseRate', 
                'pointsPerSecond', 'mode', 'age', 'gender'
            ]
            
            # Reorder explicitly to ensure no issues with DataFrame shape
            df = pd.DataFrame([features_dict])
            
            # Ensure all columns exist
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0.0
                    
            df = df[expected_cols]
            
            # LOGGING FOR DEBUGGING
            logger.info(f"Prediction Request - Strokes: {features_dict.get('totalStrokes')}, Points: {features_dict.get('dataPoints')}")
            logger.info(f"Duration: {features_dict.get('activityDuration'):.2f}s, Speed: {features_dict.get('completionSpeed'):.2f}")
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(df)
                # Assuming ADHD is class index 1
                probability = float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0][0])
            else:
                pred = self.model.predict(df)[0]
                probability = 1.0 if pred == 1 else 0.0

            logger.info(f"Model Probability Result: {probability}")

            # 6. Determine risk level and label
            if probability > 0.7:
                risk_level = "High"
                prediction = "ADHD"
            elif probability > 0.4:
                risk_level = "Moderate"
                prediction = "ADHD Risk"
            else:
                risk_level = "Low"
                prediction = "Typically Developing"
                
            return {
                "prediction": prediction,
                "probability": probability,
                "risk_level": risk_level
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Raise or return error response
            return {
                "prediction": "Error",
                "probability": 0.0,
                "risk_level": "Unknown"
            }

handwriting_service = HandwritingMLService()
