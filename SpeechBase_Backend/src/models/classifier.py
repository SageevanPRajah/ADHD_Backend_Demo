import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Placeholder model
class ADHDClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Try to load trained model if exists
        try:
            with open('models/adhd_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
                self.is_trained = True
        except:
            pass
    
    def classify(self, features):
        # If model is not trained, use dummy classification
        if not self.is_trained:
            # Simple heuristic-based classification for demonstration
            # Check for indicators: low lexical diversity, high fillers, low coherence
            lexical_div = features.get('lexical_diversity', 0.5)
            coherence = features.get('coherence', 0.5)
            fillers = features.get('fillers', 0)
            
            # Simple scoring
            score = 0
            if lexical_div < 0.3:
                score += 0.3
            if coherence < 0.4:
                score += 0.3
            if fillers > 5:
                score += 0.2
            
            # Add some randomness for variability
            import random
            score += random.uniform(0, 0.2)
            
            prob = min(max(score, 0.1), 0.9)  # Keep between 0.1 and 0.9
            adhd = prob > 0.5
            
            return {'probability': prob, 'adhd': adhd}
        
        # If model is trained, use it
        try:
            # Extract features dynamically based on what's available
            feature_keys = sorted([k for k in features.keys() if isinstance(features[k], (int, float))])
            feature_array = np.array([features[k] for k in feature_keys]).reshape(1, -1)
            
            # Check if dimensions match
            if hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                if feature_array.shape[1] != expected_features:
                    # Pad or trim to match expected features
                    if feature_array.shape[1] < expected_features:
                        # Pad with zeros
                        padding = np.zeros((1, expected_features - feature_array.shape[1]))
                        feature_array = np.hstack([feature_array, padding])
                    else:
                        # Trim
                        feature_array = feature_array[:, :expected_features]
            
            # Predict
            prob = self.model.predict_proba(feature_array)[0][1]
            adhd = prob > 0.5
            
            return {'probability': prob, 'adhd': adhd}
        except Exception as e:
            # Fallback to heuristic if prediction fails
            print(f"Prediction error: {e}")
            lexical_div = features.get('lexical_diversity', 0.5)
            coherence = features.get('coherence', 0.5)
            fillers = features.get('fillers', 0)
            
            score = 0
            if lexical_div < 0.3:
                score += 0.3
            if coherence < 0.4:
                score += 0.3
            if fillers > 5:
                score += 0.2
            
            prob = min(max(score + 0.1, 0.1), 0.9)
            adhd = prob > 0.5
            
            return {'probability': prob, 'adhd': adhd}

classifier = ADHDClassifier()

def classify_adhd(features):
    return classifier.classify(features)