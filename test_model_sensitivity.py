import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('e:/4 year/research/ADHD_Detection_Project_new/ADHD_Backend_Demo/app/ml_model/adhd_model.pkl')
features = list(model.feature_names_in_)

# Brute force check
print("\nBrute Force Check:")
for feature in features:
    data = {col: 0.0 for col in features}
    data[feature] = 100000.0  # Try a huge value
    df_test = pd.DataFrame([data])[features]
    prob = model.predict_proba(df_test)[0][1]
    print(f"{feature}: {prob}")


