# Training script
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Placeholder features: 13 mfccs + pitch + jitter + shimmer + lexical_diversity + coherence + fillers = 18
num_features = 18

# Placeholder
X = np.random.rand(100, num_features)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Save model
with open('models/adhd_model.pkl', 'wb') as f:
    pickle.dump(model, f)