print("predict")

import pickle
import numpy as np

# Load scaler and best model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Example new sample
new_sample = [[40.6918,12.9648,2.7493,0.4789,0.2954,14.5797,13.8019,3.2738,43.0868,134.069]]
new_sample_scaled = scaler.transform(new_sample)

# Predict
prediction = best_model.predict(new_sample_scaled)
print("Predicted class:", prediction[0])