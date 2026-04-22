import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import preprocess_data, reshape_for_lstm
import os

# Create dirs if needed
os.makedirs("saved_model", exist_ok=True)

# Load model (train first if not exists)
model_path = "saved_model/har_model.h5"
if not os.path.exists(model_path):
    print("❌ Model not found! Run `python train.py` first.")
    exit(1)

print("Loading model...")
model = load_model(model_path)

# Load sample data for prediction
print("Loading test data...")
data = pd.read_csv("data/data.csv")
sample_data = data.drop("activity", axis=1).iloc[:5]  # Predict first 5 samples

print("Sample data shape:", sample_data.shape)

# Preprocess
X_sample = preprocess_data(sample_data.values)
X_sample = reshape_for_lstm(X_sample)

# Predict
print("Predicting...")
predictions = model.predict(X_sample)
predicted_classes = np.argmax(predictions, axis=1)

# Activity mapping
activity_map = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS", 
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}

print("\n📊 Predictions:")
for i, pred in enumerate(predicted_classes):
    print(f"Sample {i+1}: Predicted = {activity_map[pred]} (confidence: {predictions[i][pred]:.2f})")

print("\n✅ Prediction complete!")

