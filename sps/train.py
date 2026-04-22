import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_data, reshape_for_lstm
from model import build_model
from tensorflow.keras.models import save_model
import os

print("Loading data...")
data = pd.read_csv("data/data.csv")

X = data.drop("activity", axis=1).values
y = data["activity"].values

print(f"Data shape: X={X.shape}, y={y.shape}")

# Encode labels if needed (already numeric)
y = y.astype(int)
num_classes = len(set(y))

print(f"Number of classes: {num_classes}")

# Preprocess
print("Preprocessing...")
X = preprocess_data(X)
X = reshape_for_lstm(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Build model
print("Building model...")
model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)

# Train
print("Training model...")
history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    verbose=1
)

# Create saved_model dir
os.makedirs("saved_model", exist_ok=True)

# Save model
model.save("saved_model/har_model.h5")
print("\n✅ Model trained and saved to saved_model/har_model.h5!")
print("Model accuracy on test set:", max(history.history['val_accuracy']))

