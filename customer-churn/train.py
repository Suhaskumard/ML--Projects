import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, preprocess

# Load data
df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess
X, y, encoders = preprocess(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200),

}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Accuracy: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best Model Saved (Accuracy: {best_score:.4f})")

