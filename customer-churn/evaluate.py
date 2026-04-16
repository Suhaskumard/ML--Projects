from sklearn.metrics import classification_report, confusion_matrix
import pickle
from preprocess import load_data, preprocess
from sklearn.model_selection import train_test_split

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load & preprocess data
df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X, y, _ = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Predictions
preds = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

