import joblib
from preprocess import clean_text
from feature_engineering import extract_features
from explainability import explain
from utils import explain_result

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

samples = [
    "We are hiring software engineers for our downtown office. Competitive salary and benefits.",
    "Immediate join required for data entry work from home. Earn quick money daily.",
    "Senior product manager needed at leading fintech startup. Great culture.",
    "Urgent! Work from home opportunity. Make $5000 weekly with no experience.",
]

print("=" * 60)
print("Fake Job Posting Detection - Terminal Demo")
print("=" * 60)

for text in samples:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = round(max(prob) * 100, 2)
    features = extract_features(text)
    reasons = explain(features)
    result = explain_result(pred)

    print(f"\nText: {text[:70]}...")
    print(f"Prediction: {'Fake' if pred == 1 else 'Real'}")
    print(f"Confidence: {confidence}%")
    print(f"Reasons: {reasons}")
    print(f"Result: {result}")
    print("-" * 60)

