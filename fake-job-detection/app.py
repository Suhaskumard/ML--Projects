from flask import Flask, render_template, request, jsonify
import joblib
from preprocess import clean_text

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return '<h2>Fake Job Posting Detection API Running</h2>'

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('description', '')
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return jsonify({'prediction': 'Fake' if pred == 1 else 'Real'})

if __name__ == '__main__':
    app.run(debug=True)

