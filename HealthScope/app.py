from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "Disease Prediction System Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["symptoms"]
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    app.run(debug=True)

