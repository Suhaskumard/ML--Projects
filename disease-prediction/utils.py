import pandas as pd
import pickle

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

def predict_disease(symptoms):
    model = load_model()
    
    # add feature names
    columns = ["fever", "cough", "fatigue"]
    df = pd.DataFrame([symptoms], columns=columns)
    
    prediction = model.predict(df)
    return prediction[0]