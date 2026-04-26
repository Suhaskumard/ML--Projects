import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

def predict_disease(symptoms):
    symptoms = np.array(symptoms).reshape(1, -1)
    prediction = model.predict(symptoms)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    sample = [1, 0, 1, 0, 1]  # example symptoms
    print("Predicted Disease:", predict_disease(sample))

