import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample dataset (you can replace with real CSV)
data = {
    "fever": [1,1,0,0,1],
    "cough": [1,0,1,0,1],
    "fatigue": [1,1,0,0,1],
    "disease": ["Flu","Cold","Allergy","Healthy","Flu"]
}

df = pd.DataFrame(data)

X = df.drop("disease", axis=1)
y = df["disease"]

model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
