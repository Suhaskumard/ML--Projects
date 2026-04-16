import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()

    # Drop ID
    df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Feature Engineering
    df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col != "Churn":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'AvgCharges']]
    y = df["Churn"]

    return X, y, encoders


