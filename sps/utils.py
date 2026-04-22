import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def preprocess_data(X):
    """Scale features using StandardScaler"""
    return scaler.fit_transform(X)

def reshape_for_lstm(X, timesteps=1):
    """Reshape data for LSTM input (samples, timesteps, features)"""
    return np.reshape(X, (X.shape[0], timesteps, X.shape[1]))

