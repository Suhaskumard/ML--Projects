import numpy as np
from sklearn.metrics import mean_squared_error

def compute_rmse(predictions, actual_matrix):
    mask = actual_matrix > 0
    pred = predictions[mask]
    actual = actual_matrix[mask]

    return np.sqrt(mean_squared_error(actual, pred))

