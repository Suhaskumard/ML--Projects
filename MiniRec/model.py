import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_similarity(matrix):
    return cosine_similarity(matrix)

def item_similarity(matrix):
    return cosine_similarity(matrix.T)

def predict_user_based(matrix, similarity):
    sim_sum = np.abs(similarity).sum(axis=1).reshape(-1, 1)
    # Avoid division by zero
    sim_sum = np.where(sim_sum == 0, 1, sim_sum)
    return similarity.dot(matrix) / sim_sum

def predict_item_based(matrix, similarity):
    sim_sum = np.abs(similarity).sum(axis=1)
    # Avoid division by zero
    sim_sum = np.where(sim_sum == 0, 1, sim_sum)
    return matrix.dot(similarity) / sim_sum

