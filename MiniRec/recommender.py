import numpy as np

def get_top_n_recommendations(user_index, matrix, predictions, n=5):
    user_ratings = matrix[user_index]
    scores = predictions[user_index]

    # Remove already rated items
    unseen_indices = np.where(user_ratings == 0)[0]

    if len(unseen_indices) == 0:
        return []

    unseen_scores = [(i, scores[i]) for i in unseen_indices]
    unseen_scores.sort(key=lambda x: x[1], reverse=True)

    return [item[0] for item in unseen_scores[:n]]

