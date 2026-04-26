import pandas as pd
import numpy as np

def load_data():
    np.random.seed(42)

    users = np.arange(1, 21)      # 20 users
    items = np.arange(101, 121)   # 20 items

    data = []
    for user in users:
        for item in np.random.choice(items, size=10, replace=False):
            rating = np.random.randint(1, 6)
            data.append([user, item, rating])

    df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])
    return df

def create_matrix(df):
    matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')
    return matrix.fillna(0)

