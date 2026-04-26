from sklearn.model_selection import train_test_split

def train_test_split_df(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test

