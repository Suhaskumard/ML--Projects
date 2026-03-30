import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class MovieRecommender:
    def __init__(self, csv_path='movies.csv'):
        """
        Initialize the movie recommender with movie data.
        """
        self.movies = pd.read_csv(csv_path)
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._build_model()
    
    def _preprocess_text(self, text):
        """
        Preprocess movie title and genres for TF-IDF.
        """
        # Combine title and genres, lowercase, remove special chars
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s|]', '', text)
        return text
    
    def _build_model(self):
        """
        Build TF-IDF matrix and cosine similarity matrix.
        """
        # Combine title and genres
        self.movies['combined'] = self.movies['title'] + ' ' + self.movies['genres']
        
        # Preprocess
        self.movies['combined'] = self.movies['combined'].apply(self._preprocess_text)
        
        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies['combined'])
        
        # Cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
    
    def find_movie_index(self, movie_name):
        """
        Find movie index with partial, case-insensitive search.
        Returns first match.
        """
        movie_name = movie_name.lower()
        matches = self.movies['title'].str.lower().str.contains(movie_name, na=False)
        indices = self.movies[matches].index.tolist()
        if indices:
            return indices[0]
        return None
    
    def recommend_movies(self, movie_name, top_n=5):
        """
        Recommend top N similar movies for given movie name.
        """
        idx = self.find_movie_index(movie_name)
        if idx is None:
            return f"Movie '{movie_name}' not found in the dataset."
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Top N excluding itself
        sim_scores = sim_scores[1:top_n+1]
        
        rec_movies = []
        for i, score in sim_scores:
            rec_movies.append({
                'title': self.movies.iloc[i]['title'],
                'genres': self.movies.iloc[i]['genres'],
                'similarity': round(score, 4)
            })
        
        return rec_movies

def main():
    """
    Main function to demonstrate the recommender.
    """
    recommender = MovieRecommender('movies.csv')
    
    print("=== Movie Recommendation System ===\n")
    
    while True:
        movie_input = input("Enter a movie name (or 'quit' to exit): ").strip()
        if movie_input.lower() == 'quit':
            break
        
        recommendations = recommender.recommend_movies(movie_input, top_n=5)
        
        if isinstance(recommendations, str):
            print(f"{recommendations}\n")
        else:
            print(f"\nTop 5 recommendations for '{movie_input}':\n")
            print("-" * 60)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']} ({rec['genres']})")
                print(f"   Similarity Score: {rec['similarity']}")
            print("-" * 60)
            print()

if __name__ == "__main__":
    main()

