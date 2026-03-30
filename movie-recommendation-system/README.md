# Movie Recommendation System

## Overview
A simple content-based movie recommendation system using TF-IDF and cosine similarity. Recommends top 5 similar movies based on title and genres.

## Features
- Content-based filtering using TF-IDF vectorization
- Cosine similarity for movie similarity computation
- Partial, case-insensitive movie name search
- Handles movie not found gracefully
- Interactive console interface
- Beginner-friendly, well-commented code

## Technologies
- Python 3.x
- pandas (data handling)
- scikit-learn (TF-IDF, cosine similarity)
- numpy

## Project Structure
```
movie-recommendation-system/
├── movie_recommender.py  # Main application
├── movies.csv           # Sample movie dataset
└── README.md           # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn numpy
   ```

2. **Run the application:**
   ```bash
   cd "f:/Projects/ML based projects/movie-recommendation-system"
   python movie_recommender.py
   ```

3. **Usage:**
   - Enter movie name (partial search supported, e.g., "Toy", "Batman")
   - View top 5 recommendations with similarity scores
   - Type 'quit' to exit

## Sample Output
```
=== Movie Recommendation System ===

Enter a movie name (or 'quit' to exit): Toy Story

Top 5 recommendations for 'Toy Story':

------------------------------------------------------------
1. Toy Story 2 (Animation|Children's|Comedy)
   Similarity Score: 0.7234
2. Jumanji (Adventure|Children's|Fantasy)
   Similarity Score: 0.4567
3. Tom and Huck (Adventure|Children's)
   Similarity Score: 0.3891
4. Sudden Death (Action|Children's)
   Similarity Score: 0.3124
5. GoldenEye (Action|Adventure|Thriller)
   Similarity Score: 0.2876
------------------------------------------------------------
```

## Dataset
- `movies.csv`: Contains `title` and `genres` columns
- Sample data included (20 movies)
- Easy to extend with larger datasets

## How It Works
1. **Preprocessing**: Combines title + genres, TF-IDF vectorization
2. **Similarity**: Precomputes cosine similarity matrix
3. **Recommendation**: Finds input movie, returns top similar movies
4. **Search**: Fuzzy partial matching on movie titles





