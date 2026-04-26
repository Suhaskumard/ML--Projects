# 🎬 MiniRec - Movie Recommendation System

A lightweight collaborative filtering-based movie recommendation engine built with Python, NumPy, and scikit-learn.

## 📌 Overview

MiniRec uses **user-based collaborative filtering** to recommend movies to users based on similarity patterns. The system analyzes user rating patterns to find similar users and predict ratings for unwatched movies.

## 🏗️ Architecture

```
MiniRec/
├── app.py              # Main application entry point
├── data_loader.py      # Data loading and matrix creation
├── preprocessing.py    # Data preprocessing & train/test split
├── model.py            # Similarity computation & prediction algorithms
├── recommender.py      # Top-N recommendation generation
├── evaluation.py       # Model evaluation metrics (RMSE)
└── requirements.txt   # Project dependencies
```

## 🔧 Features

- **User Similarity** — Cosine similarity between users
- **Item Similarity** — Cosine similarity between items  
- **User-Based Predictions** — Predict ratings using similar users
- **Item-Based Predictions** — Predict ratings using similar items
- **Top-N Recommendations** — Get N best movie recommendations
- **RMSE Evaluation** — Measure prediction accuracy

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the recommender
python app.py
```

## 💻 Usage

1. Run `python app.py`
2. Enter a user index when prompted
3. Get top 5 movie recommendations for that user
4. Enter `-1` to exit

## 📊 How It Works

1. **Load Data** — Load user-movie rating data
2. **Create Matrix** — Build user-item rating matrix
3. **Compute Similarity** — Calculate cosine similarity between users
4. **Generate Predictions** — Predict ratings for unrated items
5. **Evaluate** — Compute RMSE to measure accuracy
6. **Recommend** — Return top-N unwatched movies with highest predicted ratings

## 📦 Dependencies

- `numpy`
- `scikit-learn`
- `pandas`

## 🎯 Example Output

```
🚀 RecoSense Pro Running...

📊 Computing similarity...
⚙️ Generating predictions...
📈 Evaluating model...
RMSE: 0.8234

Enter user index (0 to 942, or -1 to exit): 5

🎯 Recommendations for User 5:
Movie IDs: [123, 456, 789, 101, 202]
```

## 🔬 Model Details

| Component | Description |
|-----------|-------------|
| Similarity Metric | Cosine Similarity |
| Prediction Method | Weighted sum of similar users' ratings |
| Recommendation Strategy | Top-N highest predicted ratings |

---
*Built with ❤️ using collaborative filtering*
- Deploy with FastAPI
- Add matrix factorization (SVD)
- Build frontend UI

