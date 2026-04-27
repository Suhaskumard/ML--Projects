# Fake Job Posting Detection System

## Overview
An advanced machine learning system that detects fraudulent job postings using NLP, metadata-based fraud signals, anomaly detection, and explainable AI. The system uses a Logistic Regression classifier with TF-IDF vectorization and provides confidence scores along with human-readable explanations for each prediction.

## Features
- **Text Preprocessing** — Lowercasing, URL removal, punctuation/digit stripping, whitespace normalization
- **TF-IDF Vectorization** — Converts job descriptions into numerical features (max 5000 features)
- **Logistic Regression Classifier** — Fast and interpretable binary classification
- **Metadata Feature Engineering** — Extracts fraud signals: email presence, salary mentions, urgency keywords, external link counts
- **Explainable AI** — Rule-based explainer provides human-readable fraud reasons
- **Terminal Demo Mode** — Run predictions directly in terminal without starting a server
- **Flask REST API** — Optional HTTP server for production integration
- **Confidence Scoring** — Returns prediction probability for risk assessment

## Project Structure

```
fake-job-detection/
├── app.py                    # Flask REST API server
├── predict_terminal.py       # Terminal demo script
├── train_model.py            # Model training pipeline
├── preprocess.py             # Text cleaning utilities
├── feature_engineering.py    # Fraud signal extraction
├── explainability.py         # AI explanation engine
├── utils.py                  # Helper functions
├── requirements.txt          # Python dependencies
├── fake_job_postings.csv     # Demo dataset
├── model.pkl                 # Trained model (generated)
└── vectorizer.pkl            # TF-IDF vectorizer (generated)
```

## Dataset
- **Demo Dataset**: `fake_job_postings.csv` (included for testing)
- **Full Dataset**: Download from [Kaggle - Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

Dataset format:
```csv
description,fraudulent
"Job description text...",0
"Another job posting...",1
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess `fake_job_postings.csv`
- Train a Logistic Regression classifier
- Evaluate accuracy on test set
- Save `model.pkl` and `vectorizer.pkl`

### 2. Run Terminal Demo (Recommended)

```bash
python predict_terminal.py
```

**Sample Output:**
```
============================================================
Fake Job Posting Detection - Terminal Demo
============================================================

Text: We are hiring software engineers for our downtown office. Competitiv...
Prediction: Real
Confidence: 60.39%
Reasons: ['No major fraud indicators found']
Result: This job posting appears legitimate.
------------------------------------------------------------

Text: Immediate join required for data entry work from home. Earn quick mon...
Prediction: Fake
Confidence: 56.8%
Reasons: ['Contains urgency/scam style wording']
Result: This job posting may be fraudulent. Verify company details.
------------------------------------------------------------
```

### 3. Run Flask API Server (Optional)

```bash
python app.py
```

**API Endpoint:**

`POST /predict`

**Request:**
```json
{
  "description": "Urgent! Work from home opportunity. Make $5000 weekly with no experience."
}
```

**Response:**
```json
{
  "prediction": "Fake"
}
```

## Fraud Detection Signals

The system analyzes multiple signals to detect fake job postings:

| Signal | Description |
|--------|-------------|
| **Urgency Words** | Detects phrases like "urgent", "quick money", "immediate join" |
| **External Links** | Counts HTTP links (too many = suspicious) |
| **Email Presence** | Checks for email addresses in the posting |
| **Salary Mentions** | Detects salary-related keywords or dollar signs |

## Model Performance

The Logistic Regression model achieves high accuracy on the test dataset:
- **Accuracy**: ~100% (on demo dataset)
- **Features**: TF-IDF (5000 max features) + metadata signals
- **Algorithm**: Logistic Regression with 1000 max iterations

## Requirements

- Python 3.8+
- Flask 3.1+
- scikit-learn 1.8+
- pandas 3.0+
- numpy 2.4+
- xgboost 3.2+
- textblob 0.20+
- joblib 1.5+

## How It Works

1. **Text Preprocessing**: Raw job descriptions are cleaned (lowercased, URLs/punctuation/digits removed)
2. **Feature Extraction**: TF-IDF vectorizer transforms text into numerical features
3. **Metadata Analysis**: Additional fraud signals extracted (urgency words, links, email, salary)
4. **Prediction**: Logistic Regression model predicts Fake (1) or Real (0)
5. **Explanation**: Rule-based system provides human-readable fraud reasons
6. **Confidence**: Model probability indicates prediction certainty

## Customization

To use your own dataset:
1. Replace `fake_job_postings.csv` with your data
2. Ensure columns: `description`, `fraudulent`
3. Run `python train_model.py` to retrain

## License

This project is open-source and available for educational and commercial use.

