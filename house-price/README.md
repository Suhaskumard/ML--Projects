# 🏠 Production-Ready House Price Prediction

An advanced machine learning project with modular architecture, feature engineering, hyperparameter tuning, and production-ready structure.

---

## 🚀 Highlights

- **Modular codebase** - Separated concerns (data, features, train, predict)
- **Feature engineering pipeline** - Price/sqft, area/bedroom, age factors 
- **Hyperparameter optimization** - GridSearchCV with cross-validation
- **Professional logging** - Structured logs for monitoring
- **Production endpoints** - Train, evaluate, predict separation
- **Resume-ready** - Industry-standard ML project structure

---

## 🛠️ Tech Stack

- Python 3.8+
- Pandas, NumPy
- Scikit-learn (RandomForest + Pipeline)
- Joblib (model persistence)

---

## 📂 Project Structure

```
house-price/
├── config.py              # Central configuration
├── data_loader.py         # Sample data loader
├── feature_engineering.py # Advanced features
├── utils.py               # Logging utilities
├── train.py               # Full ML pipeline + GridSearchCV
├── evaluate.py            # Model evaluation
├── predict.py             # Production prediction
├── README.md              # This file
└── house_price_model.pkl  # Trained model (auto-generated)
```

---

## ⚙️ Quick Setup & Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn joblib

# 2. Train model (generates model.pkl)
cd house-price
python train.py

# 3. Evaluate model performance
python evaluate.py

# 4. Make predictions
python predict.py
```

---

## 📊 Expected Results

```
Training model...
2024-XX-XX - INFO - MAE: 12,345.67
2024-XX-XX - INFO - R2 Score: 0.9876
2024-XX-XX - INFO - Model saved!
```

**Evaluation:**
```
MAE: $XX,XXX.XX
R² Score: 0.99XX
```

---

## 🔮 Usage Examples

```python
# Single prediction
price = predict(3200, 4, 2, 9)
print(f"Predicted: ${price:,.2f}")

# Production integration
features = [area, bedrooms, age, location_score]
prediction = model_pipeline.predict([features])
```

---

## 🎯 Key Features Implemented

✅ **Real ML Pipeline** - Train/test split, cross-validation  
✅ **Feature Engineering** - Domain-specific transformations  
✅ **Model Optimization** - Grid search over n_estimators, max_depth  
✅ **Production Logging** - Timestamped logs  
✅ **Config Management** - Centralized hyperparameters  
✅ **Model Persistence** - Joblib serialization  
✅ **CLI Interface** - Simple python scripts  

---

## 🚀 Next Level Upgrades (Optional)

- [ ] XGBoost/LightGBM integration
- [ ] Streamlit/Gradio dashboard
- [ ] Docker containerization
- [ ] MLflow experiment tracking
- [ ] Real dataset (Kaggle Ames Housing)
- [ ] CI/CD pipeline
- [ ] Model monitoring

---


