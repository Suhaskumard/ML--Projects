# 📊 Customer Churn Prediction System (Advanced)

## 📌 Overview
This project predicts whether a customer will churn using machine learning on the IBM Telco dataset from Kaggle.

---

## 🚀 Features
- Data preprocessing & feature engineering
- Multiple model training (Random Forest + XGBoost)
- Automatic best model selection
- Model evaluation (Confusion Matrix, Precision, Recall, F1-score)
- Streamlit web app for real-time predictions

---

## 📂 Dataset
- Source: Kaggle
- Dataset: Telco Customer Churn
- File: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## 🧠 Models Used
- Random Forest Classifier
- XGBoost Classifier (best performing)

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/customer-churn-project.git
cd customer-churn-project
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train Model

```bash
python train.py
```

### 2. Evaluate Model

```bash
python evaluate.py
```

### 3. Run Web App

```bash
streamlit run app.py
```

---

## 🌐 Web App Features

* Input customer details
* Instant churn prediction
* Simple UI using Streamlit

---

## 🔥 Future Improvements

* Add SHAP for explainability
* Deploy on AWS / Render
* Build REST API (Flask/FastAPI)
* Add dashboard (Power BI / Plotly)

---

## 📜 License

MIT License

