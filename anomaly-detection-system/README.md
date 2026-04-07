# 🚨 Advanced Anomaly Detection System (Multi-Model)

## 📌 Overview

This project implements an **Advanced Anomaly Detection System** using multiple machine learning algorithms to detect unusual patterns in data.

It supports:

* Isolation Forest
* Local Outlier Factor (LOF)
* One-Class SVM

The system performs preprocessing, anomaly detection, model comparison, and visualization. 

--- 

## 🎯 Features

* 🔍 Multiple anomaly detection algorithms
* ⚙️ Configurable model selection
* 📊 Model comparison (agreement analysis)
* 📉 PCA-based visualization
* ⚡ Real-time anomaly prediction
* 🧹 Automatic preprocessing (missing values + scaling)
* 💾 Export results to CSV

---

## 🧠 Algorithms Used

### 1. Isolation Forest

* Detects anomalies by isolating data points
* Efficient for large datasets

### 2. Local Outlier Factor (LOF)

* Detects anomalies based on local density deviation

### 3. One-Class SVM

* Learns boundary of normal data
* Flags outliers outside the boundary

---

## 📂 Project Structure

```
├── advanced_anomaly_detection.py
├── data.csv
├── output.csv
├── comparison.csv
└── README.md
```

---

## ⚙️ Installation

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## ▶️ Usage

1. Add your dataset as `data.csv`
2. Configure parameters in the script:

```python
MODEL_TYPE = "isolation_forest"  # options: isolation_forest, lof, svm
CONTAMINATION = 0.05
```

3. Run the program:

```bash
python advanced_anomaly_detection.py
```

---

## 📊 Output Files

### ✅ `output.csv`

* Contains original data + anomaly labels
* `1` → Normal
* `-1` → Anomaly

### 📊 `comparison.csv`

* Compares predictions from:

  * Isolation Forest
  * LOF
* Includes agreement column (True/False)

---

## 📉 Visualization

* PCA (Principal Component Analysis) reduces data to 2D
* Scatter plot shows anomalies vs normal points

---

## ⚡ Real-Time Prediction

You can predict anomalies for new data:

```python
predict_new([10, 20, 30])
```

Returns:

* `"Anomaly"` or `"Normal"`

⚠️ Note: LOF does not support real-time prediction directly.

---

## 📌 Use Cases

* 💳 Fraud Detection
* 🌐 Network Intrusion Detection
* ⚙️ Industrial Fault Detection
* 🏥 Health Monitoring
* 📈 Financial Anomaly Detection

---

## 🚀 Future Improvements

* Add web interface (Flask / Streamlit)
* Integrate real-time data streaming
* Add deep learning models (Autoencoders)
* Hyperparameter tuning for better accuracy

---




