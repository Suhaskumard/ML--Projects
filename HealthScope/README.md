# Disease Prediction System 🩺

## 📌 Overview

This project predicts diseases based on user symptoms using a machine learning model.

## ⚙️ Technologies Used

* Python  
* Scikit-learn
* Flask
* Pandas
* NumPy

## 📂 Project Structure

* `train_model.py` → trains ML model
* `predict.py` → CLI prediction
* `app.py` → Flask API
* `dataset.csv` → training dataset

## 🚀 How to Run

### Step 1: Install dependencies

```
pip install pandas scikit-learn flask numpy
```

### Step 2: Train model

```
python train_model.py
```

### Step 3: Run app

```
python app.py
```


### Step 4: Test API

POST request:

```
http://127.0.0.1:5000/predict
```

Body:

```json
{
  "symptoms": [1,0,1,0,1]
}
```

## 📊 Features

* Predicts disease from symptoms
* REST API support
* Easy to extend with more data

## 🔮 Future Improvements

* Add GUI (React / HTML)
* Use advanced ML models
* Add real medical dataset

