# 📧 Spam Email Detection System

A Machine Learning-based application that classifies messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques.

---

## 🚀 Overview

This project builds an end-to-end spam detection pipeline using:

* Text preprocessing
* Feature extraction with **TF-IDF**
* Classification using **Naive Bayes**

The model is trained on a real-world dataset and can predict spam messages through a command-line interface.

---

## ✨ Features

* 📩 Classifies messages into Spam / Not Spam
* 🧹 Text preprocessing (cleaning and normalization)
* 🤖 Machine Learning model (Multinomial Naive Bayes)
* 📊 Model evaluation with accuracy and metrics
* 💻 Interactive CLI-based prediction system
* 📁 Modular and scalable code structure

---

## 📁 Project Structure

```id="h3tr9h"
spam_email_detection_system/
│── app.py                 # CLI interface
│── train_model.py         # Train ML model
│── predictor.py           # Prediction logic
│── preprocess.py          # Text preprocessing
│── evaluate.py            # Model evaluation
│── data/                  # Dataset folder
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│── .gitignore             # Ignore unnecessary files
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Naive Bayes Algorithm

---

## 🔧 Installation

```bash id="p5h48v"
git clone https://github.com/Suhaskumard/Machine-Learning-Projects.git
cd Machine-Learning-Projects/spam_email_detection_system
```

---

## ▶️ Usage

### Train the model

```bash id="17jql6"
python train_model.py
```

### Run the application

```bash id="m2s6hj"
python app.py
```

### Evaluate the model

```bash id="98idw3"
python evaluate.py
```

---

## 📊 Model Performance

* Achieves high accuracy (~95%) on real dataset
* Evaluated using classification metrics and confusion matrix

---

## 📂 Dataset

* UCI SMS Spam Collection Dataset
* Contains labeled spam and non-spam messages

---

## 📈 Future Improvements

* Build web interface (Flask / FastAPI)
* Add visualization dashboard (Streamlit)
* Try advanced ML/DL models (SVM, BERT)
* Deploy as an online service

---

## 🧠 Working Principle

1. Input text is cleaned using preprocessing
2. Converted into numerical features using TF-IDF
3. Passed to trained Naive Bayes model
4. Output: Spam or Not Spam

---
