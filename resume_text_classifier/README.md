# 📄 Resume / Text Classification System

## 🚀 Overview 

This project is a Machine Learning-based Resume Classification System that automatically categorizes resumes into predefined domains such as:

* Data Science
* Web Development
* Finance
* Software Engineering

It leverages Natural Language Processing (NLP) and supervised learning techniques to analyze resume text and predict the most relevant job category.
  
---

## 🧠 Tech Stack

* Python
* Scikit-learn
* Pandas
* NLP (TF-IDF Vectorization)
* Logistic Regression

---

## ⚙️ Features

* Automated resume classification
* Text preprocessing using TF-IDF
* Scalable ML pipeline
* Easy-to-use CLI interface
* Extendable to web applications

---

## 📂 Project Structure

```bash
resume_text_classifier/
│── data/
│── model/
│── train.py
│── predict.py
│── requirements.txt
│── README.md
```

---

## 🔍 How It Works

1. Input resume text is collected
2. TF-IDF converts text into numerical vectors
3. Logistic Regression model classifies the text
4. Output → predicted category

---

## ▶️ Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/your-username/resume_text_classifier.git
cd resume_text_classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train.py
```

### 4. Run Prediction

```bash
python predict.py
```

---

## 📊 Example

### Input:

```text
Experienced in Python, Machine Learning, NLP, TensorFlow
```

### Output:

```text
Data Science
```

---

## 📈 Model Details

* Algorithm: Logistic Regression
* Vectorization: TF-IDF (max_features=5000)
* Evaluation Metric: Precision, Recall, F1-score

---

## 🔮 Future Enhancements

* Use Deep Learning (LSTM, BERT)
* Resume parsing from PDF/DOCX
* Deploy as a web app (Flask/Streamlit)
* Add real-time job recommendation system

---

## 💡 Use Cases

* HR automation systems
* Resume screening platforms
* Job recommendation engines

---

## 👨‍💻 Author

Suhas Kumar
