# 🧠 Image Classification using CNN

## 📌 Overview
This project implements a **Basic Convolutional Neural Network (CNN)** to classify images into categories.

---

## 🚀 Features
- Image preprocessing using ImageDataGenerator
- CNN model with Conv2D + MaxPooling
- Training + validation split
- Model saving & loading
- Image prediction script

---

## 📂 Project Structure

```
image-classification-cnn/
│── data/
│   ├── train/
│   ├── test/
│
│── models/
│   └── cnn_model.h5
│
│── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
│── requirements.txt
│── README.md
│── TODO.md
```

---

## ⚙️ Installation

```bash
cd image-classification-cnn
pip install -r requirements.txt
```

---

## ▶️ Run Training

```bash
cd src
python train.py
```

---

## 🔍 Run Prediction

```bash
cd src
python predict.py
```

---

## 🧠 Model Architecture

* Conv2D (32 filters)
* MaxPooling
* Conv2D (64 filters)
* MaxPooling
* Conv2D (128 filters)
* Fully Connected Layer
* Softmax Output

---

## 📊 Output

* Trained model saved as `../models/cnn_model.h5`
* Check TODO.md for next steps

---

## 📌 Future Improvements

* Add data augmentation
* Use transfer learning (ResNet, VGG)
* Deploy as web app

---
