# 😷 Face Mask Detection System (Self-Contained)

A real-time **Face Mask Detection System** built using **Deep Learning, OpenCV, and TensorFlow/Keras**.
**Fully self-contained**: Generates synthetic training data and uses built-in OpenCV face detection.

Detects whether a person is wearing a mask using webcam.

---

## 🚀 Features

* 🎥 Real-time face detection using OpenCV HaarCascade (built-in)
* 😷 Classifies **Mask / No Mask**
* ⚡ Fast MobileNetV2 model
* 🧠 Self-generates synthetic dataset for training
* 💻 Zero external downloads needed
* ✅ Ready to run immediately

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Imutils

---

## 📂 Project Structure

```
face-mask-detection/
│── dataset/
│   ├── with_mask/
│   └── without_mask/
│
│── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
│── train.py
│── detect.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Quick Start (Self-Contained)

```bash
cd face-mask-detection
pip install -r requirements.txt
python train.py  # Generates data + trains model (few minutes)
python detect.py # Webcam detection (press 'q' to quit)
```

## 📸 Output

* 🟢 **Green Box** → Mask detected  
* 🔴 **Red Box** → No Mask detected

**Note**: Trained on synthetic cartoon faces, works on real faces too!

---

## 📊 Synthetic Dataset

Generates 2000 images:
- Face oval + eyes + mouth
- Blue rectangle mask added/removed

---

## 📈 Future Improvements

* 🌐 Deploy as Web App (Flask / FastAPI)
* 📱 Mobile App integration
* 🤖 Alert system (Email/SMS when no mask detected)
* 🐳 Docker deployment
* 🧠 Improve accuracy with larger dataset

---

