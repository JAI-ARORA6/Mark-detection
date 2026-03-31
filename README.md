# 🚀 Advanced Face Mask Detection System

An intelligent real-time surveillance system that detects whether people are wearing masks using Computer Vision and Deep Learning.

---

## 📌 Features

* 👤 Real-time Face Detection
* 😷 Mask / No Mask Classification (MobileNetV2)
* 🔢 Face Counting (Live)
* 🧠 Face Tracking (Unique IDs)
* 🚨 Violation Detection System
* 🔊 Sound Alert for No Mask
* 📸 Automatic Screenshot Capture (Face Only)
* 🎥 Video Recording of Violations
* 📊 Live Dashboard (FPS, Counts, Stats)
* 📁 CSV Report Generation
* ⚠️ Crowd Detection Warning

---

## 🛠️ Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Scikit-learn

---

## 📂 Project Structure

```
Face-Mask-Detection/
│
├── detect_mask_video.py
├── train_mask_detector.py
├── mask_detector.keras
│
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── violations/
├── violation.avi
├── report.csv
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd Face-Mask-Detection
```

### 2️⃣ Install dependencies

```
pip install opencv-python tensorflow numpy imutils matplotlib scikit-learn
```

### 3️⃣ Prepare Dataset

```
dataset/
 ├── with_mask/
 └── without_mask/
```

### 4️⃣ Train the model (optional)

```
python train_mask_detector.py
```

### 5️⃣ Run the project

```
python detect_mask_video.py
```

---

## 🖥️ How It Works

1. Webcam captures real-time video
2. Faces are detected using OpenCV
3. Model predicts Mask / No Mask
4. If No Mask:

   * 🔊 Alert sound
   * 📸 Image saved
   * 🎥 Video recorded
   * 📊 Count updated

---

## 📊 Output

* `violations/` → saved images
* `violation.avi` → recorded video
* `report.csv` → summary

---

## 💡 Future Improvements

* Face Recognition
* Web Dashboard
* Email Alerts
* YOLO Detection

---

## 👨‍💻 Author

Jai Arora

---

⭐ If you like this project, give it a star!
