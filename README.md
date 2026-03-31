🚀 Advanced Face Mask Detection System

An intelligent real-time surveillance system that detects whether people are wearing masks using Computer Vision and Deep Learning. The system includes face detection, mask classification, violation tracking, alert system, and automated reporting.

📌 Features
👤 Real-time Face Detection
😷 Mask / No Mask Classification (MobileNetV2)
🔢 Face Counting (Live)
🧠 Face Tracking (Unique IDs)
🚨 Violation Detection System
🔊 Sound Alert for No Mask
📸 Automatic Screenshot Capture (Face Only)
🎥 Video Recording of Violations
📊 Live Dashboard (FPS, Counts, Stats)
📁 CSV Report Generation
⚠️ Crowd Detection Warning
🛠️ Tech Stack
Python
OpenCV
TensorFlow / Keras
NumPy
Scikit-learn
📂 Project Structure
Face-Mask-Detection/
│
├── detect_mask_video.py      # Main file (run this)
├── train_mask_detector.py   # Training script
├── mask_detector.keras      # Trained model
│
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── violations/              # Auto-saved violation images
├── violation.avi            # Recorded violation video
├── report.csv               # Session report
└── README.md
⚙️ Setup Instructions
1️⃣ Clone the repository
git clone <your-repo-link>
cd Face-Mask-Detection
2️⃣ Install dependencies
pip install opencv-python tensorflow numpy imutils matplotlib scikit-learn
3️⃣ Download / Prepare Dataset

Ensure your dataset folder structure is:

dataset/
 ├── with_mask/
 └── without_mask/
4️⃣ Train the model (optional)
python train_mask_detector.py

This will generate:

mask_detector.keras
5️⃣ Run the project 🚀
python detect_mask_video.py
🖥️ How It Works
Webcam captures real-time video
Faces are detected using OpenCV DNN
Each face is classified as:
Mask 😷
No Mask ❌
If "No Mask" is detected:
🔊 Alert sound plays
📸 Image is saved
🎥 Video recording starts
📊 Violation count increases
Dashboard shows:
FPS
Face count
Mask / No Mask count
Violations
📊 Output
📁 violations/ → saved face images
🎥 violation.avi → recorded video
📄 report.csv → session summary
💡 Future Improvements
Face Recognition (identify person)
Web Dashboard (Flask / React)
Email/SMS Alerts
YOLO-based detection for better accuracy


👨‍💻 Author

Jai Arora

⭐ If you like this project

Give it a ⭐ on GitHub and share!
