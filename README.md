
# 🤖 Smart Face, Eyes & Smile Detector

A **real-time computer vision project** built with [OpenCV](https://opencv.org/) that can detect:
👤 **Faces** | 👀 **Eyes** | 😀 **Smiles**

Capture live video from your webcam and get instant detection results with bounding boxes and labels.

---

## ✨ Features

* 🎥 **Real-time webcam detection**
* 👤 Detects **faces**
* 👀 Detects **eyes** inside faces
* 😀 Detects **smiles** inside faces
* 🛑 Exit anytime with `q`

---

## 📦 Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/smart-face-detector.git
   cd smart-face-detector
   ```
2. Install dependencies:

   ```bash
   pip install opencv-python
   ```

---

## ▶️ Run the Project

```bash
python smart_detector.py
```

* A window will open with your webcam feed.
* You’ll see bounding boxes and labels like **"face detected"**, **"eyes detected"**, and **"smiling"**.
* Press **q** to quit.

---
## 📂 Project Structure

```
📦 smart-face-detector
 ┣ 📜 smart_detector.py   # Main Python script
 ┣ 📜 README.md           # Documentation
 ┗ 📂 face.png               # (Optional) Store screenshots
```
