
# 🎯 Face ID System (GUI-based Face Registration & Recognition)

A **real-time face recognition system** with a modern GUI built using **OpenCV + Tkinter**.
This application allows users to **register faces with guided poses** and perform **live recognition via webcam**.

---

## ✨ Features

### 🔹 Unified Application

* Single GUI for both:

  * Face Registration
  * Face Recognition

### 🔹 Live Camera Feed

* Real-time webcam display inside the app
* Face detection with bounding boxes

### 🔹 Face Registration (Enrollment Mode)

* Guided multi-step pose capture:

  * Straight
  * Left 45°
  * Right 45°
  * Tilt Up
  * Tilt Down
* Automatic dataset building
* Countdown before each step for better alignment

### 🔹 Face Recognition (Recognition Mode)

* Real-time face identification
* Displays:

  * User ID
  * Confidence score
* Unknown faces are labeled automatically

### 🔹 Smart UI

* Dark industrial theme
* Status panel + database stats
* Adjustable confidence threshold slider

---

## 🧠 How It Works

### 1. Face Detection

* Uses Haar Cascade:

  ```python
  haarcascade_frontalface_default.xml
  ```

### 2. Preprocessing

* Resize face → 100x100
* Histogram equalization (improves lighting robustness)

### 3. Model

* Uses **EigenFaceRecognizer**
* Trained on collected grayscale face samples

### 4. Recognition Logic

* Predict label + confidence
* Compare with threshold:

  * `confidence < threshold → MATCH`
  * otherwise → UNKNOWN

---

## 📦 Requirements

Install dependencies:

```bash
pip install opencv-contrib-python numpy Pillow
```

---

## ▶️ How to Run

```bash
python main.py
```

---

## 🧾 Usage Guide

### 🟢 Register a New Face

1. Click **"REGISTER FACE"**
2. Enter a **9-digit ID**
3. Follow on-screen instructions:

   * Look straight, turn head, tilt, etc.
4. System will:

   * Capture multiple samples
   * Save to database

---

### 🔵 Start Recognition

1. Click **"START RECOGNITION"**
2. Faces will be:

   * Detected
   * Labeled in real-time

---

### 🔴 Stop Process

* Click **STOP** button to return to idle mode

---

## 🗂️ File Structure

```
project/
│
├── main.py        # Main application
├── faces_db.npz      # Stored face data (auto-generated)
├── registry.json     # ID ↔ label mapping
```

---

## ⚙️ Configuration

You can modify key parameters in code:

```python
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 5000
```

### Threshold Tips:

* Lower = stricter (more accurate, more "unknown")
* Higher = more lenient (may increase false positives)

---

## 📊 Database System

* `faces_db.npz`

  * Stores:

    * face images
    * labels

* `registry.json`

  * Maps:

    ```
    user_id → label
    ```

---

## ⚠️ Limitations

* Requires **at least 2 registered users** for recognition
* Sensitive to:

  * lighting conditions
  * camera quality
* Haar Cascade is not as robust as deep learning models
* Eigenfaces may struggle with:

  * large pose variation
  * occlusion (mask, glasses, etc.)

---

## 🚀 Future Improvements

* Replace Eigenfaces with:

  * Deep learning (FaceNet / Dlib / ArcFace)
* Add:

  * Face mask handling
  * Liveness detection
  * Database UI management
* Improve:

  * Multi-face tracking
  * Performance optimization (GPU)

---

## 👨‍💻 Author Notes

This project is designed to be:

* Simple to understand
* Easy to extend
* Suitable for:

  * learning computer vision
  * prototyping AI applications

---

## 📄 License

Free to use for educational and personal projects.

