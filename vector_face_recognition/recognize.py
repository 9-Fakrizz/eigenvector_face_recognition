"""
recognize.py  ─  Live Face Recognition (Eigenfaces / PCA)
──────────────────────────────────────────────────────────
Loads the face DB built by register.py, trains an Eigenface model,
then runs live webcam recognition and overlays the matched 9-digit ID.

Requirements:
    pip install opencv-contrib-python numpy
    (run register.py first to build the DB)
"""

import cv2
import json
import numpy as np
import os
import sys

# ── Config ─────────────────────────────────────────────────────────────────────
FACE_SIZE   = (100, 100)
DB_FILE     = "faces_db.npz"
REGISTRY    = "registry.json"
HAAR        = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Confidence threshold for Eigenfaces:
#   lower  → stricter  (fewer false positives, may miss real matches)
#   higher → looser    (more matches, may misidentify)
CONFIDENCE_THRESHOLD = 5000


# ── Load & train ───────────────────────────────────────────────────────────────

def load_and_train():
    """Load the face DB, build label→ID map, train Eigenface recognizer."""

    # ── Check files exist ────────────────────────────────────────────────────
    for path in (DB_FILE, REGISTRY):
        if not os.path.exists(path):
            print(f"  [ERROR] '{path}' not found.")
            print("  Run  register.py  first to enroll faces.")
            sys.exit(1)

    # ── Load numpy arrays ────────────────────────────────────────────────────
    data   = np.load(DB_FILE)
    images = list(data["images"])                  # list of (100,100) uint8
    labels = data["labels"].astype(int).tolist()   # list of int

    with open(REGISTRY, "r") as f:
        registry: dict = json.load(f)              # {"123456789": 0, ...}

    if len(set(labels)) < 2:
        print("  [ERROR] Need at least 2 registered people to run Eigenfaces.")
        print("  Run  register.py  to enroll more faces.")
        sys.exit(1)

    # ── Reverse registry: label_int → id_string ──────────────────────────────
    label_to_id: dict[int, str] = {v: k for k, v in registry.items()}

    # ── Train Eigenface recognizer ───────────────────────────────────────────
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(images, np.array(labels, dtype=np.int32))

    print(f"  [OK] Model trained")
    print(f"       People  : {len(registry)}")
    print(f"       Samples : {len(images)}")

    return recognizer, label_to_id


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_result(frame, x, y, w, h, display_id: str, confidence: float, matched: bool):
    color  = (0, 220, 80) if matched else (0, 60, 220)
    label  = display_id if matched else "Unknown"
    conf_s = f"conf: {confidence:.0f}"

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background pill
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 10), 1, cv2.LINE_AA)

    # Confidence small text below box
    cv2.putText(frame, conf_s, (x + 4, y + h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, people_count: int, samples_count: int):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 32), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)
    info = (f"DB: {people_count} people  |  {samples_count} samples  |  "
            f"threshold: {CONFIDENCE_THRESHOLD}  |  Q to quit")
    cv2.putText(frame, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ── Live recognition loop ──────────────────────────────────────────────────────

def recognize():
    print("\n" + "═" * 48)
    print("   LIVE  FACE  RECOGNITION")
    print("═" * 48)

    recognizer, label_to_id = load_and_train()
    people_count  = len(label_to_id)
    samples_count = sum(
        1 for _ in np.load(DB_FILE)["labels"]
    )

    detector = cv2.CascadeClassifier(HAAR)
    cap      = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open webcam.")
        sys.exit(1)

    print("\n  Press  Q  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Preprocess crop exactly as during registration
            crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            crop = cv2.equalizeHist(crop)

            label, confidence = recognizer.predict(crop)

            matched    = confidence < CONFIDENCE_THRESHOLD
            display_id = label_to_id.get(label, "???") if matched else "Unknown"

            draw_result(frame, x, y, w, h, display_id, confidence, matched)

        draw_hud(frame, people_count, samples_count)
        cv2.imshow("Face Recognition  —  press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  [EXIT] Recognition stopped.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    recognize()