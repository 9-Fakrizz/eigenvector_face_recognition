"""
register.py  ─  Face Enrollment with Guided Capture
────────────────────────────────────────────────────
Steps:
  1. Enter a 9-digit ID
  2. Follow on-screen pose instructions (straight, left, right, up, down)
  3. Faces are saved to  faces_db.npz  and  registry.json

Requirements:
    pip install opencv-contrib-python numpy
"""

import cv2
import json
import numpy as np
import os
import re
import time

# ── Config ─────────────────────────────────────────────────────────────────────
FACE_SIZE    = (100, 100)
DB_FILE      = "faces_db.npz"      # stores face image arrays + label array
REGISTRY     = "registry.json"     # maps label index → ID string
HAAR         = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Guided capture steps  (name, instruction shown on screen, samples to collect)
GUIDED_STEPS = [
    ("Straight",    "Look  S T R A I G H T  at the camera",         8),
    ("Left  45°",   "Turn head ~45° to the  L E F T",               6),
    ("Right 45°",   "Turn head ~45° to the  R I G H T",             6),
    ("Tilt Up",     "Tilt your head slightly  U P",                  5),
    ("Tilt Down",   "Tilt your head slightly  D O W N",              5),
]
# Total samples per person = sum of counts above (default: 30)


# ── Helpers ────────────────────────────────────────────────────────────────────

def preprocess(gray_crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray_crop, FACE_SIZE)
    return cv2.equalizeHist(resized)


def load_db() -> tuple[list[np.ndarray], list[int], dict]:
    """Load existing face DB from disk (or return empty structures)."""
    images: list[np.ndarray] = []
    labels: list[int]        = []
    registry: dict           = {}          # {str_id: label_int}

    if os.path.exists(REGISTRY):
        with open(REGISTRY, "r") as f:
            registry = json.load(f)        # e.g. {"123456789": 0, "987654321": 1}

    if os.path.exists(DB_FILE):
        data   = np.load(DB_FILE)
        images = list(data["images"])      # list of (100,100) arrays
        labels = list(data["labels"].astype(int))

    return images, labels, registry


def save_db(images: list[np.ndarray], labels: list[int], registry: dict) -> None:
    """Persist face DB to disk."""
    np.savez_compressed(
        DB_FILE,
        images=np.array(images, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int32),
    )
    with open(REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\n  [SAVED]  {DB_FILE}  +  {REGISTRY}")


def validate_id(id_str: str) -> bool:
    return bool(re.fullmatch(r"\d{9}", id_str))


# ── Drawing helpers ────────────────────────────────────────────────────────────

def overlay_hud(frame, step_name, instruction, collected, target, total_steps, step_idx):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    # Step badge
    badge_text = f"STEP {step_idx + 1}/{total_steps}  •  {step_name}"
    cv2.putText(frame, badge_text, (12, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 200, 255), 1, cv2.LINE_AA)

    # Instruction
    cv2.putText(frame, instruction, (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    # Progress bar (bottom)
    bar_y = h - 18
    bar_w = int(w * collected / max(target, 1))
    cv2.rectangle(frame, (0, bar_y), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, bar_y), (bar_w, h), (0, 210, 90), -1)
    cv2.putText(frame, f"{collected}/{target}", (w - 70, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def countdown_overlay(cap, detector, message: str, seconds: int = 3):
    """Show a countdown before starting the next capture step."""
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        remaining = seconds - int(time.time() - start)
        if remaining <= 0:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 255), 2)

        # Darken + show text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, message,
                    (frame.shape[1] // 2 - 200, frame.shape[0] // 2 - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(remaining),
                    (frame.shape[1] // 2 - 20, frame.shape[0] // 2 + 60),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow("Registration", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            return False
    return True


# ── Main registration flow ─────────────────────────────────────────────────────

def register():
    print("\n" + "═" * 48)
    print("   FACE  REGISTRATION  SYSTEM")
    print("═" * 48)

    # ── 1. Get valid ID ──────────────────────────────
    while True:
        uid = input("\n  Enter your 9-digit ID : ").strip()
        if validate_id(uid):
            break
        print("  [ERROR] ID must be exactly 9 digits (e.g. 123456789).")

    # ── 2. Load existing DB ──────────────────────────
    images, labels, registry = load_db()

    if uid in registry:
        print(f"\n  [WARN] ID '{uid}' is already registered.")
        choice = input("  Overwrite? (y/N): ").strip().lower()
        if choice != 'y':
            print("  Registration cancelled.")
            return
        # Remove old samples for this ID
        old_label = registry[uid]
        keep = [(img, lbl) for img, lbl in zip(images, labels) if lbl != old_label]
        images = [k[0] for k in keep]
        labels = [k[1] for k in keep]
        new_label = old_label      # reuse the same label slot
        print(f"  [INFO] Old samples removed. Re-capturing for ID {uid}.")
    else:
        # Assign a new label = next available integer
        used = set(registry.values())
        new_label = 0
        while new_label in used:
            new_label += 1
        registry[uid] = new_label

    print(f"\n  ID       : {uid}")
    print(f"  Label    : {new_label}")
    total_needed = sum(s[2] for s in GUIDED_STEPS)
    print(f"  Samples  : {total_needed}  (across {len(GUIDED_STEPS)} poses)\n")

    # ── 3. Open webcam ───────────────────────────────
    cap      = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(HAAR)
    if not cap.isOpened():
        print("  [ERROR] Cannot open webcam.")
        return

    all_ok = True

    # ── 4. Guided capture loop ───────────────────────
    for step_idx, (step_name, instruction, target) in enumerate(GUIDED_STEPS):

        # Countdown before each step
        msg = f"Get ready :  {step_name}"
        if not countdown_overlay(cap, detector, msg, seconds=3):
            all_ok = False
            break

        collected = 0
        print(f"  [{step_idx+1}/{len(GUIDED_STEPS)}] {step_name}  — capturing {target} samples …")

        while collected < target:
            ret, frame = cap.read()
            if not ret:
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                if collected >= target:
                    break
                crop = preprocess(gray[y:y + h, x:x + w])
                images.append(crop)
                labels.append(new_label)
                collected += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 230, 80), 2)

            overlay_hud(frame, step_name, instruction,
                        collected, target, len(GUIDED_STEPS), step_idx)
            cv2.imshow("Registration", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n  [ABORT] Registration cancelled by user.")
                all_ok = False
                break

        if not all_ok:
            break

        print(f"     {collected} samples captured.")

    cap.release()
    cv2.destroyAllWindows()

    # ── 5. Save or rollback ──────────────────────────
    if all_ok:
        save_db(images, labels, registry)
        print(f"\n    Registration complete for ID: {uid}")
        print(f"      Total samples in DB : {len(images)}")
        print(f"      Total people in DB  : {len(registry)}")
    else:
        print("\n      Registration aborted — no data was saved.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    register()