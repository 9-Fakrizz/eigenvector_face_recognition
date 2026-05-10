"""
main.py — ESP32-Driven Face Recognition
════════════════════════════════════════════════════════════════
Listens on Serial for "TRIGGER" sent by the ESP32 when its
D35 sensor fires.  Runs a 30-second recognition session with
multi-frame glitch protection, then sends the result back.

Protocol (Serial, 115200 baud):
  ESP32 → Python : "TRIGGER"
  Python → ESP32 : "ID:<9-digit-uid>"   ← face confirmed
                   "TIMEOUT"            ← 30 s elapsed, no match
                   "NO_MATCH"           ← face detected but unknown
                   "NO_MODEL"           ← DB missing / only 1 person

Requirements:
    pip install opencv-contrib-python numpy pyserial
"""

import cv2
import json
import numpy as np
import os
import serial
import serial.tools.list_ports
import time

# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════
FACE_SIZE               = (100, 100)
DB_FILE                 = "faces_db.npz"
REGISTRY                = "registry.json"
HAAR                    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

CONFIDENCE_THRESHOLD    = 5000    # lower = stricter (EigenFace distance)
RECOGNITION_TIMEOUT     = 30      # seconds per session
CAMERA_INDEX            = 0

# ── Glitch protection ──────────────────────────────────────────
# Face must match the SAME person for this many consecutive frames
# before we accept it as a real match.
MATCH_FRAMES_REQUIRED   = 10

# If match breaks (face disappears / different person), the running
# counter decays by this amount per blank frame instead of resetting
# to zero — tolerates brief occlusions without requiring a full restart.
MATCH_DECAY_PER_FRAME   = 2

# Serial
SERIAL_PORT             = None    # None = auto-detect
SERIAL_BAUD             = 115200
SHOW_PREVIEW            = True    # set False for headless

# ══════════════════════════════════════════════════════════════
#  DB / model helpers  (shared with the GUI registration app)
# ══════════════════════════════════════════════════════════════

def preprocess(gray_crop: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(cv2.resize(gray_crop, FACE_SIZE))


def load_recognizer():
    """Return (EigenFaceRecognizer, label_to_uid_dict) or (None, {})."""
    if not os.path.exists(DB_FILE) or not os.path.exists(REGISTRY):
        return None, {}
    data   = np.load(DB_FILE)
    images = list(data["images"])
    labels = data["labels"].astype(int).tolist()
    with open(REGISTRY) as f:
        registry = json.load(f)
    if len(set(labels)) < 2:
        return None, {}
    label_to_uid = {v: k for k, v in registry.items()}
    rec = cv2.face.EigenFaceRecognizer_create()
    rec.train(images, np.array(labels, dtype=np.int32))
    return rec, label_to_uid


# ══════════════════════════════════════════════════════════════
#  Recognition session
# ══════════════════════════════════════════════════════════════

def run_recognition_session(recognizer, label_to_uid: dict) -> str | None:
    """
    Open camera, attempt to recognise a face within RECOGNITION_TIMEOUT seconds.

    Returns
    -------
    str   — 9-digit UID if a face was confirmed
    None  — timeout reached without a confirmed match

    Glitch protection
    -----------------
    A single detected match is NOT accepted.  The same UID must appear in
    MATCH_FRAMES_REQUIRED consecutive (or near-consecutive) frames.
    Brief mis-detections decrement the counter rather than resetting it.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[Camera] ERROR: cannot open camera")
        return None

    detector     = cv2.CascadeClassifier(HAAR)
    deadline     = time.time() + RECOGNITION_TIMEOUT

    # Glitch-protection state
    candidate_uid   = None   # current leading candidate
    consecutive     = 0      # how many frames this candidate has scored

    print(f"[Session] Started — timeout in {RECOGNITION_TIMEOUT}s, "
          f"need {MATCH_FRAMES_REQUIRED} consecutive matches")

    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                print("[Session] Timeout reached")
                return None

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

            # ── Find best match in this frame ─────────────────
            frame_uid  = None
            frame_conf = float("inf")

            for (x, y, w, h) in faces:
                crop = preprocess(gray[y:y+h, x:x+w])
                lbl, conf = recognizer.predict(crop)
                if conf < CONFIDENCE_THRESHOLD and conf < frame_conf:
                    frame_conf = conf
                    frame_uid  = label_to_uid.get(lbl)

            # ── Update glitch-protection counter ──────────────
            if frame_uid is not None and frame_uid == candidate_uid:
                # Same person again → increment
                consecutive += 1
            elif frame_uid is not None and frame_uid != candidate_uid:
                # Different person → new candidate, but keep partial credit
                # proportional to how sure we are of the new person
                candidate_uid = frame_uid
                consecutive   = 1
            else:
                # No match this frame → decay, don't hard-reset
                consecutive = max(0, consecutive - MATCH_DECAY_PER_FRAME)
                if consecutive == 0:
                    candidate_uid = None

            confirmed = consecutive >= MATCH_FRAMES_REQUIRED

            print(f"[Session] t={RECOGNITION_TIMEOUT - remaining:.1f}s | "
                  f"candidate={candidate_uid or 'none':>10} | "
                  f"streak={consecutive}/{MATCH_FRAMES_REQUIRED}"
                  + (" ✓ CONFIRMED" if confirmed else ""))

            if confirmed:
                print(f"[Session] Match confirmed: {candidate_uid}")
                return candidate_uid

            # ── Optional preview window ───────────────────────
            if SHOW_PREVIEW:
                for (x, y, w, h) in faces:
                    color = (0, 230, 80) if frame_uid else (80, 80, 220)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                bar_w    = int(frame.shape[1] * consecutive / MATCH_FRAMES_REQUIRED)
                cv2.rectangle(frame, (0, frame.shape[0]-14),
                              (bar_w, frame.shape[0]), (0, 200, 80), -1)

                status = (f"Scanning  {remaining:.1f}s  |  "
                          f"streak {consecutive}/{MATCH_FRAMES_REQUIRED}"
                          + (f"  ID:{candidate_uid}" if candidate_uid else ""))
                cv2.putText(frame, status, (10, 28),
                            cv2.FONT_HERSHEY_DUPLEX, 0.58,
                            (0, 220, 255), 1, cv2.LINE_AA)

                cv2.imshow("Face Recognition — press ESC to abort", frame)
                if cv2.waitKey(1) == 27:
                    print("[Session] Aborted by user")
                    return None
    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
#  Serial helpers
# ══════════════════════════════════════════════════════════════

def auto_detect_port() -> str | None:
    """Try to find the ESP32's COM/ttyUSB port automatically."""
    keywords = ("CP210", "CH340", "CH9102", "UART", "USB Serial", "ESP32")
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "") + (p.manufacturer or "")
        if any(k.lower() in desc.lower() for k in keywords):
            print(f"[Serial] Auto-detected: {p.device}  ({p.description})")
            return p.device
    ports = serial.tools.list_ports.comports()
    if ports:
        print(f"[Serial] No ESP32 keyword match; using first port: {ports[0].device}")
        return ports[0].device
    return None


def open_serial(port: str | None, baud: int) -> serial.Serial:
    port = port or auto_detect_port()
    if port is None:
        raise RuntimeError("No serial port found. Set SERIAL_PORT manually.")
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(2.0)   # wait for ESP32 boot / DTR reset
    print(f"[Serial] Connected: {port} @ {baud}")
    return ser


# ══════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ESP32 Face ID — Python side")
    print("=" * 60)

    ser = open_serial(SERIAL_PORT, SERIAL_BAUD)

    # Pre-load recognizer once; reload before each session in case
    # someone registers a new face between triggers.
    print("[System] Loading face model...")
    recognizer, label_to_uid = load_recognizer()
    if recognizer is None:
        print("[System] WARNING: no face database found — "
              "run the registration GUI first.")

    print("[System] Entering main loop — waiting for TRIGGER from ESP32\n")

    while True:
        # ── Drain incoming serial line ────────────────────────
        if not ser.in_waiting:
            time.sleep(0.05)
            continue

        raw  = ser.readline()
        line = raw.decode(errors="replace").strip()
        if not line:
            continue

        print(f"[Serial RX] {line!r}")

        if line != "TRIGGER":
            # Status messages from ESP32 (READY, ACCESS GRANTED, etc.)
            continue

        # ── TRIGGER received ──────────────────────────────────
        print("\n[System] ─── TRIGGER ─────────────────────────────────")

        # Reload model so freshly registered faces are included
        recognizer, label_to_uid = load_recognizer()

        if recognizer is None:
            print("[System] No model — sending NO_MODEL")
            ser.write(b"NO_MODEL\n")
            continue

        # Run the recognition session
        uid = run_recognition_session(recognizer, label_to_uid)

        if uid:
            msg = f"ID:{uid}\n"
            ser.write(msg.encode())
            print(f"[Serial TX] {msg.strip()}")
        else:
            # Distinguish timeout from "face seen but unknown"
            # (here we treat both as TIMEOUT; refine if needed)
            ser.write(b"TIMEOUT\n")
            print("[Serial TX] TIMEOUT")

        print("[System] ─────────────────────────────────────────────\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user — shutting down")
    except Exception as e:
        print(f"[System] Fatal error: {e}")
        raise