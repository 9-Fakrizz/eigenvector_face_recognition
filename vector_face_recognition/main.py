"""
Unified Face Registration & Recognition GUI
═══════════════════════════════════════════════════════════════
Single-window app combining register.py + recognize.py with:
  • Live webcam feed embedded in the GUI
  • REGISTER mode  – guided multi-pose enrollment
  • RECOGNIZE mode – real-time ID overlay on detected faces

Requirements:
    pip install opencv-contrib-python numpy Pillow
"""

import cv2
import json
import numpy as np
import os
import re
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
import tkinter.messagebox as mb
import tkinter.simpledialog as sd
from PIL import Image, ImageTk

# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════
FACE_SIZE   = (100, 100)
DB_FILE     = "faces_db.npz"
REGISTRY    = "registry.json"
HAAR        = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 5000

GUIDED_STEPS = [
    ("Straight",   "Look STRAIGHT at the camera",      8),
    ("Left 45°",   "Turn head ~45° to the LEFT",        6),
    ("Right 45°",  "Turn head ~45° to the RIGHT",       6),
    ("Tilt Up",    "Tilt your head slightly UP",         5),
    ("Tilt Down",  "Tilt your head slightly DOWN",       5),
]

# ══════════════════════════════════════════════════════════════
#  Palette (dark industrial)
# ══════════════════════════════════════════════════════════════
BG          = "#0d0f14"
PANEL       = "#13161e"
CARD        = "#1a1e2a"
ACCENT      = "#00e5ff"
ACCENT2     = "#ff3f6c"
GREEN       = "#00e676"
YELLOW      = "#ffd740"
TEXT_PRI    = "#e8eaf0"
TEXT_SEC    = "#6b7280"
BORDER      = "#252a38"

# ══════════════════════════════════════════════════════════════
#  DB helpers
# ══════════════════════════════════════════════════════════════

def preprocess(gray_crop):
    return cv2.equalizeHist(cv2.resize(gray_crop, FACE_SIZE))


def load_db():
    images, labels, registry = [], [], {}
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            registry = json.load(f)
    if os.path.exists(DB_FILE):
        data   = np.load(DB_FILE)
        images = list(data["images"])
        labels = list(data["labels"].astype(int))
    return images, labels, registry


def save_db(images, labels, registry):
    np.savez_compressed(DB_FILE,
                        images=np.array(images, dtype=np.uint8),
                        labels=np.array(labels, dtype=np.int32))
    with open(REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)


def train_recognizer():
    data     = np.load(DB_FILE)
    images   = list(data["images"])
    labels   = data["labels"].astype(int).tolist()
    with open(REGISTRY) as f:
        registry = json.load(f)
    if len(set(labels)) < 2:
        return None, {}
    label_to_id = {v: k for k, v in registry.items()}
    rec = cv2.face.EigenFaceRecognizer_create()
    rec.train(images, np.array(labels, dtype=np.int32))
    return rec, label_to_id


# ══════════════════════════════════════════════════════════════
#  Draw helpers (OpenCV frame)
# ══════════════════════════════════════════════════════════════

def cv_text(frame, text, pos, scale=0.6, color=(255,255,255), thick=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_face_box(frame, x, y, w, h, label, confidence, matched):
    color = (0, 230, 118) if matched else (255, 63, 108)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Corner accents
    seg = 16
    for (px, py, dx, dy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame, (px, py), (px+dx*seg, py), color, 3)
        cv2.line(frame, (px, py), (px, py+dy*seg), color, 3)

    # Label pill
    pill_text = label if matched else "UNKNOWN"
    (tw, th), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame, (x, y-th-14), (x+tw+12, y), color, -1)
    cv2.putText(frame, pill_text, (x+6, y-6),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (10,10,10), 1, cv2.LINE_AA)

    # Confidence
    cv_text(frame, f"conf {confidence:.0f}", (x+4, y+h+18),
            scale=0.42, color=color)


def draw_register_overlay(frame, step_name, instruction, collected, target, step_idx, total_steps):
    h, w = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0,0), (w,72), (13,16,30), -1)
    cv2.addWeighted(bar, 0.72, frame, 0.28, 0, frame)

    badge = f"STEP {step_idx+1}/{total_steps}  •  {step_name}"
    cv_text(frame, badge, (12,26), scale=0.62, color=(0,229,255), thick=1)
    cv_text(frame, instruction, (12,54), scale=0.58, color=(230,230,230))

    # Progress bar
    bh = h - 20
    bar_w = int(w * collected / max(target, 1))
    cv2.rectangle(frame, (0, bh), (w, h), (30,30,40), -1)
    cv2.rectangle(frame, (0, bh), (bar_w, h), (0,210,90), -1)
    cv_text(frame, f"{collected}/{target}", (w-68, h-4),
            scale=0.44, color=(190,190,190))


# ══════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════

class FaceApp:
    MODE_IDLE       = "idle"
    MODE_RECOGNIZE  = "recognize"
    MODE_REGISTER   = "register"

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Face ID System")
        root.configure(bg=BG)
        root.resizable(False, False)

        # State
        self.mode        = self.MODE_IDLE
        self.cap         = None
        self.running     = False
        self.reg_thread  = None
        self.reg_images  = []
        self.reg_labels  = []
        self.reg_registry= {}
        self.reg_new_label = 0
        self.reg_uid     = ""
        self.reg_step_idx   = 0
        self.reg_collected  = 0
        self.reg_target     = 0
        self.reg_step_name  = ""
        self.reg_instruction= ""
        self.reg_waiting_countdown = False
        self.reg_countdown_end     = 0
        self.reg_countdown_msg     = ""
        self.recognizer  = None
        self.label_to_id = {}
        self.db_info     = {"people": 0, "samples": 0}

        self.detector = cv2.CascadeClassifier(HAAR)

        self._build_ui()
        self._refresh_db_info()
        self._start_camera()

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── Left: video ──────────────────────────────────────
        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, padx=(18,8), pady=18)

        # Video canvas
        self.canvas = tk.Canvas(left, width=640, height=480,
                                bg="#000000", highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack()

        # Mode badge below video
        self.mode_var = tk.StringVar(value="● IDLE")
        mode_lbl = tk.Label(left, textvariable=self.mode_var,
                            bg=BG, fg=TEXT_SEC,
                            font=("Courier", 11, "bold"))
        mode_lbl.pack(pady=(8,0))

        # ── Right: control panel ─────────────────────────────
        right = tk.Frame(root, bg=PANEL, width=260,
                         highlightthickness=1, highlightbackground=BORDER)
        right.grid(row=0, column=1, padx=(0,18), pady=18, sticky="ns")
        right.grid_propagate(False)

        pad = {"padx": 18}

        # Logo / title
        title_f = tk.Frame(right, bg=PANEL)
        title_f.pack(fill="x", padx=18, pady=(24,0))
        tk.Label(title_f, text="FACE  ID", bg=PANEL, fg=ACCENT,
                 font=("Courier", 22, "bold")).pack()
        tk.Label(title_f, text="RECOGNITION SYSTEM", bg=PANEL, fg=TEXT_SEC,
                 font=("Courier", 8)).pack()

        self._sep(right)

        # ── DB Stats ─────────────────────────────────────────
        stats_f = tk.Frame(right, bg=CARD,
                           highlightthickness=1, highlightbackground=BORDER)
        stats_f.pack(fill="x", padx=14, pady=(0,6))

        tk.Label(stats_f, text="DATABASE", bg=CARD, fg=TEXT_SEC,
                 font=("Courier", 8)).pack(anchor="w", padx=10, pady=(8,2))

        row1 = tk.Frame(stats_f, bg=CARD)
        row1.pack(fill="x", padx=10, pady=(0,4))
        tk.Label(row1, text="People", bg=CARD, fg=TEXT_SEC,
                 font=("Courier", 10)).pack(side="left")
        self.lbl_people = tk.Label(row1, text="0", bg=CARD, fg=ACCENT,
                                   font=("Courier", 14, "bold"))
        self.lbl_people.pack(side="right")

        row2 = tk.Frame(stats_f, bg=CARD)
        row2.pack(fill="x", padx=10, pady=(0,8))
        tk.Label(row2, text="Samples", bg=CARD, fg=TEXT_SEC,
                 font=("Courier", 10)).pack(side="left")
        self.lbl_samples = tk.Label(row2, text="0", bg=CARD, fg=ACCENT,
                                    font=("Courier", 14, "bold"))
        self.lbl_samples.pack(side="right")

        self._sep(right)

        # ── Recognize button ─────────────────────────────────
        tk.Label(right, text="LIVE RECOGNITION", bg=PANEL, fg=TEXT_SEC,
                 font=("Courier", 8)).pack(anchor="w", **pad, pady=(0,4))

        self.btn_rec = tk.Button(right, text="▶  START  RECOGNITION",
                                 bg=ACCENT, fg="#000000",
                                 font=("Courier", 11, "bold"),
                                 activebackground="#00b8d4",
                                 relief="flat", cursor="hand2",
                                 command=self._toggle_recognize)
        self.btn_rec.pack(fill="x", padx=14, pady=(0,6))

        self._sep(right)

        # ── Register button ──────────────────────────────────
        tk.Label(right, text="ENROLL NEW FACE", bg=PANEL, fg=TEXT_SEC,
                 font=("Courier", 8)).pack(anchor="w", **pad, pady=(0,4))

        self.btn_reg = tk.Button(right, text="✚  REGISTER  FACE",
                                 bg=GREEN, fg="#000000",
                                 font=("Courier", 11, "bold"),
                                 activebackground="#00c853",
                                 relief="flat", cursor="hand2",
                                 command=self._start_register)
        self.btn_reg.pack(fill="x", padx=14, pady=(0,4))

        # ── Stop button ──────────────────────────────────────
        self._sep(right)

        self.btn_stop = tk.Button(right, text="■  STOP",
                                  bg=ACCENT2, fg="#ffffff",
                                  font=("Courier", 11, "bold"),
                                  activebackground="#c62828",
                                  relief="flat", cursor="hand2",
                                  state="disabled",
                                  command=self._stop)
        self.btn_stop.pack(fill="x", padx=14, pady=(0,6))

        self._sep(right)

        # ── Status log ───────────────────────────────────────
        tk.Label(right, text="STATUS", bg=PANEL, fg=TEXT_SEC,
                 font=("Courier", 8)).pack(anchor="w", **pad, pady=(0,4))

        log_f = tk.Frame(right, bg=CARD,
                         highlightthickness=1, highlightbackground=BORDER)
        log_f.pack(fill="x", padx=14, pady=(0,14))

        self.status_var = tk.StringVar(value="System ready.")
        tk.Label(log_f, textvariable=self.status_var,
                 bg=CARD, fg=TEXT_PRI,
                 font=("Courier", 9),
                 wraplength=210, justify="left",
                 padx=10, pady=10).pack()

        # Threshold slider
        self._sep(right)
        tk.Label(right, text="CONFIDENCE THRESHOLD", bg=PANEL, fg=TEXT_SEC,
                 font=("Courier", 8)).pack(anchor="w", **pad, pady=(0,2))

        self.thresh_var = tk.IntVar(value=CONFIDENCE_THRESHOLD)
        thresh_lbl_f = tk.Frame(right, bg=PANEL)
        thresh_lbl_f.pack(fill="x", padx=14)
        self.thresh_display = tk.Label(thresh_lbl_f, text=str(CONFIDENCE_THRESHOLD),
                                       bg=PANEL, fg=YELLOW,
                                       font=("Courier", 12, "bold"))
        self.thresh_display.pack(side="right")

        self.slider = tk.Scale(right, from_=500, to=15000,
                               orient="horizontal", variable=self.thresh_var,
                               bg=PANEL, fg=TEXT_SEC, troughcolor=CARD,
                               highlightthickness=0, showvalue=False,
                               command=self._on_thresh)
        self.slider.pack(fill="x", padx=14, pady=(0,18))

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14, pady=8)

    def _on_thresh(self, _=None):
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = self.thresh_var.get()
        self.thresh_display.config(text=str(CONFIDENCE_THRESHOLD))

    # ── Camera ─────────────────────────────────────────────────

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self._loop()

    def _loop(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = self._process(frame)
            img = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.image = img
        self.root.after(15, self._loop)

    def _process(self, frame):
        if self.mode == self.MODE_RECOGNIZE:
            return self._frame_recognize(frame)
        if self.mode == self.MODE_REGISTER:
            return self._frame_register(frame)
        # Idle: just detection overlay
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5, minSize=(60,60))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,80), 1)
        self._hud_idle(frame)
        return frame

    def _hud_idle(self, frame):
        fh, fw = frame.shape[:2]
        cv_text(frame, "IDLE  —  select a mode",
                (10, fh-10), scale=0.42, color=(100,100,130))

    # ── Recognition frame processing ───────────────────────────

    def _frame_recognize(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5, minSize=(60,60))

        for (x,y,w,h) in faces:
            crop = preprocess(gray[y:y+h, x:x+w])
            if self.recognizer:
                lbl, conf = self.recognizer.predict(crop)
                matched    = conf < CONFIDENCE_THRESHOLD
                display    = self.label_to_id.get(lbl, "???") if matched else "Unknown"
                draw_face_box(frame, x, y, w, h, display, conf, matched)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(80,80,80),1)
                cv_text(frame,"No model",(x,y-6),scale=0.5,color=(180,180,180))

        fh, fw = frame.shape[:2]
        cv_text(frame, f"RECOGNIZING  |  threshold {CONFIDENCE_THRESHOLD}",
                (10, fh-10), scale=0.42, color=(0,200,220))
        return frame

    # ── Registration frame processing ──────────────────────────

    def _frame_register(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5, minSize=(60,60))

        # Countdown phase
        if self.reg_waiting_countdown:
            remaining = max(0, int(self.reg_countdown_end - time.time()))
            # Darken
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]),
                          (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(80,80,180),1)

            cv_text(frame, self.reg_countdown_msg,
                    (frame.shape[1]//2 - 210, frame.shape[0]//2 - 30),
                    scale=0.75, color=(0,220,255), thick=2)
            cv_text(frame, str(remaining) if remaining > 0 else "GO!",
                    (frame.shape[1]//2 - 22, frame.shape[0]//2 + 65),
                    scale=2.8, color=(255,255,255), thick=3)
            return frame

        # Capture phase
        for (x,y,w,h) in faces:
            if self.reg_collected < self.reg_target:
                crop = preprocess(gray[y:y+h, x:x+w])
                self.reg_images.append(crop)
                self.reg_labels.append(self.reg_new_label)
                self.reg_collected += 1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,230,80),2)

                if self.reg_collected >= self.reg_target:
                    # Advance step (done on next loop tick via after())
                    self.root.after(50, self._advance_step)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(60,60,60),1)

        draw_register_overlay(frame,
                              self.reg_step_name, self.reg_instruction,
                              self.reg_collected, self.reg_target,
                              self.reg_step_idx, len(GUIDED_STEPS))
        return frame

    # ── Registration logic ──────────────────────────────────────

    def _start_register(self):
        if self.mode != self.MODE_IDLE:
            mb.showwarning("Busy", "Stop the current mode first.")
            return

        uid = sd.askstring("Register", "Enter your 9-digit ID:",
                           parent=self.root)
        if not uid:
            return
        uid = uid.strip()
        if not re.fullmatch(r"\d{9}", uid):
            mb.showerror("Invalid ID", "ID must be exactly 9 digits.")
            return

        images, labels, registry = load_db()

        if uid in registry:
            if not mb.askyesno("Overwrite", f"ID '{uid}' already exists.\nOverwrite?"):
                return
            old_label = registry[uid]
            keep = [(i,l) for i,l in zip(images,labels) if l != old_label]
            images = [k[0] for k in keep]
            labels = [k[1] for k in keep]
            new_label = old_label
        else:
            used = set(registry.values())
            new_label = 0
            while new_label in used:
                new_label += 1
            registry[uid] = new_label

        self.reg_images   = images
        self.reg_labels   = labels
        self.reg_registry = registry
        self.reg_new_label= new_label
        self.reg_uid      = uid
        self.reg_step_idx = 0

        self._set_mode(self.MODE_REGISTER)
        self._set_status(f"Registering ID: {uid}\nFollow on-screen poses.")
        self._begin_step(0)

    def _begin_step(self, idx):
        if idx >= len(GUIDED_STEPS):
            self._finish_register()
            return
        name, instruction, target = GUIDED_STEPS[idx]
        self.reg_step_idx    = idx
        self.reg_step_name   = name
        self.reg_instruction = instruction
        self.reg_target      = target
        self.reg_collected   = 0

        # Start countdown
        self.reg_waiting_countdown = True
        self.reg_countdown_end     = time.time() + 3
        self.reg_countdown_msg     = f"Get ready:  {name}"
        self.root.after(3100, self._end_countdown)

    def _end_countdown(self):
        self.reg_waiting_countdown = False

    def _advance_step(self):
        self._begin_step(self.reg_step_idx + 1)

    def _finish_register(self):
        save_db(self.reg_images, self.reg_labels, self.reg_registry)
        self._refresh_db_info()
        self._set_mode(self.MODE_IDLE)
        mb.showinfo("Done",
                    f"Registration complete!\nID: {self.reg_uid}\n"
                    f"Samples: {len(self.reg_images)}")
        self._set_status(f"Registered: {self.reg_uid}\nDB updated.")

    # ── Recognize toggle ────────────────────────────────────────

    def _toggle_recognize(self):
        if self.mode == self.MODE_RECOGNIZE:
            self._stop()
            return

        if self.mode != self.MODE_IDLE:
            mb.showwarning("Busy", "Stop the current mode first.")
            return

        if not os.path.exists(DB_FILE) or not os.path.exists(REGISTRY):
            mb.showerror("No Database",
                         "No face database found.\nPlease register faces first.")
            return

        try:
            rec, lbl_map = train_recognizer()
        except Exception as e:
            mb.showerror("Model Error", str(e))
            return

        if rec is None:
            mb.showerror("Not Enough Data",
                         "Need at least 2 enrolled people to run recognition.")
            return

        self.recognizer  = rec
        self.label_to_id = lbl_map
        self._set_mode(self.MODE_RECOGNIZE)
        self._set_status("Recognition active.\nFaces will be labeled live.")
        self.btn_rec.config(text="■  STOP  RECOGNITION", bg=ACCENT2, fg="#ffffff")

    # ── Stop / idle ─────────────────────────────────────────────

    def _stop(self):
        self._set_mode(self.MODE_IDLE)
        self._set_status("Stopped. Ready.")
        self.btn_rec.config(text="▶  START  RECOGNITION", bg=ACCENT, fg="#000000")

    # ── Helpers ─────────────────────────────────────────────────

    def _set_mode(self, mode):
        self.mode = mode
        labels = {
            self.MODE_IDLE:      ("● IDLE",       TEXT_SEC),
            self.MODE_RECOGNIZE: ("◉ RECOGNIZING", ACCENT),
            self.MODE_REGISTER:  ("◉ REGISTERING", GREEN),
        }
        txt, col = labels.get(mode, ("● IDLE", TEXT_SEC))
        self.mode_var.set(txt)

        disabled = mode != self.MODE_IDLE
        self.btn_stop.config(state="normal" if disabled else "disabled")
        self.btn_reg.config(state="disabled" if disabled else "normal")

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _refresh_db_info(self):
        people  = 0
        samples = 0
        if os.path.exists(REGISTRY):
            with open(REGISTRY) as f:
                people = len(json.load(f))
        if os.path.exists(DB_FILE):
            samples = len(np.load(DB_FILE)["labels"])
        self.lbl_people.config(text=str(people))
        self.lbl_samples.config(text=str(samples))

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = FaceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
