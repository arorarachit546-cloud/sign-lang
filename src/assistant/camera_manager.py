"""
Thread-Safe Hardware Camera Manager
Manages an on-demand DirectShow capture thread for Windows (cv2.CAP_DSHOW)
Only opens camera device when explicitly requested, preventing device contention
with browser WebRTC/HTML5 webcam access.
"""

import threading
import time
from typing import Optional
import cv2
import numpy as np


class HardwareCameraManager:
    """
    On-demand camera worker that manages DirectShow camera capture.
    Only active when DirectShow mode is explicitly toggled by user.
    """

    def __init__(self, camera_idx: int = 0):
        self.camera_idx = camera_idx
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.is_hardware_ready = False
        self.frame_count = 0

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()

    def _capture_worker(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.is_hardware_ready = True
                print(f"[CameraManager] Opened camera {self.camera_idx} via DirectShow.")
            else:
                print(f"[CameraManager] Camera {self.camera_idx} unavailable via DirectShow.")
                self.is_hardware_ready = False
        except Exception as e:
            print(f"[CameraManager] Error opening camera: {e}")
            self.is_hardware_ready = False

        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
                else:
                    time.sleep(0.03)
            else:
                # Provide a clean placeholder frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "DirectShow Camera not available",
                    (80, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    placeholder,
                    "Please switch to 'Browser Webcam' mode",
                    (80, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                with self.lock:
                    self.latest_frame = placeholder
                time.sleep(0.1)

        # Cleanup on stop
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_hardware_ready = False
        print("[CameraManager] Released DirectShow camera device.")

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
