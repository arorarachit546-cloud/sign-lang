"""
Real-time Sign Language Recognition and Translation Assistant
Orchestrates webcam capture, MediaPipe Holistic extraction, deep learning inference,
temporal debouncing, HUD rendering, and auditory speech synthesis.
"""

import time
from typing import List, Optional
import cv2
import numpy as np
import torch

from src.assistant.buffer_manager import SlidingWindowBuffer, TemporalDebouncer
from src.assistant.tts_engine import TTSEngine
from src.data.dataset import normalize_landmarks
from src.extraction.holistic_extractor import HolisticLandmarkExtractor


class SignLanguageAssistant:
    """
    Real-time Interactive Assistant for Sign Language Translation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        confidence_threshold: float = 0.75,
        debounce_frames: int = 5,
        camera_index: int = 0,
        enable_tts: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.class_names = class_names
        self.camera_index = camera_index

        self.extractor = HolisticLandmarkExtractor()
        self.buffer = SlidingWindowBuffer(window_size=30, feature_dim=201)
        self.debouncer = TemporalDebouncer(
            confidence_threshold=confidence_threshold,
            consecutive_frames_required=debounce_frames,
        )
        self.tts = TTSEngine(enabled=enable_tts)

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Runs one step of the real-time translation pipeline on an incoming video frame.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks, presence = self.extractor.extract_landmarks(frame_rgb)

        self.buffer.append(landmarks)

        predicted_class_name = "Waiting for gesture..."
        confidence = 0.0
        top_probs = []

        if self.buffer.is_ready():
            window = self.buffer.get_window()  # (30, 201)
            norm_window = normalize_landmarks(window)
            tensor_x = torch.from_numpy(norm_window).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                logits = self.model(tensor_x)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            predicted_class_name = self.class_names[pred_idx]

            # Update debouncer
            committed, committed_name = self.debouncer.update(
                predicted_class=pred_idx,
                confidence=confidence,
                class_name=predicted_class_name,
            )

            if committed and committed_name:
                self.tts.speak(committed_name)

            # Get top 3 predictions for display
            top_indices = np.argsort(probs)[-3:][::-1]
            top_probs = [(self.class_names[idx], probs[idx]) for idx in top_indices]

        # Draw HUD on frame
        display_frame = self._render_hud(
            frame_bgr=frame_bgr,
            current_pred=predicted_class_name,
            confidence=confidence,
            sentence=self.debouncer.get_sentence(),
            top_probs=top_probs,
            presence=presence,
        )
        return display_frame

    def _render_hud(
        self,
        frame_bgr: np.ndarray,
        current_pred: str,
        confidence: float,
        sentence: str,
        top_probs: List,
        presence: dict,
    ) -> np.ndarray:
        h, w, _ = frame_bgr.shape
        overlay = frame_bgr.copy()

        # Top banner for live prediction
        cv2.rectangle(overlay, (0, 0), (w, 65), (25, 25, 30), -1)
        # Bottom banner for translated sentence transcript
        cv2.rectangle(overlay, (0, h - 60), (w, h), (25, 25, 30), -1)

        # Alpha blend banners
        cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0, frame_bgr)

        # Live prediction text
        pred_text = f"Sign: {current_pred} ({confidence * 100:.1f}%)"
        color = (0, 255, 127) if confidence >= 0.75 else (0, 191, 255)
        cv2.putText(frame_bgr, pred_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = w - 240, 22, 200, 20
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        fill_w = int(bar_w * min(1.0, confidence))
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        # Bottom transcript text
        transcript = f"Translation: \"{sentence}\"" if sentence else "Translation: [Listening for signs...]"
        cv2.putText(frame_bgr, transcript, (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Sidebar HUD for Top-3 probabilities
        if top_probs:
            panel_y = 90
            for sign, prob in top_probs:
                p_text = f"{sign}: {prob*100:.1f}%"
                cv2.putText(frame_bgr, p_text, (w - 200, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                panel_y += 24

        return frame_bgr

    def run(self, max_frames: Optional[int] = None) -> None:
        """Runs the live webcam loop."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[Error]: Unable to open webcam at index {self.camera_index}.")
            return

        print("--- Real-time Sign Language Translation Assistant Active ---")
        print("Press 'q' in the camera window to exit. Press 'c' to clear transcript.")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed = self.process_frame(frame)
                cv2.imshow("Sign Language Recognition & Translation Assistant", processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    self.debouncer.reset()

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            self.tts.stop()
            print("--- Assistant terminated cleanly ---")
