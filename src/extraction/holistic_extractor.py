"""
Holistic Landmark Extractor Module
Extracts 3D skeletal topologies (Hands, Upper Body Pose) using Google MediaPipe Holistic.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class HolisticLandmarkExtractor:
    """
    Extracts 75 keypoints (201 coordinates: x, y, z) per frame:
    - Left Hand: 21 landmarks (63 coords)
    - Right Hand: 21 landmarks (63 coords)
    - Upper Pose: 25 landmarks (75 coords) - indices 0 through 24
    Total feature dimensions: (21 + 21 + 25) * 3 = 201 coordinates per frame.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._holistic = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self._holistic = self.mp_holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=self.model_complexity,
                refine_face_landmarks=False,
            )
            self.available = True
        except (ImportError, Exception) as e:
            self.available = False
            self.error_msg = str(e)

    def extract_landmarks(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Extract coordinates from a single RGB image frame.

        Args:
            frame_rgb: numpy array of shape (H, W, 3) in RGB order.

        Returns:
            landmarks_vector: np.ndarray of shape (201,)
            presence_dict: Dict indicating presence of left_hand, right_hand, pose
        """
        presence = {"left_hand": False, "right_hand": False, "pose": False}
        if not self.available or self._holistic is None:
            # Fallback zero vector
            return np.zeros(201, dtype=np.float32), presence

        results = self._holistic.process(frame_rgb)

        # 1. Pose landmarks (Upper 25 landmarks: indices 0 to 24)
        pose_coords = []
        if results.pose_landmarks:
            presence["pose"] = True
            for landmark in results.pose_landmarks.landmark[:25]:
                pose_coords.extend([landmark.x, landmark.y, landmark.z])
        else:
            pose_coords = [0.0] * (25 * 3)

        # 2. Left Hand landmarks (21 landmarks)
        left_hand_coords = []
        if results.left_hand_landmarks:
            presence["left_hand"] = True
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_coords.extend([landmark.x, landmark.y, landmark.z])
        else:
            left_hand_coords = [0.0] * (21 * 3)

        # 3. Right Hand landmarks (21 landmarks)
        right_hand_coords = []
        if results.right_hand_landmarks:
            presence["right_hand"] = True
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_coords.extend([landmark.x, landmark.y, landmark.z])
        else:
            right_hand_coords = [0.0] * (21 * 3)

        # Concatenate: [Pose (75), Left Hand (63), Right Hand (63)] -> 201 dims
        feature_vector = np.array(pose_coords + left_hand_coords + right_hand_coords, dtype=np.float32)
        return feature_vector, presence

    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw skeleton visual landmarks on an OpenCV BGR frame for HUD."""
        if not self.available:
            return frame_bgr
        annotated = frame_bgr.copy()
        try:
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                )
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                )
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                )
        except Exception:
            pass
        return annotated

    def close(self):
        if self._holistic:
            self._holistic.close()
