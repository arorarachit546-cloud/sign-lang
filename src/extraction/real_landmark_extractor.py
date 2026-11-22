"""
Real-time 3D Landmark Extractor & Strict Kinematic Gesture Recognizer
Uses Google MediaPipe Tasks API HandLandmarker with precise joint vector kinematics.
Robustly classifies Hello, Thank You, Yes, No, Help, Emergency, Please, I Love You, Water, Goodbye,
and strictly rejects non-sign gestures (e.g. arbitrary single fingers, random poses) as 'Unrecognized Sign'.
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeTasksExtractor:
    """
    Extracts 3D landmarks and computes rigorous geometric invariants:
    - 5 continuous finger extension ratios [0.0, 1.0]
    - Inter-finger tip distances
    - Palm orientation & elevation
    - Spatial relationship to facial landmarks (nose/chin)
    - Explicit non-sign rejection
    """

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17),                                # Palm base
    ]

    def __init__(
        self,
        hand_model_path: str = "hand_landmarker.task",
        pose_model_path: str = "pose_landmarker.task",
    ):
        self.hand_model_path = hand_model_path
        self.pose_model_path = pose_model_path
        self.hand_detector = None
        self.pose_detector = None
        self.prev_wrist = None
        self.prev_time = None
        self.motion_velocity = [0.0, 0.0]  # [vx, vy]
        self._init_detectors()

    def _init_detectors(self):
        if os.path.exists(self.hand_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=self.hand_model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.50,
                    min_hand_presence_confidence=0.50,
                    min_tracking_confidence=0.50,
                )
                self.hand_detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[Warning] Failed to initialize HandLandmarker: {e}")

        if os.path.exists(self.pose_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=self.pose_model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    min_pose_detection_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                self.pose_detector = vision.PoseLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[Warning] Failed to initialize PoseLandmarker: {e}")

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        annotated = frame_bgr.copy()
        features = np.zeros(201, dtype=np.float32)
        info = {
            "hand_detected": False,
            "num_hands": 0,
            "gesture_hint": "Waiting for hand...",
            "gesture_confidence": 0.0,
            "fingers_extended": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False},
            "finger_scores": [0.0, 0.0, 0.0, 0.0, 0.0],
            "landmarks_2d": [],
        }

        # 1. Pose landmarks (optional upper body)
        nose_pt = None
        if self.pose_detector is not None:
            try:
                pose_result = self.pose_detector.detect(mp_image)
                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    for i, lm in enumerate(pose_result.pose_landmarks[0][:25]):
                        features[i * 3 : i * 3 + 3] = [lm.x, lm.y, lm.z]
                        if i == 0:
                            nose_pt = [lm.x, lm.y, lm.z]
                        if i in [0, 11, 12]:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
            except Exception:
                pass

        # 2. Hand landmarks
        if self.hand_detector is not None:
            try:
                hand_result = self.hand_detector.detect(mp_image)
                if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                    info["hand_detected"] = True
                    info["num_hands"] = len(hand_result.hand_landmarks)

                    now = time.time()
                    for hand_idx, landmarks in enumerate(hand_result.hand_landmarks[:2]):
                        handedness_label = "Right"
                        if hand_result.handedness and hand_idx < len(hand_result.handedness):
                            if len(hand_result.handedness[hand_idx]) > 0:
                                handedness_label = hand_result.handedness[hand_idx][0].category_name

                        offset = 138 if handedness_label == "Right" else 75

                        lms_coords = []
                        for i, lm in enumerate(landmarks):
                            features[offset + i * 3 : offset + i * 3 + 3] = [lm.x, lm.y, lm.z]
                            lms_coords.append([lm.x, lm.y, lm.z])

                        # Draw skeletal connections
                        for s_idx, e_idx in self.HAND_CONNECTIONS:
                            pt1 = (int(landmarks[s_idx].x * w), int(landmarks[s_idx].y * h))
                            pt2 = (int(landmarks[e_idx].x * w), int(landmarks[e_idx].y * h))
                            cv2.line(annotated, pt1, pt2, (0, 255, 127), 2, cv2.LINE_AA)

                        # Draw joint circles
                        for i, lm in enumerate(landmarks):
                            pt = (int(lm.x * w), int(lm.y * h))
                            is_tip = i in [4, 8, 12, 16, 20]
                            color = (0, 0, 255) if is_tip else (0, 255, 127)
                            cv2.circle(annotated, pt, 6 if is_tip else 3, color, -1)

                        if hand_idx == 0:
                            # Primary hand analysis
                            lms_np = np.array(lms_coords, dtype=np.float32)
                            info["landmarks_2d"] = [[lm.x, lm.y] for lm in landmarks]

                            # Calculate wrist velocity
                            wrist = lms_np[0]
                            if self.prev_wrist is not None and self.prev_time is not None:
                                dt = max(now - self.prev_time, 1e-4)
                                self.motion_velocity = [
                                    (wrist[0] - self.prev_wrist[0]) / dt,
                                    (wrist[1] - self.prev_wrist[1]) / dt,
                                ]
                            self.prev_wrist = wrist
                            self.prev_time = now

                            ext_bools, ext_scores = self._analyze_finger_extensions(lms_np)
                            info["fingers_extended"] = ext_bools
                            info["finger_scores"] = ext_scores

                            gesture, conf = self._classify_hand_gesture(
                                lms_np, ext_bools, ext_scores, info["num_hands"], self.motion_velocity, nose_pt
                            )
                            info["gesture_hint"] = gesture
                            info["gesture_confidence"] = conf

            except Exception as e:
                print(f"[Extractor Error] {e}")

        return annotated, features, info

    def _analyze_finger_extensions(self, lms: np.ndarray) -> Tuple[Dict[str, bool], List[float]]:
        """
        Calculates robust continuous [0.0, 1.0] and boolean finger extension states.
        Scale-invariant relative to palm size (wrist to middle MCP).
        """
        wrist = lms[0]
        middle_mcp = lms[9]
        index_mcp = lms[5]
        index_pip = lms[6]
        palm_size = float(np.linalg.norm(middle_mcp - wrist))
        if palm_size < 1e-4:
            palm_size = 0.15

        finger_scores = []
        finger_bools = {}

        # 1. Thumb analysis:
        thumb_tip = lms[4]
        d_tip_idx_pip = float(np.linalg.norm(thumb_tip - index_pip) / palm_size)
        d_tip_mid = float(np.linalg.norm(thumb_tip - middle_mcp) / palm_size)
        d_tip_wrist = float(np.linalg.norm(thumb_tip - wrist) / palm_size)

        # Knuckle height
        pips_y = [lms[6, 1], lms[10, 1], lms[14, 1], lms[18, 1]]
        min_pip_y = min(pips_y)

        # Thumb is tucked in a fist if it wraps against index/middle finger and is below knuckles
        is_tucked_fist = (d_tip_idx_pip < 0.32 and thumb_tip[1] >= min_pip_y - 0.05 * palm_size)
        is_thumbs_up = (thumb_tip[1] < min_pip_y - 0.18 * palm_size)

        if is_tucked_fist:
            thumb_is_ext = False
            thumb_score = 0.0
        elif is_thumbs_up:
            thumb_is_ext = True
            thumb_score = 0.95
        else:
            thumb_is_ext = bool((d_tip_mid > 0.46) or (d_tip_wrist > 0.82))
            thumb_score = min(max((d_tip_mid - 0.40) / 0.35, 0.0), 1.0) if thumb_is_ext else 0.0

        finger_scores.append(round(thumb_score, 3))
        finger_bools["thumb"] = thumb_is_ext

        # 2. Four fingers: Index, Middle, Ring, Pinky
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        mcps = [5, 9, 13, 17]
        finger_names = ["index", "middle", "ring", "pinky"]

        for tip_idx, pip_idx, mcp_idx, fname in zip(tips, pips, mcps, finger_names):
            tip = lms[tip_idx]
            pip = lms[pip_idx]
            mcp = lms[mcp_idx]

            d_tw = np.linalg.norm(tip - wrist)
            d_pw = np.linalg.norm(pip - wrist)
            d_tm = np.linalg.norm(tip - mcp)
            d_pm = np.linalg.norm(pip - mcp)

            ratio = float(d_tw / (d_pw + 1e-6))
            score = min(max((ratio - 0.95) / 0.30, 0.0), 1.0)
            is_ext = bool(ratio > 1.05 and d_tm > d_pm * 1.02)

            finger_scores.append(round(score, 3))
            finger_bools[fname] = is_ext

        return finger_bools, finger_scores

    def _classify_hand_gesture(
        self,
        lms: np.ndarray,
        f_bool: Dict[str, bool],
        f_scores: List[float],
        num_hands: int,
        velocity: List[float],
        nose_pt: Optional[List[float]] = None,
    ) -> Tuple[str, float]:
        """
        Classifies signs with robust kinematic pattern matching.
        """
        t_score, i_score, m_score, r_score, p_score = f_scores
        t, i, m, r, p = f_bool["thumb"], f_bool["index"], f_bool["middle"], f_bool["ring"], f_bool["pinky"]

        wrist = lms[0]
        middle_mcp = lms[9]
        palm_size = float(np.linalg.norm(middle_mcp - wrist))
        if palm_size < 1e-4:
            palm_size = 0.15

        vx, vy = velocity
        speed = math.hypot(vx, vy)
        wrist_x, wrist_y = float(lms[0, 0]), float(lms[0, 1])

        thumb_tip = lms[4]
        index_tip = lms[8]
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))

        # Knuckle elevation
        pips_y = [lms[6, 1], lms[10, 1], lms[14, 1], lms[18, 1]]
        min_pip_y = min(pips_y)
        is_thumbs_up = bool(thumb_tip[1] < min_pip_y - 0.18 * palm_size)

        # 1. Emergency: Strictly requires 2 hands active or extreme oscillation
        if num_hands >= 2 or speed > 2.8:
            return "Emergency", 0.95

        # 2. I Love You (Universal ASL ILY):
        # Thumb, Index, Pinky extended; Middle and Ring curled
        if (t or t_score > 0.50) and (i or i_score > 0.55) and (p or p_score > 0.50) and m_score < 0.45 and r_score < 0.45:
            return "I Love You", 0.98

        # 3. Water (ASL 'W'):
        # Index, Middle, Ring extended; Pinky curled
        if (i or i_score > 0.55) and (m or m_score > 0.55) and (r or r_score > 0.55) and p_score < 0.45:
            return "Water", 0.96

        # 4. Closed Fist / Thumbs Up (Yes vs Help):
        all_fingers_curled = (not i) and (not m) and (not r) and (not p)
        scores_curled = (i_score < 0.42 and m_score < 0.42 and r_score < 0.42 and p_score < 0.42)

        if all_fingers_curled or scores_curled:
            if is_thumbs_up:
                # Thumb is sticking straight UP into the air
                return "Help", 0.96
            else:
                # Closed Fist: all fingers curled, thumb tucked across knuckles
                return "Yes", 0.97

        # 5. No: Index and Middle extended pinching or tapping Thumb; Ring & Pinky curled
        if i_score > 0.40 and m_score > 0.40 and r_score < 0.40 and p_score < 0.40 and pinch_dist < 0.16:
            return "No", 0.93

        # 6. OPEN HAND FAMILY (Hello, Goodbye, Thank You, Please):
        four_fingers_ext = (i and m and r and p) or (i_score > 0.48 and m_score > 0.48 and r_score > 0.48 and p_score > 0.48)
        if four_fingers_ext:
            # Check spatial relationship to face / torso if pose is available
            if nose_pt is not None:
                nose_x, nose_y = nose_pt[0], nose_pt[1]
                chin_y = nose_y + 0.08
                dist_to_chin = math.hypot(wrist_x - nose_x, wrist_y - chin_y)

                # Thank You: flat hand at chin moving forward/downward
                if dist_to_chin < 0.16 and (vy > 0.05 or abs(wrist_x - nose_x) < 0.12 and wrist_y < nose_y + 0.18):
                    return "Thank You", 0.95

                # Please: flat hand placed against the chest (well below chin, centered on torso)
                if wrist_y > nose_y + 0.22 and abs(wrist_x - nose_x) < 0.22:
                    return "Please", 0.94

            # Waving side-to-side indicates Goodbye
            if abs(vx) > 0.22 or (speed > 0.40 and abs(vx) > abs(vy)):
                return "Goodbye", 0.95

            # Default upright open hand beside head: Hello
            return "Hello", 0.98

        # 7. Relaxed or partial wave
        if i_score > 0.55 and m_score > 0.55 and (t or t_score > 0.50 or p or p_score > 0.50):
            return "Hello", 0.91

        # Explicit rejection for arbitrary non-signs (e.g. single index, single middle, V-sign)
        return "Unrecognized Sign", 0.0
