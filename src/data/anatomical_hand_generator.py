"""
Anatomical 3D Hand Skeletal Trajectory Generator
Constructs biomechanically accurate 21-joint MediaPipe 3D coordinates
for real sign language handshapes (Hello, Thank You, Yes, No, Help, Emergency, Please, I Love You, Water, Goodbye, Neutral).
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def create_hand_skeleton(
    finger_curls: Dict[str, float],
    finger_spreads: Dict[str, float] = None,
    wrist_pos: Tuple[float, float, float] = (0.5, 0.6, 0.0),
    hand_scale: float = 0.25,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """
    Creates 21 3D landmarks for a human hand based on joint bending angles:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, Tip)
    5-8: Index (MCP, PIP, DIP, Tip)
    9-12: Middle (MCP, PIP, DIP, Tip)
    13-16: Ring (MCP, PIP, DIP, Tip)
    17-20: Pinky (MCP, PIP, DIP, Tip)

    finger_curls: float in [0.0, 1.0] where 0.0 = fully extended, 1.0 = curled into fist
    """
    if finger_spreads is None:
        finger_spreads = {"thumb": -25.0, "index": -10.0, "middle": 0.0, "ring": 10.0, "pinky": 20.0}

    landmarks = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist_pos
    landmarks[0] = [wx, wy, wz]

    # Finger bone segment lengths (MCP, PIP, DIP, Tip) relative to hand_scale
    finger_bones = {
        "thumb":  [0.30, 0.25, 0.22],
        "index":  [0.40, 0.28, 0.20, 0.16],
        "middle": [0.42, 0.30, 0.22, 0.17],
        "ring":   [0.39, 0.27, 0.19, 0.16],
        "pinky":  [0.35, 0.22, 0.16, 0.14],
    }

    # Base MCP offsets from wrist
    mcp_bases = {
        "thumb":  (-0.15, -0.18, 0.02),
        "index":  (-0.10, -0.40, 0.0),
        "middle": ( 0.00, -0.42, 0.0),
        "ring":   ( 0.09, -0.39, 0.0),
        "pinky":  ( 0.16, -0.34, 0.0),
    }

    # Rotate coordinates in 2D plane
    rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    def rot2d(x, y):
        return x * cos_r - y * sin_r, x * sin_r + y * cos_r

    # Build fingers
    joint_indices = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }

    for finger, indices in joint_indices.items():
        base_ox, base_oy, base_oz = mcp_bases[finger]
        spread_deg = finger_spreads.get(finger, 0.0)
        curl = finger_curls.get(finger, 0.0)  # 0 = straight up, 1 = curled down

        cur_x = wx + base_ox * hand_scale
        cur_y = wy + base_oy * hand_scale
        cur_z = wz + base_oz * hand_scale

        landmarks[indices[0]] = [cur_x, cur_y, cur_z]

        # Finger heading angle
        base_angle_rad = math.radians(-90.0 + spread_deg)
        bones = finger_bones[finger]

        for b_idx, bone_len in enumerate(bones[1:], start=1):
            seg_len = bone_len * hand_scale

            # Each curled joint bends further towards the palm
            bend_angle = curl * (math.pi * 0.55) * b_idx
            effective_angle = base_angle_rad + (bend_angle if finger != "thumb" else bend_angle * 0.6)

            dx = seg_len * math.cos(effective_angle)
            dy = seg_len * math.sin(effective_angle)
            dz = -seg_len * math.sin(bend_angle * 0.5) * curl

            cur_x += dx
            cur_y += dy
            cur_z += dz

            landmarks[indices[b_idx]] = [cur_x, cur_y, cur_z]

    return landmarks


def generate_anatomical_gesture(class_idx: int, num_frames: int = 30) -> np.ndarray:
    """
    Generates a realistic 30-frame sequence of 201-dim landmark vectors
    based on true human hand configurations and dynamic motions:
    0: Hello       (All 5 fingers extended, waving side-to-side)
    1: Thank You   (Flat hand touching chin, moving forward/downward)
    2: Yes         (Closed fist with all 5 fingers curled, nodding vertically)
    3: No          (Index & middle finger snapping to thumb tip)
    4: Help        (Thumbs up, all 4 fingers curled)
    5: Emergency   (Both hands or rapid oscillation)
    6: Please      (Flat open hand rubbing chest in circle)
    7: I Love You  (Thumb, Index, Pinky extended; Middle, Ring curled)
    8: Water       ('W' shape: Index, Middle, Ring extended; Pinky curled)
    9: Goodbye     (Open hand waving laterally)
    10: Neutral    (Random relaxed or partial hand poses)
    """
    seq = np.zeros((num_frames, 201), dtype=np.float32)

    # Base pose landmarks: Upper torso
    for f in range(num_frames):
        pose = np.zeros(75, dtype=np.float32)
        pose[11 * 3 : 11 * 3 + 3] = [-0.25, -0.2, 0.0]  # Left shoulder
        pose[12 * 3 : 12 * 3 + 3] = [ 0.25, -0.2, 0.0]  # Right shoulder
        pose[0 : 3] = [0.0, -0.45, 0.0]                 # Nose
        seq[f, :75] = pose

    # Hand shape parameters
    if class_idx == 0:  # Hello: All extended, waving laterally
        curls = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        spreads = {"thumb": -35.0, "index": -15.0, "middle": 0.0, "ring": 15.0, "pinky": 30.0}
        for f in range(num_frames):
            t = f / num_frames * 2 * math.pi
            wx = 0.35 + 0.08 * math.sin(3 * t)
            wy = -0.35 + 0.02 * math.cos(3 * t)
            rot = 10.0 * math.sin(3 * t)
            hand_lms = create_hand_skeleton(curls, spreads, (wx, wy, 0.0), rotation_deg=rot)
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 1:  # Thank you: Flat hand moving forward from chin
        curls = {"thumb": 0.1, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        spreads = {"thumb": -20.0, "index": -5.0, "middle": 0.0, "ring": 5.0, "pinky": 10.0}
        for f in range(num_frames):
            progress = f / num_frames
            wx = 0.05
            wy = -0.35 + 0.20 * progress
            wz = -0.10 - 0.25 * progress
            hand_lms = create_hand_skeleton(curls, spreads, (wx, wy, wz))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 2:  # Yes: Closed fist nodding up and down
        curls = {"thumb": 0.95, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}
        spreads = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        for f in range(num_frames):
            t = f / num_frames * 2 * math.pi
            wx = 0.25
            wy = -0.15 + 0.08 * math.sin(4 * t)
            hand_lms = create_hand_skeleton(curls, spreads, (wx, wy, 0.0))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 3:  # No: Index & Middle tapping Thumb
        spreads = {"thumb": -15.0, "index": -5.0, "middle": 5.0, "ring": 25.0, "pinky": 35.0}
        for f in range(num_frames):
            t = f / num_frames * 2 * math.pi
            pinch = 0.5 + 0.5 * math.sin(4 * t)
            curls = {"thumb": 0.3 * pinch, "index": 0.7 * pinch, "middle": 0.7 * pinch, "ring": 1.0, "pinky": 1.0}
            hand_lms = create_hand_skeleton(curls, spreads, (0.2, -0.2, 0.0))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 4:  # Help: Thumbs up (Thumb extended, 4 curled)
        curls = {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}
        spreads = {"thumb": -45.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        for f in range(num_frames):
            hand_lms = create_hand_skeleton(curls, spreads, (0.15, -0.2, 0.0))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 5:  # Emergency: Rapid oscillation
        curls = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        spreads = {"thumb": -30.0, "index": -15.0, "middle": 0.0, "ring": 15.0, "pinky": 30.0}
        for f in range(num_frames):
            t = f / num_frames * 2 * math.pi
            wx = 0.25 + 0.12 * math.sin(6 * t)
            wy = -0.25 + 0.06 * math.cos(6 * t)
            hand_lms = create_hand_skeleton(curls, spreads, (wx, wy, 0.0))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 6:  # Please: Flat hand rubbing chest in circle
        curls = {"thumb": 0.1, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        spreads = {"thumb": -20.0, "index": -5.0, "middle": 0.0, "ring": 5.0, "pinky": 10.0}
        for f in range(num_frames):
            t = f / num_frames * 2 * math.pi
            wx = 0.0 + 0.08 * math.cos(2 * t)
            wy = -0.15 + 0.08 * math.sin(2 * t)
            hand_lms = create_hand_skeleton(curls, spreads, (wx, wy, 0.0))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 7:  # I Love You: Thumb, Index, Pinky extended; Middle, Ring curled
        curls = {"thumb": 0.0, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0}
        spreads = {"thumb": -40.0, "index": -15.0, "middle": 0.0, "ring": 10.0, "pinky": 30.0}
        for f in range(num_frames):
            progress = f / num_frames
            hand_lms = create_hand_skeleton(curls, spreads, (0.25, -0.3, -0.05 * progress))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 8:  # Water: W-shape (Index, Middle, Ring extended; Pinky curled)
        curls = {"thumb": 0.8, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0}
        spreads = {"thumb": -20.0, "index": -20.0, "middle": 0.0, "ring": 20.0, "pinky": 30.0}
        for f in range(num_frames):
            t = f / num_frames * 2 * math.pi
            wy = -0.35 + 0.04 * math.sin(4 * t)
            hand_lms = create_hand_skeleton(curls, spreads, (0.05, wy, 0.0))
            seq[f, 138:201] = hand_lms.flatten()

    elif class_idx == 9:  # Goodbye: Open hand waving laterally
        curls = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        spreads = {"thumb": -35.0, "index": -15.0, "middle": 0.0, "ring": 15.0, "pinky": 30.0}
        for f in range(num_frames):
            t = f / num_frames * 2 * math.pi
            wx = 0.35 + 0.15 * math.sin(2 * t)
            rot = 20.0 * math.sin(2 * t)
            hand_lms = create_hand_skeleton(curls, spreads, (wx, -0.4, 0.0), rotation_deg=rot)
            seq[f, 138:201] = hand_lms.flatten()

    else:  # Neutral / Non-sign hand shapes
        random_curls = {
            "thumb": float(np.random.uniform(0.3, 0.8)),
            "index": float(np.random.uniform(0.2, 0.8)),
            "middle": float(np.random.uniform(0.2, 0.8)),
            "ring": float(np.random.uniform(0.3, 0.9)),
            "pinky": float(np.random.uniform(0.3, 0.9)),
        }
        for f in range(num_frames):
            hand_lms = create_hand_skeleton(random_curls, wrist_pos=(0.3, -0.2, 0.0))
            seq[f, 138:201] = hand_lms.flatten()

    # Add realistic jitter
    seq += np.random.normal(0, 0.005, size=seq.shape).astype(np.float32)
    return seq


def create_anatomical_dataset(
    num_classes: int = 10,
    samples_per_class: int = 180,
    num_signers: int = 6,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Creates multi-signer dataset grounded in real hand anatomy."""
    samples = []
    labels = []
    signer_ids = []

    samples_per_signer = samples_per_class // num_signers
    for c in range(num_classes):
        for s in range(num_signers):
            for _ in range(samples_per_signer):
                traj = generate_anatomical_gesture(class_idx=c, num_frames=30)
                samples.append(traj)
                labels.append(c)
                signer_ids.append(s)

    return samples, labels, signer_ids
