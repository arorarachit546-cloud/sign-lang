"""
Synthetic Sign Language Landmark Sequence Generator
Generates realistic 3D spatiotemporal kinematic trajectories across multiple distinct signers
for immediate reproducibility, unit testing, and benchmarking.
"""

from typing import Dict, List, Tuple
import numpy as np


def generate_gesture_trajectory(
    class_idx: int,
    num_frames: int = 30,
    feature_dim: int = 201,
    signer_variation_factor: float = 1.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """
    Generates a 3D spatiotemporal trajectory with class-distinct kinematic patterns:
    Classes:
    0: Hello (Hand waving above shoulder)
    1: Thank You (Hand moves from chin outward)
    2: Yes (Fist nodding up and down)
    3: No (Index and middle finger tapping thumb)
    4: Help (Flat palm uplifting closed fist)
    5: Emergency (Both hands oscillating rapidly)
    6: Please (Hand rubbing in circular motion on chest)
    7: I Love You (Thumb, index, pinky extended with forward motion)
    8: Water (W-handshape tapping chin)
    9: Goodbye (Open hand swinging side to side)
    """
    t = np.linspace(0, 2 * np.pi, num_frames)
    seq = np.zeros((num_frames, feature_dim), dtype=np.float32)

    # Base pose skeleton: upper torso coordinates (25 keypoints * 3 coords = 75)
    # Nose=(0, -0.5, 0), Left Shoulder=(-0.3, -0.2, 0), Right Shoulder=(0.3, -0.2, 0)
    for frame_idx in range(num_frames):
        # Upper pose default stationary coordinates
        pose = np.zeros(75, dtype=np.float32)
        pose[11 * 3 : 11 * 3 + 3] = [-0.3 * signer_variation_factor, -0.2, 0.0]  # Left shoulder
        pose[12 * 3 : 12 * 3 + 3] = [0.3 * signer_variation_factor, -0.2, 0.0]   # Right shoulder
        seq[frame_idx, :75] = pose

    # Left hand: 63 coordinates (indices 75 to 137)
    # Right hand: 63 coordinates (indices 138 to 200)
    freq = (class_idx + 1) * 0.5
    phase_shift = class_idx * (np.pi / 5.0)

    for i in range(num_frames):
        cur_t = t[i]
        # Class-specific kinematic trajectory for right hand (primary dominant hand)
        if class_idx == 0:  # Hello: waving
            rh_x = 0.35 + 0.12 * np.sin(3 * cur_t)
            rh_y = -0.55 + 0.05 * np.cos(3 * cur_t)
            rh_z = -0.1
        elif class_idx == 1:  # Thank you: chin outward
            rh_x = 0.05
            rh_y = -0.35 + 0.25 * (i / num_frames)
            rh_z = -0.1 - 0.3 * (i / num_frames)
        elif class_idx == 2:  # Yes: nodding fist
            rh_x = 0.25
            rh_y = -0.1 + 0.15 * np.sin(4 * cur_t)
            rh_z = -0.15
        elif class_idx == 3:  # No: tapping
            rh_x = 0.20 + 0.03 * np.sin(5 * cur_t)
            rh_y = -0.25 + 0.03 * np.cos(5 * cur_t)
            rh_z = -0.10
        elif class_idx == 4:  # Help: supporting fist
            rh_x = 0.0 + 0.05 * (i / num_frames)
            rh_y = -0.15 - 0.2 * (i / num_frames)
            rh_z = -0.1
            # Left hand supports underneath
            seq[i, 75 : 75 + 63] = -0.05
        elif class_idx == 5:  # Emergency: rapid oscillation
            rh_x = 0.25 + 0.18 * np.sin(6 * cur_t)
            rh_y = -0.20 + 0.12 * np.cos(6 * cur_t)
            rh_z = -0.1
            seq[i, 75 : 75 + 63] = -rh_x  # Both hands move symmetrically
        elif class_idx == 6:  # Please: circular rub on chest
            rh_x = 0.0 + 0.10 * np.cos(2 * cur_t)
            rh_y = -0.15 + 0.10 * np.sin(2 * cur_t)
            rh_z = -0.05
        elif class_idx == 7:  # I Love You: forward motion
            rh_x = 0.25
            rh_y = -0.30
            rh_z = -0.1 - 0.35 * (i / num_frames)
        elif class_idx == 8:  # Water: W tapping chin
            rh_x = 0.05 + 0.04 * np.sin(4 * cur_t)
            rh_y = -0.40 + 0.03 * np.cos(4 * cur_t)
            rh_z = -0.08
        else:  # Goodbye: broad hand swing
            rh_x = 0.35 + 0.20 * np.sin(2 * cur_t)
            rh_y = -0.45 + 0.08 * np.cos(2 * cur_t)
            rh_z = -0.1

        # Populate right hand 21 landmarks around the center
        for joint in range(21):
            offset_x = (joint % 5 - 2) * 0.015 * signer_variation_factor
            offset_y = (joint // 5) * 0.015 * signer_variation_factor
            idx = 138 + joint * 3
            seq[i, idx] = rh_x + offset_x
            seq[i, idx + 1] = rh_y + offset_y
            seq[i, idx + 2] = rh_z

    # Add Gaussian jitter
    seq += np.random.normal(0, noise_level, size=seq.shape).astype(np.float32)
    return seq


def create_synthetic_dataset(
    num_classes: int = 10,
    samples_per_class: int = 150,
    num_signers: int = 6,
    seq_len: int = 30,
    feature_dim: int = 201,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    Creates a multi-subject dataset of sign language skeletal sequences.
    
    Returns:
        samples: list of (T, 201) arrays
        labels: list of class integer labels
        signer_ids: list of signer subject IDs (for strict signer-disjoint split)
    """
    samples = []
    labels = []
    signer_ids = []

    samples_per_signer = samples_per_class // num_signers
    for c in range(num_classes):
        for s in range(num_signers):
            signer_scale = 0.85 + (s * 0.06)  # Distinct physical body scale per signer
            for _ in range(samples_per_signer):
                # Varied temporal length before resampling
                raw_len = np.random.randint(24, 38)
                traj = generate_gesture_trajectory(
                    class_idx=c,
                    num_frames=raw_len,
                    feature_dim=feature_dim,
                    signer_variation_factor=signer_scale,
                    noise_level=0.008,
                )
                samples.append(traj)
                labels.append(c)
                signer_ids.append(s)

    return samples, labels, signer_ids
