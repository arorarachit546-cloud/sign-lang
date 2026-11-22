"""
Sign Language Landmark Dataset and Preprocessing Pipeline
Includes spatial normalization (shoulder-centered), temporal interpolation (T=30),
on-the-fly spatiotemporal data augmentation, and PyTorch Dataset bindings.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_landmarks(sequence: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Spatially normalizes a landmark sequence of shape (T, 201).
    Pose landmarks occupy indices 0 to 74 (25 landmarks * 3 coords).
    Landmark 11 = Left Shoulder (indices 33:36)
    Landmark 12 = Right Shoulder (indices 36:39)

    Translations are centered at the midpoint of the shoulders when shoulders are present,
    or at the wrist of the active hand when operating in hand-only webcam view.
    """
    seq_float = np.asarray(sequence, dtype=np.float32)
    T = seq_float.shape[0]
    coords = seq_float.reshape(T, -1, 3).copy()  # Shape: (T, 67, 3) where 25 pose + 21 LH + 21 RH = 67

    left_shoulder = coords[:, 11, :]   # (T, 3)
    right_shoulder = coords[:, 12, :]  # (T, 3)

    # Calculate shoulder midpoint & distance
    mid_shoulder = 0.5 * (left_shoulder + right_shoulder)  # (T, 3)
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)  # (T, 1)

    valid_mask = shoulder_dist > 0.05
    if np.any(valid_mask):
        scale = max(float(np.mean(shoulder_dist[valid_mask])), eps)
        for t in range(T):
            coords[t] = (coords[t] - mid_shoulder[t]) / scale
    else:
        # Hand-centric normalization (essential for close-up webcam framing)
        rh_active = np.any(np.abs(coords[:, 46:67, :]) > eps)
        lh_active = np.any(np.abs(coords[:, 25:46, :]) > eps)
        wrist_idx = 46 if rh_active else (25 if lh_active else 0)
        mcp_idx = 55 if rh_active else (34 if lh_active else 9)

        for t in range(T):
            wrist_pos = coords[t, wrist_idx, :].copy()
            mcp_pos = coords[t, mcp_idx, :].copy()
            palm_len = float(np.linalg.norm(mcp_pos - wrist_pos))
            hand_scale = palm_len if palm_len > 0.02 else 0.2
            coords[t] = (coords[t] - wrist_pos) / hand_scale

    return coords.reshape(T, -1)


def interpolate_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """
    Uniformly resamples a temporal landmark sequence of shape (T_orig, D)
    to target_length (default T=30) using 1D linear interpolation across time.
    """
    orig_length, feature_dim = sequence.shape
    if orig_length == target_length:
        return sequence.astype(np.float32)

    orig_indices = np.linspace(0, orig_length - 1, num=orig_length)
    target_indices = np.linspace(0, orig_length - 1, num=target_length)

    interpolated = np.zeros((target_length, feature_dim), dtype=np.float32)
    for d in range(feature_dim):
        interpolated[:, d] = np.interp(target_indices, orig_indices, sequence[:, d])

    return interpolated


class SignDataAugmentor:
    """
    Performs on-the-fly spatiotemporal data augmentations:
    - Spatial scaling: U(0.85, 1.15)
    - Spatial rotation: U(-15 deg, +15 deg) around z-axis
    - Temporal warping: resampling factor U(0.8, 1.2)
    - Gaussian noise injection: N(0, 0.005^2)
    - Keypoint dropout: 5% random occlusion
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        temporal_warp_range: Tuple[float, float] = (0.8, 1.2),
        gaussian_noise_std: float = 0.005,
        keypoint_dropout_rate: float = 0.05,
    ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.temporal_warp_range = temporal_warp_range
        self.gaussian_noise_std = gaussian_noise_std
        self.keypoint_dropout_rate = keypoint_dropout_rate

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """
        Augment sequence of shape (T, D) where D = num_points * 3.
        """
        T, D = sequence.shape
        num_points = D // 3
        coords = sequence.reshape(T, num_points, 3).copy()

        # 1. Random Spatial Scaling
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        coords *= scale

        # 2. Random 2D In-Plane Rotation (x, y coordinates)
        angle_deg = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
        theta = math.radians(angle_deg)
        rot_matrix = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ], dtype=np.float32)

        # Apply rotation to (x, y)
        xy = coords[:, :, :2].reshape(-1, 2)
        rotated_xy = np.dot(xy, rot_matrix.T)
        coords[:, :, :2] = rotated_xy.reshape(T, num_points, 2)

        # 3. Gaussian Noise Injection
        if self.gaussian_noise_std > 0:
            noise = np.random.normal(0, self.gaussian_noise_std, size=coords.shape).astype(np.float32)
            coords += noise

        # 4. Keypoint Dropout (random joint occlusion)
        if self.keypoint_dropout_rate > 0:
            dropout_mask = (np.random.rand(1, num_points, 1) > self.keypoint_dropout_rate).astype(np.float32)
            coords *= dropout_mask

        # 5. Temporal Warping
        warp_factor = np.random.uniform(self.temporal_warp_range[0], self.temporal_warp_range[1])
        new_len = max(5, int(T * warp_factor))
        flattened = coords.reshape(T, D)
        warped = interpolate_sequence(flattened, target_length=new_len)
        result = interpolate_sequence(warped, target_length=T)

        return result.astype(np.float32)


class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset for Sign Language Skeletal Coordinate Sequences.
    """

    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        signer_ids: Optional[List[int]] = None,
        target_seq_len: int = 30,
        augment: bool = False,
        augmentor: Optional[SignDataAugmentor] = None,
        normalize: bool = True,
    ):
        assert len(samples) == len(labels), "Samples and labels must have identical count."
        self.samples = samples
        self.labels = labels
        self.signer_ids = signer_ids
        self.target_seq_len = target_seq_len
        self.augment = augment
        self.augmentor = augmentor if augmentor is not None else SignDataAugmentor()
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx].copy()  # Shape: (T_orig, 201)

        # Temporal interpolation to fixed length T=30
        if seq.shape[0] != self.target_seq_len:
            seq = interpolate_sequence(seq, target_length=self.target_seq_len)

        # Spatial normalization
        if self.normalize:
            seq = normalize_landmarks(seq)

        # On-the-fly Data Augmentation
        if self.augment:
            seq = self.augmentor(seq)

        tensor_x = torch.from_numpy(seq).float()  # (T, D)
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_x, tensor_y
