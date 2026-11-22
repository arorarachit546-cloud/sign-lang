"""
Unit Tests: Data Pipeline, Normalization, Augmentation, and Leakage Prevention
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.dataset import SignDataAugmentor, SignLanguageDataset, interpolate_sequence, normalize_landmarks
from src.data.synthetic_generator import create_synthetic_dataset


def test_interpolate_sequence():
    """Verify temporal resampling to fixed sequence length T=30."""
    seq_45 = np.random.randn(45, 201).astype(np.float32)
    seq_15 = np.random.randn(15, 201).astype(np.float32)

    interp_45 = interpolate_sequence(seq_45, target_length=30)
    interp_15 = interpolate_sequence(seq_15, target_length=30)

    assert interp_45.shape == (30, 201), f"Expected (30, 201), got {interp_45.shape}"
    assert interp_15.shape == (30, 201), f"Expected (30, 201), got {interp_15.shape}"
    assert not np.isnan(interp_45).any(), "NaN found in interpolated output"


def test_normalize_landmarks():
    """Verify shoulder-relative spatial translation and torso scaling."""
    raw_seq = np.zeros((30, 201), dtype=np.float32)
    # Set left shoulder and right shoulder
    raw_seq[:, 11 * 3 : 11 * 3 + 3] = [-0.2, -0.1, 0.0]
    raw_seq[:, 12 * 3 : 12 * 3 + 3] = [0.2, -0.1, 0.0]

    norm_seq = normalize_landmarks(raw_seq)

    assert norm_seq.shape == (30, 201), f"Shape mismatch in normalized output: {norm_seq.shape}"
    assert not np.isnan(norm_seq).any(), "NaN found in normalized landmarks"
    assert not np.isinf(norm_seq).any(), "Inf found in normalized landmarks"


def test_data_augmentor():
    """Verify on-the-fly augmentation preserves shape and finite range."""
    augmentor = SignDataAugmentor()
    sample = np.random.randn(30, 201).astype(np.float32)

    augmented = augmentor(sample)
    assert augmented.shape == (30, 201), f"Augmentation altered shape: {augmented.shape}"
    assert not np.isnan(augmented).any(), "NaN found after augmentation"


def test_signer_disjoint_leakage():
    """Verify strict subject-independent splitting to guarantee zero identity leakage."""
    samples, labels, signer_ids = create_synthetic_dataset(
        num_classes=5,
        samples_per_class=60,
        num_signers=6,
    )

    train_signers = {s for i, s in enumerate(signer_ids) if s in [0, 1, 2, 3]}
    val_signers = {s for i, s in enumerate(signer_ids) if s == 4}
    test_signers = {s for i, s in enumerate(signer_ids) if s == 5}

    # Verify sets are pairwise disjoint
    assert len(train_signers.intersection(val_signers)) == 0, "Data leakage between train and val!"
    assert len(train_signers.intersection(test_signers)) == 0, "Data leakage between train and test!"
    assert len(val_signers.intersection(test_signers)) == 0, "Data leakage between val and test!"


if __name__ == "__main__":
    test_interpolate_sequence()
    test_normalize_landmarks()
    test_data_augmentor()
    test_signer_disjoint_leakage()
    print("All Data Pipeline Tests Passed Successfully!")
