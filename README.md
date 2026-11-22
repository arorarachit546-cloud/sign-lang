# Sign Language Recognition and Translation Assistant
### Production-Grade Deep Learning Framework for Assistive Communication

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2B-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00C49F.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

The **Sign Language Recognition and Translation Assistant** is an end-to-end deep learning system designed to bridge communication barriers between Deaf and Hard-of-Hearing (DHH) signers and non-signers. By combining **Google MediaPipe Holistic 3D skeletal tracking**, a **Bidirectional LSTM with Additive Temporal Attention**, consensus-based **temporal debouncing**, and **Text-to-Speech (TTS)** synthesis, the application provides real-time, low-latency visual-to-auditory translation of continuous sign gestures.

### Key Highlights
- **Decoupled Kinematic Representation**: Extracts 75 anatomical landmarks ($201$ 3D coordinates per frame), achieving a **187x compression ratio** over raw RGB video and eliminating bias from skin tone, lighting, and backgrounds.
- **Invariant Affine Normalization**: Centered relative to the mid-shoulder anatomical origin and scale-normalized by torso width to guarantee translation and scale invariance.
- **Architectural Benchmark**: Directly contrasts a **Spatiotemporal 1D-CNN Baseline** against a **Bidirectional LSTM with Temporal Attention**.
- **Signer-Disjoint Generalization**: Evaluated on strict subject-independent splits where testing signers never appear during training, preventing identity leakage.
- **Assistive Real-Time Interface**: Live OpenCV HUD display with predicted sign confidence bars, translation history transcripts, and threaded non-blocking TTS audio feedback.

---

## 2. Directory Structure

```
SignLanguage/
├── config/
│   └── config.yaml               # Centralized configuration (hyperparameters, paths, seeds)
├── data/
│   ├── raw/                      # Raw landmark recordings / videos
│   └── processed/                # Preprocessed and normalized sequences (test splits)
├── src/
│   ├── extraction/
│   │   └── holistic_extractor.py # MediaPipe Holistic 3D skeletal extractor
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset, spatial normalizer, and augmentations
│   │   └── synthetic_generator.py# Realistic multi-subject kinematic trajectory generator
│   ├── models/
│   │   ├── baseline_1dcnn.py     # Spatiotemporal 1D-CNN baseline model
│   │   ├── attention.py          # Additive Temporal Self-Attention layer
│   │   └── bilstm_attention.py   # Proposed BiLSTM + Attention architecture
│   ├── training/
│   │   ├── trainer.py            # Training loop with validation and early stopping
│   │   └── utils.py              # Seed locking, checkpointing, and label-smoothing loss
│   ├── evaluation/
│   │   ├── metrics.py            # Top-1/Top-5 accuracy, Macro F1, and profiling
│   │   └── visualizer.py         # Publication plots (loss curves, confusion matrix, attention)
│   └── assistant/
│       ├── buffer_manager.py     # 30-frame sliding window buffer and debouncing logic
│       ├── tts_engine.py         # Multi-threaded non-blocking speech synthesis
│       └── real_time_app.py      # OpenCV interactive HUD assistant
├── docs/
│   ├── MATHEMATICAL_FORMULATION.md # Full mathematical derivations and proofs
│   ├── ARCHITECTURE.md           # Model topology, tensor shapes, and design rationale
│   └── VIVA_QUESTIONS.md         # 25+ comprehensive defense questions and answers
├── tests/
│   ├── test_data_pipeline.py     # Unit tests for normalization, interpolation, leakage
│   └── test_model_forward.py     # Unit tests for tensor shapes, attention, and gradients
├── requirements.txt              # Pinned environment dependencies
├── train.py                      # Master training pipeline entry point
├── evaluate.py                   # Evaluation, benchmarking, and visualization entry point
└── run_assistant.py              # Real-time assistant application launcher
```

---

## 3. Quick Start Guide

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Automated Verification (Unit Tests)
Verify that data transformations, model dimensions, and gradient backpropagation are mathematically consistent:
```bash
python tests/test_data_pipeline.py
python tests/test_model_forward.py
```

### Step 3: Train Both Models
Trains both the **Baseline 1D-CNN** and **Proposed BiLSTM-Attn** models using early stopping, label-smoothed cross-entropy, and cosine annealing:
```bash
python train.py
```
Checkpoints are automatically saved to `checkpoints/` and training curves to `results/figures/`.

### Step 4: Run Comprehensive Evaluation & Benchmarking
Evaluates checkpoints on the unseen test signer, generating confusion matrices, latency benchmarks, and attention plots:
```bash
python evaluate.py
```

### Step 5: Launch Real-Time Translation Assistant
Launch the interactive live camera assistant with auditory speech feedback:
```bash
# Live webcam mode:
python run_assistant.py --camera 0

# Automated simulated verification mode:
python run_assistant.py --test-mode
```

---

## 4. Hyperparameter Specifications

| Parameter | Configuration Value | Justification |
| :--- | :--- | :--- |
| **Window Length ($T$)** | $30\text{ frames}$ ($\approx 1.0\text{s}$) | Matches natural temporal duration of isolated gestures |
| **Input Dimension ($D$)** | $201\text{ coords}$ ($67 \times 3$) | $25\text{ pose} + 21\text{ left hand} + 21\text{ right hand}$ |
| **Batch Size** | $32$ | Balances gradient variance with cache efficiency |
| **Optimizer** | $\text{AdamW}$ ($\lambda = 10^{-4}$) | Decoupled weight decay prevents overfitting on landmark arrays |
| **Learning Rate** | $10^{-3} \to 10^{-5}$ | Cosine Annealing with Warm Restarts ($T_0=10, T_{\text{mult}}=2$) |
| **Loss Function** | Label Smoothing ($\epsilon=0.1$) | Mitigates overconfident predictions on visually similar signs |
| **Gradient Clipping** | $\|\mathbf{g}\|_2 \le 1.0$ | Prevents exploding gradients during BPTT |
| **Consensus Threshold** | $K = 5\text{ frames}, \tau = 0.75$ | Suppresses transition flicker and false positives |

---

## 5. Comprehensive Documentation
- 📘 [Mathematical Formulations & Derivations](docs/MATHEMATICAL_FORMULATION.md)
- 📐 [Architectural Topology & Specifications](docs/ARCHITECTURE.md)
- 🎓 [Viva Defense & Examination Q&A Guide](docs/VIVA_QUESTIONS.md)
