"""
Evaluation Metrics and Computational Profiling Module
Computes Top-1/Top-5 accuracy, Macro F1, Confusion Matrix, and Latency/FLOPs.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics:
    - Top-1 Accuracy
    - Top-5 Accuracy (if probabilities provided)
    - Macro Precision, Recall, F1
    - Per-class metrics
    - Confusion Matrix
    """
    top1_acc = float(np.mean(y_true == y_pred))

    top5_acc = None
    if y_probs is not None and y_probs.shape[1] >= 5:
        top5_indices = np.argsort(y_probs, axis=1)[:, -5:]
        top5_correct = [y_true[i] in top5_indices[i] for i in range(len(y_true))]
        top5_acc = float(np.mean(top5_correct))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    conf_mat = confusion_matrix(y_true, y_pred)
    # Normalized confusion matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_conf_mat = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)
        norm_conf_mat = np.nan_to_num(norm_conf_mat)

    per_class_summary = {}
    if class_names is not None:
        for idx, name in enumerate(class_names):
            if idx < len(prec_per_class):
                per_class_summary[name] = {
                    "precision": float(prec_per_class[idx]),
                    "recall": float(rec_per_class[idx]),
                    "f1_score": float(f1_per_class[idx]),
                    "support": int(support[idx]),
                }

    metrics = {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "per_class": per_class_summary,
        "confusion_matrix": conf_mat.tolist(),
        "normalized_confusion_matrix": norm_conf_mat.tolist(),
    }
    return metrics


def profile_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 30, 201),
    device: Optional[torch.device] = None,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Profiles model parameters, inference latency (ms), and FPS.
    """
    dev = device or next(model.parameters()).device
    model.eval()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dummy input
    dummy_input = torch.randn(*input_shape).to(dev)

    # Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_input)

    # Latency timing
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1),
    }
