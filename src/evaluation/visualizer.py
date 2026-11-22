"""
Visualization Module for Experimental Rigor and Viva Presentation
Generates loss curves, confusion matrix heatmaps, attention weight distributions,
and baseline comparison plots.
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title_prefix: str = "Model",
) -> None:
    """Plots side-by-side training and validation loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r--", label="Val Loss", linewidth=2)
    axes[0].set_title(f"{title_prefix}: Loss vs. Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Label Smoothed Cross-Entropy Loss", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Accuracy curve
    train_acc = [acc * 100 for acc in history["train_acc"]]
    val_acc = [acc * 100 for acc in history["val_acc"]]
    axes[1].plot(epochs, train_acc, "b-", label="Train Acc (%)", linewidth=2)
    axes[1].plot(epochs, val_acc, "g--", label="Val Acc (%)", linewidth=2)
    axes[1].set_title(f"{title_prefix}: Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Normalized Confusion Matrix",
) -> None:
    """Renders normalized confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 7))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Sign Class", fontsize=11)
    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_attention_weights(
    attention_weights: np.ndarray,
    class_name: str,
    save_path: str,
) -> None:
    """
    Plots dynamic attention distribution across T=30 frames for a sample gesture.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    frames = np.arange(1, len(attention_weights) + 1)

    plt.bar(frames, attention_weights, color="#2b5c8f", width=0.7, edgecolor="black", alpha=0.85)
    plt.plot(frames, attention_weights, color="#d9534f", marker="o", linewidth=1.5)

    plt.title(f"Temporal Attention Weight Distribution: '{class_name}'", fontsize=12, fontweight="bold")
    plt.xlabel("Temporal Frame Index (t = 1 .. 30)", fontsize=10)
    plt.ylabel("Attention Coefficient alpha_t", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(
    baseline_metrics: Dict[str, float],
    proposed_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Generates comparative bar chart comparing Baseline 1D-CNN vs BiLSTM-Attention."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics = ["Top-1 Acc (%)", "Macro F1 (%)", "Latency (ms)"]
    
    baseline_vals = [
        baseline_metrics.get("top1_accuracy", 0.0) * 100,
        baseline_metrics.get("macro_f1", 0.0) * 100,
        baseline_metrics.get("mean_latency_ms", 0.0),
    ]
    proposed_vals = [
        proposed_metrics.get("top1_accuracy", 0.0) * 100,
        proposed_metrics.get("macro_f1", 0.0) * 100,
        proposed_metrics.get("mean_latency_ms", 0.0),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (1D-CNN)", color="#708090")
    rects2 = ax.bar(x + width/2, proposed_vals, width, label="Proposed (BiLSTM-Attn)", color="#1E88E5")

    ax.set_ylabel("Scores / Metric Value")
    ax.set_title("Architectural Comparison: Baseline vs. Proposed BiLSTM-Attention", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Label with values
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
