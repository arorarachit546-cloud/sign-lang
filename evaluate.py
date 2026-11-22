"""
Comprehensive Evaluation and Benchmarking Suite
Evaluates both trained models on unseen test subjects, generates confusion matrices,
computes efficiency profiling (parameters, latency, FPS), extracts attention weights,
and generates comparative benchmark plots.
"""

import json
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SignLanguageDataset
from src.evaluation.metrics import compute_classification_metrics, profile_model_efficiency
from src.evaluation.visualizer import (
    plot_attention_weights,
    plot_confusion_matrix,
    plot_model_comparison,
)
from src.models.baseline_1dcnn import Spatiotemporal1DCNN
from src.models.bilstm_attention import BiLSTMTemporalAttention
from src.training.utils import load_checkpoint, set_seed


def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs)

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["project"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config["data"]["classes"]
    num_classes = len(classes)
    feature_dim = config["data"]["feature_dim"]
    seq_len = config["data"]["sequence_length"]

    checkpoint_dir = config["training"]["checkpoint_dir"]
    results_dir = config["evaluation"]["results_dir"]
    figures_dir = config["evaluation"]["figures_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"==================================================================")
    print(f" EVALUATION & BENCHMARKING ON UNSEEN TEST SUBJECTS")
    print(f" DEVICE: {device} | CLASSES: {num_classes}")
    print(f"==================================================================")

    # 1. Load Test Data
    test_samples_path = "data/processed/test_samples.npy"
    test_labels_path = "data/processed/test_labels.npy"
    if not os.path.exists(test_samples_path) or not os.path.exists(test_labels_path):
        raise FileNotFoundError("Processed test data not found. Please run train.py first.")

    raw_test_samples = np.load(test_samples_path, allow_pickle=True)
    test_samples = [s for s in raw_test_samples]
    test_labels = list(np.load(test_labels_path))

    test_dataset = SignLanguageDataset(
        samples=test_samples,
        labels=test_labels,
        target_seq_len=seq_len,
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"[Data]: Loaded {len(test_dataset)} test samples (Signer 5).")

    # -------------------------------------------------------------------------
    # 2. Evaluate Baseline 1D-CNN
    # -------------------------------------------------------------------------
    print("\n[Evaluation]: Assessing Baseline 1D-CNN...")
    baseline_model = Spatiotemporal1DCNN(
        in_channels=feature_dim,
        num_classes=num_classes,
        filter_sizes=tuple(config["models"]["baseline_1dcnn"]["num_filters"]),
        dropout=config["models"]["baseline_1dcnn"]["dropout"],
        dense_dim=config["models"]["baseline_1dcnn"]["dense_dim"],
    ).to(device)

    load_checkpoint(os.path.join(checkpoint_dir, "baseline_1dcnn_best.pt"), baseline_model)
    y_true, y_pred, y_probs = evaluate_model(baseline_model, test_loader, device, num_classes)
    baseline_metrics = compute_classification_metrics(y_true, y_pred, y_probs, class_names=classes)
    baseline_profile = profile_model_efficiency(baseline_model, input_shape=(1, seq_len, feature_dim), device=device)
    baseline_metrics.update(baseline_profile)

    plot_confusion_matrix(
        np.array(baseline_metrics["normalized_confusion_matrix"]),
        class_names=classes,
        save_path=os.path.join(figures_dir, "baseline_confusion_matrix.png"),
        title="Baseline (1D-CNN) Normalized Confusion Matrix",
    )

    # -------------------------------------------------------------------------
    # 3. Evaluate Proposed BiLSTM-Attention
    # -------------------------------------------------------------------------
    print("\n[Evaluation]: Assessing Proposed BiLSTM with Temporal Attention...")
    proposed_model = BiLSTMTemporalAttention(
        input_dim=feature_dim,
        spatial_embed_dim=config["models"]["bilstm_attention"]["spatial_embed_dim"],
        hidden_dim=config["models"]["bilstm_attention"]["hidden_dim"],
        num_layers=config["models"]["bilstm_attention"]["num_layers"],
        num_classes=num_classes,
        recurrent_dropout=config["models"]["bilstm_attention"]["recurrent_dropout"],
        dense_dim=config["models"]["bilstm_attention"]["dense_dim"],
        fc_dropout=config["models"]["bilstm_attention"]["fc_dropout"],
    ).to(device)

    load_checkpoint(os.path.join(checkpoint_dir, "bilstm_attention_best.pt"), proposed_model)
    y_true_p, y_pred_p, y_probs_p = evaluate_model(proposed_model, test_loader, device, num_classes)
    proposed_metrics = compute_classification_metrics(y_true_p, y_pred_p, y_probs_p, class_names=classes)
    proposed_profile = profile_model_efficiency(proposed_model, input_shape=(1, seq_len, feature_dim), device=device)
    proposed_metrics.update(proposed_profile)

    plot_confusion_matrix(
        np.array(proposed_metrics["normalized_confusion_matrix"]),
        class_names=classes,
        save_path=os.path.join(figures_dir, "bilstm_confusion_matrix.png"),
        title="Proposed (BiLSTM-Attn) Normalized Confusion Matrix",
    )

    # -------------------------------------------------------------------------
    # 4. Extract Sample Attention Weights for Explainability
    # -------------------------------------------------------------------------
    sample_x, sample_y = test_dataset[0]
    sample_tensor = sample_x.unsqueeze(0).to(device)
    proposed_model.eval()
    with torch.no_grad():
        _, sample_attn = proposed_model(sample_tensor, return_attention=True)
        attn_np = sample_attn.squeeze(0).cpu().numpy()

    plot_attention_weights(
        attn_np,
        class_name=classes[sample_y.item()],
        save_path=os.path.join(figures_dir, "temporal_attention_weights.png"),
    )

    # -------------------------------------------------------------------------
    # 5. Comparative Visualizations and Output Table
    # -------------------------------------------------------------------------
    plot_model_comparison(
        baseline_metrics=baseline_metrics,
        proposed_metrics=proposed_metrics,
        save_path=os.path.join(figures_dir, "model_comparison_benchmark.png"),
    )

    all_results = {
        "baseline_1dcnn": baseline_metrics,
        "bilstm_attention": proposed_metrics,
    }
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print Formatted Markdown Comparison Table
    print("\n" + "=" * 70)
    print(" SUMMARY BENCHMARK COMPARISON TABLE (UNSEEN TEST SIGNER)")
    print("=" * 70)
    print(f"| Metric / Characteristic         | Baseline (1D-CNN) | Proposed (BiLSTM-Attn) |")
    print(f"| :------------------------------ | :---------------- | :--------------------- |")
    print(f"| Top-1 Test Accuracy             | {baseline_metrics['top1_accuracy']*100:6.2f}%          | {proposed_metrics['top1_accuracy']*100:6.2f}%               |")
    print(f"| Top-5 Test Accuracy             | {baseline_metrics['top5_accuracy']*100:6.2f}%          | {proposed_metrics['top5_accuracy']*100:6.2f}%               |")
    print(f"| Macro Precision                 | {baseline_metrics['macro_precision']*100:6.2f}%          | {proposed_metrics['macro_precision']*100:6.2f}%               |")
    print(f"| Macro Recall                    | {baseline_metrics['macro_recall']*100:6.2f}%          | {proposed_metrics['macro_recall']*100:6.2f}%               |")
    print(f"| Macro F1-Score                  | {baseline_metrics['macro_f1']*100:6.2f}%          | {proposed_metrics['macro_f1']*100:6.2f}%               |")
    print(f"| Total Parameters                | {baseline_metrics['total_parameters']:,}            | {proposed_metrics['total_parameters']:,}                 |")
    print(f"| Model Size                      | {baseline_metrics['model_size_mb']:.2f} MB            | {proposed_metrics['model_size_mb']:.2f} MB                 |")
    print(f"| Mean Latency (per window)       | {baseline_metrics['mean_latency_ms']:5.2f} ms          | {proposed_metrics['mean_latency_ms']:5.2f} ms               |")
    print(f"| 95th Percentile Latency         | {baseline_metrics['p95_latency_ms']:5.2f} ms          | {proposed_metrics['p95_latency_ms']:5.2f} ms               |")
    print(f"| Throughput (FPS)                | {baseline_metrics['throughput_fps']:5.1f} FPS         | {proposed_metrics['throughput_fps']:5.1f} FPS              |")
    print("=" * 70)
    print(f"Artifacts and figures successfully saved to '{figures_dir}/'.")


if __name__ == "__main__":
    main()
