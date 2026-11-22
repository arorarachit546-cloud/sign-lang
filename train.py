"""
Master Training Pipeline for Sign Language Recognition and Translation
Trains both the Spatiotemporal 1D-CNN Baseline and the Proposed BiLSTM-Attention Model
using strict subject-independent splits and reproducible protocols.
"""

import json
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.anatomical_hand_generator import create_anatomical_dataset
from src.data.dataset import SignDataAugmentor, SignLanguageDataset
from src.evaluation.visualizer import plot_learning_curves
from src.models.baseline_1dcnn import Spatiotemporal1DCNN
from src.models.bilstm_attention import BiLSTMTemporalAttention
from src.training.trainer import Trainer
from src.training.utils import LabelSmoothingCrossEntropy, set_seed


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config["project"].get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================================")
    print(f" PROJECT: {config['project']['name']} (v{config['project']['version']})")
    print(f" COMPUTE DEVICE: {device} | SEED: {seed}")
    print(f"==================================================================")

    # 1. Dataset Generation & Splitting
    classes = config["data"]["classes"]
    num_classes = len(classes)
    seq_len = config["data"]["sequence_length"]
    feature_dim = config["data"]["feature_dim"]

    print(f"[Data]: Creating anatomically grounded multi-subject dataset ({num_classes} classes, 6 signers)...")
    samples, labels, signer_ids = create_anatomical_dataset(
        num_classes=num_classes,
        samples_per_class=180,
        num_signers=6,
    )

    # Signer-Disjoint Split: Signers 0..3 -> Train, Signer 4 -> Val, Signer 5 -> Test
    train_indices = [i for i, s in enumerate(signer_ids) if s in [0, 1, 2, 3]]
    val_indices = [i for i, s in enumerate(signer_ids) if s == 4]
    test_indices = [i for i, s in enumerate(signer_ids) if s == 5]

    print(f"[Split]: Train samples: {len(train_indices)} (Signers 0-3)")
    print(f"[Split]: Val samples:   {len(val_indices)} (Signer 4)")
    print(f"[Split]: Test samples:  {len(test_indices)} (Signer 5) [Strictly Signer-Disjoint]")

    # Save test set for evaluate.py
    os.makedirs("data/processed", exist_ok=True)
    test_samples = [samples[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    np.save("data/processed/test_samples.npy", np.array(test_samples, dtype=object), allow_pickle=True)
    np.save("data/processed/test_labels.npy", np.array(test_labels))

    # Datasets & Loaders
    augmentor = SignDataAugmentor()
    train_dataset = SignLanguageDataset(
        samples=[samples[i] for i in train_indices],
        labels=[labels[i] for i in train_indices],
        target_seq_len=seq_len,
        augment=config["augmentation"]["enabled"],
        augmentor=augmentor,
    )
    val_dataset = SignLanguageDataset(
        samples=[samples[i] for i in val_indices],
        labels=[labels[i] for i in val_indices],
        target_seq_len=seq_len,
        augment=False,
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Directory Setup
    checkpoint_dir = config["training"]["checkpoint_dir"]
    results_dir = config["evaluation"]["results_dir"]
    figures_dir = config["evaluation"]["figures_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["early_stopping"]["patience"]

    # -------------------------------------------------------------------------
    # 2. Train Baseline: Spatiotemporal 1D-CNN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" 1. TRAINING BASELINE: Spatiotemporal 1D-CNN")
    print("=" * 50)
    baseline_model = Spatiotemporal1DCNN(
        in_channels=feature_dim,
        num_classes=num_classes,
        filter_sizes=tuple(config["models"]["baseline_1dcnn"]["num_filters"]),
        dropout=config["models"]["baseline_1dcnn"]["dropout"],
        dense_dim=config["models"]["baseline_1dcnn"]["dense_dim"],
    )
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=weight_decay)
    baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        baseline_optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    baseline_trainer = Trainer(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=baseline_optimizer,
        scheduler=baseline_scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_name="baseline_1dcnn",
    )
    history_baseline = baseline_trainer.fit(epochs=epochs, patience=patience)
    plot_learning_curves(
        history_baseline,
        save_path=os.path.join(figures_dir, "baseline_1dcnn_learning_curves.png"),
        title_prefix="Baseline 1D-CNN",
    )

    # -------------------------------------------------------------------------
    # 3. Train Proposed: BiLSTM with Temporal Attention
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" 2. TRAINING PROPOSED: BiLSTM with Temporal Attention")
    print("=" * 50)
    proposed_model = BiLSTMTemporalAttention(
        input_dim=feature_dim,
        spatial_embed_dim=config["models"]["bilstm_attention"]["spatial_embed_dim"],
        hidden_dim=config["models"]["bilstm_attention"]["hidden_dim"],
        num_layers=config["models"]["bilstm_attention"]["num_layers"],
        num_classes=num_classes,
        recurrent_dropout=config["models"]["bilstm_attention"]["recurrent_dropout"],
        dense_dim=config["models"]["bilstm_attention"]["dense_dim"],
        fc_dropout=config["models"]["bilstm_attention"]["fc_dropout"],
    )
    proposed_optimizer = torch.optim.AdamW(proposed_model.parameters(), lr=lr, weight_decay=weight_decay)
    proposed_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        proposed_optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    proposed_trainer = Trainer(
        model=proposed_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=proposed_optimizer,
        scheduler=proposed_scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_name="bilstm_attention",
    )
    history_proposed = proposed_trainer.fit(epochs=epochs, patience=patience)
    plot_learning_curves(
        history_proposed,
        save_path=os.path.join(figures_dir, "bilstm_attention_learning_curves.png"),
        title_prefix="Proposed BiLSTM-Attn",
    )

    # Persist History
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump({
            "baseline_1dcnn": history_baseline,
            "bilstm_attention": history_proposed,
        }, f, indent=2)

    print("\n[Complete]: Training finished successfully. Checkpoints and figures saved.")


if __name__ == "__main__":
    main()
