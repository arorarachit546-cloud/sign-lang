"""
Model Training Engine with Early Stopping and Cosine Annealing Learning Rate
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.utils import LabelSmoothingCrossEntropy, save_checkpoint


class Trainer:
    """
    Standardized trainer supporting:
    - Label smoothing regularization
    - AdamW optimizer with weight decay
    - CosineAnnealingWarmRestarts scheduling
    - Gradient norm clipping to avoid exploding gradients
    - Validation loss monitoring & early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or LabelSmoothingCrossEntropy(smoothing=0.1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            loss.backward()

            # Gradient Clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def fit(
        self,
        epochs: int = 60,
        patience: int = 10,
        min_delta: float = 0.001,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        best_val_loss = float("inf")
        patience_counter = 0
        best_checkpoint_path = ""

        if verbose:
            print(f"--- Starting training [{self.model_name}] on {self.device} for {epochs} epochs ---")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

            # Checkpoint on best validation loss
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                patience_counter = 0
                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "model_name": self.model_name,
                }
                best_checkpoint_path = save_checkpoint(
                    state,
                    self.checkpoint_dir,
                    f"{self.model_name}_best.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                    break

        if verbose:
            print(f"Training completed for {self.model_name}. Checkpoint: {best_checkpoint_path}")
        return self.history
