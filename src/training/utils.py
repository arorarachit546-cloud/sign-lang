"""
Training Utilities: Reproducibility, Checkpointing, and Loss Functions
"""

import os
import random
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    """Enforces strict deterministic behavior across random generators and GPU backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Categorical Cross-Entropy with Label Smoothing regularization:
        q(k) = (1 - epsilon) * delta(k, y) + epsilon / C
        Loss = -sum_k q(k) * log(p_k)
    Prevents overconfidence on visually adjacent sign gestures.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Negative log likelihood of true label
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        
        # Smooth loss across all classes
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "best_model.pt",
) -> str:
    """Persists model parameters, optimizer state, and training metadata."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    return filepath


def load_checkpoint(filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Loads weights and state from saved checkpoint."""
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
