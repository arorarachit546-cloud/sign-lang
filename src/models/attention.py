"""
Temporal Attention Mechanism Module
Implements Additive (Bahdanau-style) Temporal Self-Attention over recurrent hidden states.
Computes dynamic frame-wise importance weights and produces a context-aware aggregated representation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Computes temporal attention weights across sequence time steps:
        e_t = v_a^T * tanh(W_a * h_t + b_a)
        alpha_t = softmax(e_t)
        c = sum(alpha_t * h_t)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Tensor of shape (Batch, Time, HiddenDim) = (B, T, 2H)
        Returns:
            context_vector: Aggregated representation of shape (Batch, HiddenDim)
            attention_weights: Normalized attention weights of shape (Batch, Time)
        """
        # Score computation: (B, T, HiddenDim) -> (B, T, AttentionDim)
        energy = torch.tanh(self.projection(hidden_states))

        # (B, T, AttentionDim) -> (B, T, 1) -> (B, T)
        scores = self.context_vector(energy).squeeze(-1)

        # Softmax over temporal dimension
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (B, T)

        # Weighted sum: (B, 1, T) @ (B, T, HiddenDim) -> (B, 1, HiddenDim) -> (B, HiddenDim)
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)

        return context, attention_weights
