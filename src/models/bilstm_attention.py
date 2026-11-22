"""
Primary Architecture: Bidirectional LSTM with Temporal Attention (BiLSTM-Attn)
Translates 3D landmark sequence dynamics into gesture class representations.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import TemporalAttention


class BiLSTMTemporalAttention(nn.Module):
    """
    Proposed Architecture:
    1. Spatial Linear Projection: 201 -> 128 with LayerNorm & Dropout
    2. Recurrent Spatiotemporal Backbone: 2-layer Bidirectional LSTM (256 dims)
    3. Temporal Attention: Learns dynamic frame importance weights
    4. Dense Classification Head: 256 -> 128 -> NumClasses
    """

    def __init__(
        self,
        input_dim: int = 201,
        spatial_embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        recurrent_dropout: float = 0.25,
        dense_dim: int = 128,
        fc_dropout: float = 0.4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Spatial Projection Layer
        self.spatial_projection = nn.Sequential(
            nn.Linear(input_dim, spatial_embed_dim),
            nn.LayerNorm(spatial_embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # Bidirectional LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=spatial_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional hidden size = hidden_dim * 2
        bi_hidden_dim = hidden_dim * 2

        # Temporal Attention Layer
        self.attention = TemporalAttention(
            hidden_dim=bi_hidden_dim,
            attention_dim=128,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(bi_hidden_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.ReLU(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (Batch, Time, InputDim) = (B, 30, 201)
            return_attention: If True, returns attention weights tuple (logits, attn_weights)
        Returns:
            logits: Tensor of shape (Batch, NumClasses)
            attention_weights (optional): Tensor of shape (Batch, Time)
        """
        # Spatial Projection: (B, T, 201) -> (B, T, 128)
        proj = self.spatial_projection(x)

        # Recurrent Encoding: (B, T, 128) -> (B, T, 256)
        lstm_out, _ = self.lstm(proj)

        # Attention Aggregation: (B, T, 256) -> context (B, 256)
        context, attn_weights = self.attention(lstm_out)

        # Classification: (B, 256) -> (B, NumClasses)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits
