"""
Baseline Model: Spatiotemporal 1D-CNN Network
Extracts local temporal multi-scale kinematic features across landmark trajectories.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Spatiotemporal1DCNN(nn.Module):
    """
    Baseline 1D Convolutional Neural Network for Sign Language Recognition.
    Takes temporal sequence of landmarks (B, T, D) where T=30, D=201.
    Transposes to (B, D, T) for 1D temporal convolutions.
    """

    def __init__(
        self,
        in_channels: int = 201,
        num_classes: int = 10,
        filter_sizes: Tuple[int, int, int] = (64, 128, 256),
        dropout: float = 0.3,
        dense_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filter_sizes[0],
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(filter_sizes[0])

        # Block 2
        self.conv2 = nn.Conv1d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(filter_sizes[1])

        # Block 3
        self.conv3 = nn.Conv1d(
            in_channels=filter_sizes[1],
            out_channels=filter_sizes[2],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(filter_sizes[2])

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.fc1 = nn.Linear(filter_sizes[2], dense_dim)
        self.fc_bn = nn.BatchNorm1d(dense_dim)
        self.classifier = nn.Linear(dense_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, Channels) = (B, 30, 201)
        Returns:
            logits: Tensor of shape (Batch, NumClasses)
        """
        # Permute to (B, Channels, Time)
        out = x.transpose(1, 2)

        # Block 1
        out = F.relu(self.bn1(self.conv1(out)))

        # Block 2
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout(out)

        # Block 3
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.dropout(out)

        # Global average pooling across temporal dimension
        out = self.global_pool(out)  # Shape: (B, 256, 1)
        out = out.squeeze(-1)         # Shape: (B, 256)

        # Dense head
        out = F.relu(self.fc_bn(self.fc1(out)))
        out = self.dropout(out)
        logits = self.classifier(out)

        return logits
