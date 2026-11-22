"""
Unit Tests: Forward Propagation, Attention Weights, and Gradient Backpropagation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from src.models.attention import TemporalAttention
from src.models.baseline_1dcnn import Spatiotemporal1DCNN
from src.models.bilstm_attention import BiLSTMTemporalAttention


def test_baseline_1dcnn_forward():
    model = Spatiotemporal1DCNN(in_channels=201, num_classes=10)
    x = torch.randn(4, 30, 201)  # (Batch, Time, Dims)
    out = model(x)
    assert out.shape == (4, 10), f"Expected output shape (4, 10), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in 1D-CNN output"


def test_temporal_attention():
    attn = TemporalAttention(hidden_dim=256, attention_dim=128)
    hidden_states = torch.randn(4, 30, 256)
    context, weights = attn(hidden_states)

    assert context.shape == (4, 256), f"Expected context shape (4, 256), got {context.shape}"
    assert weights.shape == (4, 30), f"Expected weights shape (4, 30), got {weights.shape}"

    # Verify attention weights sum to 1.0 across temporal axis
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Attention weights do not sum to 1.0"


def test_bilstm_attention_forward_and_backward():
    model = BiLSTMTemporalAttention(
        input_dim=201,
        spatial_embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=10,
    )
    x = torch.randn(4, 30, 201)
    targets = torch.tensor([0, 3, 7, 9], dtype=torch.long)

    # Test forward with attention
    logits, attn_weights = model(x, return_attention=True)
    assert logits.shape == (4, 10), f"Expected logits shape (4, 10), got {logits.shape}"
    assert attn_weights.shape == (4, 30), f"Expected attn shape (4, 30), got {attn_weights.shape}"

    # Test backward gradient propagation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    # Verify gradients computed for all trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


if __name__ == "__main__":
    test_baseline_1dcnn_forward()
    test_temporal_attention()
    test_bilstm_attention_forward_and_backward()
    print("All Model Forward & Gradient Propagation Tests Passed Successfully!")
