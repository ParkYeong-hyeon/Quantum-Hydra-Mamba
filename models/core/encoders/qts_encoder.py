"""
QTS Feature Encoder - Unified Classical Feature Extraction

This module provides the IDENTICAL encoder used in QTSTransformer (QuixerTSModel),
ensuring controlled comparison across all ablation models.

The encoder is CLASSICAL - all quantum computation happens in the
mixing layer (SSM, Attention, or Gated).

Components:
- Conv2dFeatureExtractor: 2D CNN for spatio-temporal patterns
- GatedFeedForward: GLU-based non-linear transformation
- Conv2dGLUPreprocessor: Combined Conv2d + GLU (RECOMMENDED)
- QTSFeatureEncoder: Main encoder class with multiple projection options

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Optional


class Conv2dFeatureExtractor(nn.Module):
    """
    2D Convolutional feature extractor from QTSTransformer.

    Treats (feature_dim, n_timesteps) as a single-channel "image"
    and extracts spatio-temporal patterns using 2D convolutions.

    Architecture:
        Input: (batch, 1, feature_dim, n_timesteps)
        Conv2d(1→16) + ReLU + MaxPool(4×4)
        Conv2d(16→32) + ReLU + MaxPool(4×4)
        Flatten → Linear → Output: (batch, n_timesteps, n_output)
    """

    def __init__(self, feature_dim: int, n_timesteps: int, n_output: int):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_output = n_output

        # 2D Convolutional Network (identical to QTSTransformer)
        self.conv_net = nn.Sequential(
            # Layer 1: 1 → 16 channels
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),

            # Layer 2: 16 → 32 channels
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, feature_dim, n_timesteps)
            dummy_output = self.conv_net(dummy_input)
            self.flattened_size = dummy_output.numel()

        # Final projection to quantum parameter dimensions
        self.final_linear = nn.Linear(self.flattened_size, n_timesteps * n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim, n_timesteps)
        Returns:
            (batch, n_timesteps, n_output)
        """
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch, 1, feature_dim, n_timesteps)
        x = self.conv_net(x)
        x = x.view(batch_size, -1)
        x = self.final_linear(x)
        x = x.view(batch_size, self.n_timesteps, self.n_output)
        return x


class GatedFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) from QTSTransformer.

    Provides expressive non-linear transformation with gating mechanism.
    Uses GELU activation and sigmoid gating.

    Architecture:
        Input: (batch, ..., input_dim)
        Linear(input_dim → 2 * hidden_dim)
        Split into gate and content
        Output = GELU(content) * Sigmoid(gate)
        Linear(hidden_dim → output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_multiplier: int = 4):
        super().__init__()

        hidden_dim = input_dim * hidden_multiplier

        # Project to 2x hidden for gate + content split
        self.W_in = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, ..., input_dim)
        Returns:
            (batch, ..., output_dim)
        """
        x_proj = self.W_in(x)
        gate, content = x_proj.chunk(2, dim=-1)
        gated = self.activation(content) * torch.sigmoid(gate)
        return self.W_out(gated)


class Conv2dGLUPreprocessor(nn.Module):
    """
    Combined Conv2d + GLU from QTSTransformer.

    This is the RECOMMENDED encoder for best performance.
    Combines spatial feature extraction with gated non-linearity.

    Architecture:
        Input: (batch, feature_dim, n_timesteps)
        Conv2d Network (same as Conv2dFeatureExtractor)
        Flatten
        GatedFeedForward
        Reshape to (batch, n_timesteps, n_output)
    """

    def __init__(self, feature_dim: int, n_timesteps: int, n_output: int):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_output = n_output

        # 2D Convolutional Network
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, feature_dim, n_timesteps)
            dummy_out = self.conv_net(dummy)
            self.flattened_size = dummy_out.numel()

        # Gated Feed-Forward Network
        self.gated_ffn = GatedFeedForward(
            input_dim=self.flattened_size,
            output_dim=n_timesteps * n_output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim, n_timesteps)
        Returns:
            (batch, n_timesteps, n_output)
        """
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # Add channel dim
        x = self.conv_net(x)
        x = x.view(batch_size, -1)
        x = self.gated_ffn(x)
        return x.view(batch_size, self.n_timesteps, self.n_output)


class QTSFeatureEncoder(nn.Module):
    """
    Unified Feature Encoder from QTSTransformer.

    CRITICAL: This encoder must be IDENTICAL across all ablation models
    to ensure controlled comparison. Only the quantum mixing differs.

    Projection Types:
    - 'Linear': Simple linear projection (fastest, least expressive)
    - 'Conv1d': 1D temporal convolution
    - 'Conv2d': 2D CNN feature extraction
    - 'Conv2d_GLU': 2D CNN + Gated Linear Unit (RECOMMENDED)
    - 'GLU': Gated Linear Unit only

    Output: Angles in [0, π] ready for quantum circuit encoding

    Usage:
        encoder = QTSFeatureEncoder(
            feature_dim=129,      # Input feature dimension
            n_timesteps=200,      # Sequence length
            n_output=4,           # n_qubits for quantum circuit
            projection_type='Conv2d_GLU'
        )
        angles = encoder(x)  # (batch, n_timesteps, n_output) in [0, π]
    """

    PROJECTION_TYPES = Literal['Linear', 'Conv1d', 'Conv2d', 'Conv2d_GLU', 'GLU']

    def __init__(
        self,
        feature_dim: int,
        n_timesteps: int,
        n_output: int,
        projection_type: str = 'Conv2d_GLU',
        dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_timesteps = n_timesteps
        self.n_output = n_output
        self.projection_type = projection_type

        self.dropout = nn.Dropout(dropout)

        # Build projection layer (IDENTICAL to QTSTransformer)
        if projection_type == 'Linear':
            self.projection = nn.Linear(feature_dim, n_output)
            self._needs_transpose = True
        elif projection_type == 'Conv1d':
            self.projection = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=n_output,
                kernel_size=3,
                padding=1
            )
            self._needs_transpose = False
        elif projection_type == 'Conv2d':
            self.projection = Conv2dFeatureExtractor(feature_dim, n_timesteps, n_output)
            self._needs_transpose = False
        elif projection_type == 'Conv2d_GLU':
            self.projection = Conv2dGLUPreprocessor(feature_dim, n_timesteps, n_output)
            self._needs_transpose = False
        elif projection_type == 'GLU':
            self.projection = GatedFeedForward(feature_dim, n_output)
            self._needs_transpose = True
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        # Sigmoid activation scales to [0, π] for quantum angles
        self.angle_scale = np.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features and convert to quantum angles.

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)
        Returns:
            angles: (batch, n_timesteps, n_output) in [0, π]
        """
        # Apply dropout
        x = self.dropout(x)

        # Handle different input formats and projection types
        if self.projection_type == 'Linear' or self.projection_type == 'GLU':
            # These need (batch, n_timesteps, feature_dim)
            if x.shape[1] == self.feature_dim and x.shape[2] == self.n_timesteps:
                x = x.transpose(1, 2)  # (batch, feature_dim, n_timesteps) → (batch, n_timesteps, feature_dim)
            features = self.projection(x)
        elif self.projection_type == 'Conv1d':
            # Conv1d needs (batch, feature_dim, n_timesteps)
            if x.shape[2] == self.feature_dim:
                x = x.transpose(1, 2)
            features = self.projection(x)  # (batch, n_output, n_timesteps)
            features = features.transpose(1, 2)  # (batch, n_timesteps, n_output)
        else:
            # Conv2d and Conv2d_GLU expect (batch, feature_dim, n_timesteps)
            if x.shape[2] == self.feature_dim:
                x = x.transpose(1, 2)
            features = self.projection(x)

        # Scale to quantum angles [0, π] using sigmoid
        angles = torch.sigmoid(features) * self.angle_scale

        return angles

    def get_param_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"QTSFeatureEncoder(\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  n_timesteps={self.n_timesteps},\n"
            f"  n_output={self.n_output},\n"
            f"  projection_type='{self.projection_type}',\n"
            f"  params={self.get_param_count():,}\n"
            f")"
        )


def create_qts_encoder(
    feature_dim: int,
    n_timesteps: int,
    n_qubits: int,
    projection_type: str = 'Conv2d_GLU',
    dropout: float = 0.1
) -> QTSFeatureEncoder:
    """
    Convenience function to create QTS encoder with standard configuration.

    Args:
        feature_dim: Input feature dimension (e.g., 129 for EEG channels)
        n_timesteps: Sequence length (e.g., 200 for genomic sequences)
        n_qubits: Number of qubits (output dimension for quantum circuit)
        projection_type: Type of projection ('Linear', 'Conv2d', 'Conv2d_GLU', 'GLU')
        dropout: Dropout rate

    Returns:
        QTSFeatureEncoder instance ready for use
    """
    return QTSFeatureEncoder(
        feature_dim=feature_dim,
        n_timesteps=n_timesteps,
        n_output=n_qubits,
        projection_type=projection_type,
        dropout=dropout
    )


# Export main classes
__all__ = [
    'QTSFeatureEncoder',
    'Conv2dFeatureExtractor',
    'GatedFeedForward',
    'Conv2dGLUPreprocessor',
    'create_qts_encoder'
]
