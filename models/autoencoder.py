"""Lightweight convolutional autoencoder for anomaly detection baseline."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for reconstruction-based anomaly detection.

    Anomaly score is computed as per-pixel MSE between input and reconstruction.
    Trained on normal images only; anomalous images produce higher reconstruction error.

    Args:
        in_channels: Number of input channels (3 for RGB).
        base_channels: Base channel count (doubled at each encoder stage).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        self.encoder = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # Block 2: 112 -> 56
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            # Block 3: 56 -> 28
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            # Block 4: 28 -> 14
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # Block 1: 14 -> 28
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            # Block 2: 28 -> 56
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            # Block 3: 56 -> 112
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # Block 4: 112 -> 224
            nn.ConvTranspose2d(c, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input image.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Reconstructed tensor of shape [B, 3, 224, 224].
        """
        z = self.encoder(x)
        return self.decoder(z)

    def anomaly_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-pixel anomaly score as reconstruction MSE.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Tuple of (score, heatmap).
            - score: Scalar anomaly score per image [B].
            - heatmap: Per-pixel MSE averaged over channels [B, 224, 224].
        """
        recon = self.forward(x)
        # Per-pixel MSE across channels
        heatmap = torch.mean((x - recon) ** 2, dim=1)  # [B, H, W]
        # Image-level score is the max pixel error
        score = heatmap.view(heatmap.shape[0], -1).max(dim=1).values  # [B]
        return score, heatmap
