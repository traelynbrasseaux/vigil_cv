"""Loss functions for anomaly detection training."""

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """Combined MSE + SSIM-like reconstruction loss for autoencoder training.

    Args:
        mse_weight: Weight for the MSE component.
        l1_weight: Weight for the L1 component.
    """

    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 0.1) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined reconstruction loss.

        Args:
            reconstruction: Reconstructed image tensor [B, C, H, W].
            target: Original image tensor [B, C, H, W].

        Returns:
            Scalar loss tensor.
        """
        mse = self.mse_loss(reconstruction, target)
        l1 = self.l1_loss(reconstruction, target)
        return self.mse_weight * mse + self.l1_weight * l1
