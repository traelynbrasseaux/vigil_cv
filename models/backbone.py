"""Multi-scale feature extractor backbones for anomaly detection."""

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Multi-scale feature extractor wrapping pretrained classification backbones.

    Extracts intermediate feature maps from layers 2 and 3 of the backbone,
    concatenates them at 1/8 spatial resolution for rich mid-level semantics.

    Args:
        backbone_name: Which backbone to use ('efficientnet' or 'mobilenet').
        pretrained: Whether to load pretrained ImageNet weights.
    """

    def __init__(
        self,
        backbone_name: Literal["efficientnet", "mobilenet"] = "efficientnet",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "efficientnet":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            # EfficientNet features: 0=stem, 1=block0, 2=block1, 3=block2, ...
            self.layer0 = nn.Sequential(base.features[0], base.features[1])  # stem + block0
            self.layer1 = base.features[2]  # block1: 1/4 res
            self.layer2 = base.features[3]  # block2: 1/8 res, 40 channels
            self.layer3 = base.features[4]  # block3: 1/16 res, 80 channels
            self._layer2_channels = 40
            self._layer3_channels = 80

        elif backbone_name == "mobilenet":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            features = list(base.features.children())
            self.layer0 = nn.Sequential(*features[:2])   # stem + first inverted residual
            self.layer1 = nn.Sequential(*features[2:4])   # 1/4 res
            self.layer2 = nn.Sequential(*features[4:7])   # 1/8 res, 24 channels
            self.layer3 = nn.Sequential(*features[7:9])   # 1/16 res, 40 channels
            self._layer2_channels = 24
            self._layer3_channels = 40

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}. Use 'efficientnet' or 'mobilenet'.")

        # Upsample layer3 features to match layer2 spatial dimensions (1/8 res)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Freeze all backbone parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    @property
    def output_channels(self) -> int:
        """Total number of output channels (layer2 + layer3 concatenated)."""
        return self._layer2_channels + self._layer3_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate multi-scale features.

        Args:
            x: Input tensor of shape [B, 3, 224, 224].

        Returns:
            Feature tensor of shape [B, C, H/8, W/8] where C = layer2_ch + layer3_ch.
        """
        x = self.layer0(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)       # [B, C2, H/8, W/8]
        feat3 = self.layer3(feat2)    # [B, C3, H/16, W/16]

        # Upsample feat3 to match feat2 spatial size
        feat3_up = self.upsample(feat3)  # [B, C3, H/8, W/8]

        # Concatenate along channel dimension
        return torch.cat([feat2, feat3_up], dim=1)  # [B, C2+C3, H/8, W/8]
