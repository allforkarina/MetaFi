from __future__ import annotations

"""Amp-phase WPFormer variant with independent Shared CNN encoders."""

import torch
from torch import Tensor, nn

from .shared_cnn import SharedCNN
from .transformer_decoder import TransformerDecoderModule


class WPFormerAmpPhase(nn.Module):
    """Fuse CSI amplitude and phase-cosine features before the shared decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.amp_encoder = SharedCNN()                      # double feature: amp and phase-cosine
        self.phase_encoder = SharedCNN()                    # different weight of share CNN
        self.fusion = nn.Conv2d(1024, 512, kernel_size=1)   # downsample to fit TransformerDecoderModule
        self.decoder = TransformerDecoderModule()

    def forward(self, amplitude: Tensor, phase_cos: Tensor) -> Tensor:
        amp_features = self.amp_encoder(amplitude)
        phase_features = self.phase_encoder(phase_cos)
        fused_features = torch.cat([amp_features, phase_features], dim=1)   # concat 512 -> 1024
        fused_features = self.fusion(fused_features)                        # fusion downsample 1024 -> 512
        return self.decoder(fused_features)
