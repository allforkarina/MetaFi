from __future__ import annotations

"""Top-level WPFormer wrapper for end-to-end pose regression."""

from torch import Tensor, nn

from .shared_cnn import SharedCNN
from .transformer_decoder import TransformerDecoderModule


class WPFormer(nn.Module):
    """Compose Shared CNN and Transformer/Decoder into one end-to-end model."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SharedCNN()
        self.decoder = TransformerDecoderModule()

    def forward(self, x_c: Tensor) -> Tensor:
        features = self.encoder(x_c)
        return self.decoder(features)
