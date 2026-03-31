import torch
import torch.nn as nn
from typing import Optional


def _default_encoding_layers(input_dim: int) -> list[int]:
    if input_dim <= 8:
        return [32, 16, 8]
    elif input_dim <= 12:
        return [48, 24, 12]
    else:
        return [64, 32, 16]


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_layers: Optional[list[int]] = None):
        super().__init__()

        if encoding_layers is None:
            encoding_layers = _default_encoding_layers(input_dim)

        # Build encoder: input_dim -> ... -> bottleneck
        encoder_dims = [input_dim] + encoding_layers
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:  # no BN/ReLU on bottleneck
                encoder_layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder: bottleneck -> ... -> input_dim (symmetric)
        decoder_dims = list(reversed(encoding_layers)) + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:  # no activation on final output
                decoder_layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
