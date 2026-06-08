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


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        seq_len: int = 24,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.encoder(x)
        bottleneck = h_n[-1]

        dec_input = bottleneck.unsqueeze(1).expand(-1, self.seq_len, -1)
        dec_output, _ = self.decoder(dec_input)
        reconstructed = self.output_layer(dec_output)

        return reconstructed, bottleneck


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


class VAE(nn.Module):
    """Variational Autoencoder para detecção de anomalias.

    Usa MSE de reconstrução como score (sem KL em inferência).
    Em treino, aplica reparameterization trick; em eval, usa a média diretamente.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_layers: Optional[list[int]] = None,
        latent_dim: int = 8,
    ):
        super().__init__()
        if encoding_layers is None:
            encoding_layers = _default_encoding_layers(input_dim)

        # Encoder: input_dim → encoding_layers → mean / log_var
        enc_dims = [input_dim] + encoding_layers
        enc_layers: list[nn.Module] = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            enc_layers.append(nn.BatchNorm1d(enc_dims[i + 1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mean = nn.Linear(encoding_layers[-1], latent_dim)
        self.fc_log_var = nn.Linear(encoding_layers[-1], latent_dim)

        # Decoder: latent_dim → reversed(encoding_layers) → input_dim
        dec_dims = [latent_dim] + list(reversed(encoding_layers))
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Linear(dec_dims[-1], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_log_var(h)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mean + torch.randn_like(std) * std
        return mean  # inferência determinística

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon = self.decode(z)
        return recon, mean, log_var
