"""Encoders for the band-power and log-mel representations.

Both exist to address a measured failure rather than to add capacity.

:class:`EEGDEEncoder` is **deliberately small**. The raw-signal encoder has
577,088 parameters and reached 99.75% training accuracy against 42.17%
validation --- it memorised its 28 training subjects. Band-power features reduce
the input from 75,000 numbers per trial to 3,000, and this encoder reduces the
parameter count by roughly a factor of five. Less capacity is the intervention,
not a compromise.

:class:`AudioMelEncoder` goes the other way. Audio carries effectively all of
this system's signal on a 62,208-parameter encoder over 13 MFCCs, which is the
weakest component in the pipeline. It is given a richer input (64 mel bands) and
enough capacity to use it, with dropout and SpecAugment supplying the
regularisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGDEEncoder(nn.Module):
    """Encoder for differential-entropy band-power features.

    Input is ``(batch, n_bands * n_channels, n_windows)`` --- by default
    ``(batch, 150, 20)`` for 5 bands x 30 channels over 20 one-second windows.

    The leading BatchNorm standardises each band/channel feature across the
    batch, which is why the dataset does not z-score these features itself:
    per-trial normalisation would destroy the band-power magnitudes that carry
    the emotional signal.
    """

    def __init__(
        self,
        in_features: int = 150,
        latent_dim: int = 128,
        hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.input_norm = nn.BatchNorm1d(in_features)

        self.conv1 = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, latent_dim)

    def forward(self, x, return_sequence: bool = False):
        """``return_sequence`` yields ``(batch, time, latent)`` before pooling.

        The convolutional trunk is identical either way, so a sequence-fusion
        ablation differs from the pooled one only in where pooling happens --
        which is the thing being tested.
        """
        x = self.input_norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        if return_sequence:
            # (B, hidden, T) -> (B, T, latent)
            return self.fc(self.dropout(x).transpose(1, 2))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class AudioMelEncoder(nn.Module):
    """Encoder for log-mel spectrograms.

    Input is ``(batch, n_mels, frames)``, treating mel bands as channels and
    convolving over time. Three strided blocks reduce the ~1,313-frame sequence
    before global pooling.
    """

    def __init__(
        self,
        n_mels: int = 64,
        latent_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim

        self.input_norm = nn.BatchNorm1d(n_mels)

        self.conv1 = nn.Conv1d(n_mels, 128, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 192, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x, return_sequence: bool = False):
        """``return_sequence`` yields ``(batch, frames, latent)`` before pooling."""
        x = self.input_norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        if return_sequence:
            return self.fc(self.dropout(x).transpose(1, 2))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def build_eeg_encoder(features: str, in_features: int, latent_dim: int = 128,
                      dropout: float = 0.5) -> nn.Module:
    """Return the encoder matching an EEG representation."""
    if features == "de":
        return EEGDEEncoder(
            in_features=in_features, latent_dim=latent_dim, dropout=dropout
        )
    if features == "raw":
        from .eeg_encoder import EEGEncoder

        return EEGEncoder(in_channels=in_features, latent_dim=latent_dim)
    raise ValueError(f"unknown EEG feature type {features!r}")


def build_audio_encoder(features: str, in_features: int, latent_dim: int = 128,
                        dropout: float = 0.3) -> nn.Module:
    """Return the encoder matching an audio representation."""
    if features == "mel":
        return AudioMelEncoder(
            n_mels=in_features, latent_dim=latent_dim, dropout=dropout
        )
    if features == "mfcc":
        from .eeg_encoder import AudioEncoder

        return AudioEncoder(n_mfcc=in_features, latent_dim=latent_dim)
    raise ValueError(f"unknown audio feature type {features!r}")
