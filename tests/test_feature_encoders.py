"""Tests for the band-power and log-mel encoders."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from src.models.eeg_encoder import AudioEncoder, EEGEncoder
from src.models.feature_encoders import (
    AudioMelEncoder,
    EEGDEEncoder,
    build_audio_encoder,
    build_eeg_encoder,
)

N_DE, N_WINDOWS = 150, 20
N_MELS, N_MEL_FRAMES = 64, 1313


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_de_encoder_output_shape():
    model = EEGDEEncoder(in_features=N_DE, latent_dim=128)
    out = model(torch.randn(4, N_DE, N_WINDOWS))
    assert out.shape == (4, 128)
    assert torch.isfinite(out).all()


def test_mel_encoder_output_shape():
    model = AudioMelEncoder(n_mels=N_MELS, latent_dim=128)
    out = model(torch.randn(2, N_MELS, N_MEL_FRAMES))
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_de_encoder_is_substantially_smaller_than_the_raw_encoder():
    """Reduced capacity is the intervention against subject memorisation.

    The raw-signal encoder reached 99.75% training accuracy against 42.17%
    validation. If a future change grows this encoder back toward that size, the
    intervention is undone and this test should fail.
    """
    raw = EEGEncoder(in_channels=30, latent_dim=128)
    de = EEGDEEncoder(in_features=N_DE, latent_dim=128)
    assert n_params(de) < n_params(raw) / 3


def test_mel_encoder_has_more_capacity_than_the_mfcc_encoder():
    """Audio carries the whole system; the MFCC encoder was the weakest link."""
    mfcc = AudioEncoder(n_mfcc=13, latent_dim=128)
    mel = AudioMelEncoder(n_mels=N_MELS, latent_dim=128)
    assert n_params(mel) > n_params(mfcc) * 3


def test_de_encoder_normalises_its_input():
    """The dataset does not z-score DE features, so the encoder must."""
    model = EEGDEEncoder(in_features=N_DE, latent_dim=128)
    assert isinstance(model.input_norm, torch.nn.BatchNorm1d)
    assert model.input_norm.num_features == N_DE


def test_encoders_are_deterministic_in_eval_mode():
    model = EEGDEEncoder(in_features=N_DE, latent_dim=128).eval()
    x = torch.randn(3, N_DE, N_WINDOWS)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


def test_dropout_is_active_in_train_mode():
    torch.manual_seed(0)
    model = EEGDEEncoder(in_features=N_DE, latent_dim=128, dropout=0.5).train()
    x = torch.randn(8, N_DE, N_WINDOWS)
    assert not torch.allclose(model(x), model(x))


@pytest.mark.parametrize(
    "features,in_features,expected",
    [
        ("de", N_DE, EEGDEEncoder),
        ("raw", 30, EEGEncoder),
    ],
)
def test_build_eeg_encoder_dispatch(features, in_features, expected):
    assert isinstance(build_eeg_encoder(features, in_features), expected)


@pytest.mark.parametrize(
    "features,in_features,expected",
    [
        ("mel", N_MELS, AudioMelEncoder),
        ("mfcc", 13, AudioEncoder),
    ],
)
def test_build_audio_encoder_dispatch(features, in_features, expected):
    assert isinstance(build_audio_encoder(features, in_features), expected)


def test_build_rejects_unknown_feature_types():
    with pytest.raises(ValueError, match="unknown EEG feature type"):
        build_eeg_encoder("wavelet", 30)
    with pytest.raises(ValueError, match="unknown audio feature type"):
        build_audio_encoder("spectrogram", 64)


def test_gradients_flow_through_both_encoders():
    for model, x in (
        (EEGDEEncoder(in_features=N_DE), torch.randn(4, N_DE, N_WINDOWS)),
        (AudioMelEncoder(n_mels=N_MELS), torch.randn(2, N_MELS, 256)),
    ):
        model(x).sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
