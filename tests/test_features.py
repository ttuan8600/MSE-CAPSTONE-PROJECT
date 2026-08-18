"""Tests for the band-power, alignment and spectrogram feature extractors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.preprocessing.features import (
    EEG_BANDS,
    SpecAugment,
    differential_entropy,
    euclidean_alignment,
    euclidean_alignment_matrix,
    log_mel_spectrogram,
)

FS = 125
N_CHANNELS = 30
N_TIME = 2500          # 20 s at 125 Hz


def sine_trial(freq: float, amplitude: float = 1.0, fs: int = FS, n_time: int = N_TIME):
    """A trial whose every channel is a sinusoid at one frequency."""
    t = np.arange(n_time) / fs
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    return np.tile(wave, (N_CHANNELS, 1)).astype(np.float32)


# --- differential entropy ----------------------------------------------------

def test_de_output_shape():
    de = differential_entropy(sine_trial(10.0), sample_rate=FS)
    assert de.shape == (len(EEG_BANDS) * N_CHANNELS, N_TIME // FS)
    assert de.dtype == np.float32


def test_de_compresses_the_input():
    """The whole point: 75,000 raw values become 3,000."""
    trial = sine_trial(10.0)
    de = differential_entropy(trial, sample_rate=FS)
    assert de.size < trial.size / 20


def test_de_localises_energy_to_the_correct_band():
    """A 10 Hz sinusoid must put its energy in alpha (8-13 Hz)."""
    de = differential_entropy(sine_trial(10.0), sample_rate=FS)
    per_band = de.reshape(len(EEG_BANDS), N_CHANNELS, -1).mean(axis=(1, 2))
    alpha_index = [i for i, (name, _, _) in enumerate(EEG_BANDS) if name == "alpha"][0]
    assert int(np.argmax(per_band)) == alpha_index


def test_de_tracks_amplitude_logarithmically():
    """DE is log band-power, so a 10x amplitude is a fixed additive offset."""
    quiet = differential_entropy(sine_trial(10.0, amplitude=1.0), sample_rate=FS)
    loud = differential_entropy(sine_trial(10.0, amplitude=10.0), sample_rate=FS)
    alpha_index = [i for i, (name, _, _) in enumerate(EEG_BANDS) if name == "alpha"][0]
    lo = quiet.reshape(len(EEG_BANDS), N_CHANNELS, -1)[alpha_index].mean()
    hi = loud.reshape(len(EEG_BANDS), N_CHANNELS, -1)[alpha_index].mean()
    # power scales by 100, so DE rises by 0.5 * ln(100) ~= 2.30
    assert hi - lo == pytest.approx(0.5 * np.log(100.0), abs=0.05)


def test_de_survives_a_dead_channel():
    """A zero-variance channel must not produce -inf."""
    trial = sine_trial(10.0)
    trial[3] = 0.0
    de = differential_entropy(trial, sample_rate=FS)
    assert np.isfinite(de).all()


def test_de_rejects_wrong_rank():
    with pytest.raises(ValueError, match="channels, time"):
        differential_entropy(np.zeros(N_TIME), sample_rate=FS)


def test_de_rejects_window_longer_than_trial():
    with pytest.raises(ValueError, match="shorter than one window"):
        differential_entropy(sine_trial(10.0, n_time=50), sample_rate=FS, window_seconds=1.0)


# --- Euclidean alignment -----------------------------------------------------

def test_alignment_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    trials = rng.standard_normal((20, N_CHANNELS, 500))
    whitening = euclidean_alignment_matrix(trials)
    assert whitening.shape == (N_CHANNELS, N_CHANNELS)
    np.testing.assert_allclose(whitening, whitening.T, atol=1e-8)


def test_alignment_whitens_mean_covariance_to_identity():
    """After alignment, a subject's mean covariance should be ~I."""
    rng = np.random.default_rng(1)
    mixing = rng.standard_normal((N_CHANNELS, N_CHANNELS))
    trials = np.einsum("cd,ndt->nct", mixing, rng.standard_normal((40, N_CHANNELS, 500)))

    aligned = euclidean_alignment(trials)
    n_time = aligned.shape[-1]
    mean_cov = np.einsum("nct,ndt->ncd", aligned, aligned).mean(axis=0) / n_time
    np.testing.assert_allclose(mean_cov, np.eye(N_CHANNELS), atol=0.05)


def test_alignment_makes_two_subjects_comparable():
    """Different mixing matrices -> different covariances -> same after alignment."""
    rng = np.random.default_rng(2)
    source = rng.standard_normal((40, N_CHANNELS, 500))

    a = euclidean_alignment(np.einsum("cd,ndt->nct", rng.standard_normal((N_CHANNELS, N_CHANNELS)), source))
    b = euclidean_alignment(np.einsum("cd,ndt->nct", rng.standard_normal((N_CHANNELS, N_CHANNELS)), source))

    def mean_cov(x):
        return np.einsum("nct,ndt->ncd", x, x).mean(axis=0) / x.shape[-1]

    np.testing.assert_allclose(mean_cov(a), mean_cov(b), atol=0.1)


def test_alignment_preserves_shape_and_dtype():
    rng = np.random.default_rng(3)
    trials = rng.standard_normal((10, N_CHANNELS, 300))
    aligned = euclidean_alignment(trials)
    assert aligned.shape == trials.shape
    assert aligned.dtype == np.float32
    assert np.isfinite(aligned).all()


def test_alignment_rejects_wrong_rank():
    with pytest.raises(ValueError, match="trials, channels, time"):
        euclidean_alignment(np.zeros((N_CHANNELS, 500)))


# --- log-mel -----------------------------------------------------------------

def test_log_mel_shape_and_range():
    rng = np.random.default_rng(4)
    waveform = rng.standard_normal(16000 * 2).astype(np.float32)
    mel = log_mel_spectrogram(waveform, sample_rate=16000, n_mels=64, hop_length=256)
    assert mel.shape[0] == 64
    assert mel.dtype == np.float32
    # power_to_db with top_db=80 bounds the output relative to its own maximum.
    assert mel.max() <= 0.0 + 1e-5
    assert mel.min() >= -80.0 - 1e-5


def test_log_mel_richer_than_mfcc():
    """64 mel bands carry more than 13 MFCCs -- the reason for the change."""
    rng = np.random.default_rng(5)
    waveform = rng.standard_normal(16000).astype(np.float32)
    mel = log_mel_spectrogram(waveform, sample_rate=16000, n_mels=64)
    assert mel.shape[0] > 13


def test_log_mel_rejects_empty():
    with pytest.raises(ValueError, match="empty waveform"):
        log_mel_spectrogram(np.array([], dtype=np.float32), sample_rate=16000)


# --- SpecAugment -------------------------------------------------------------

def test_specaugment_preserves_shape():
    batch = torch.randn(8, 64, 200)
    out = SpecAugment()(batch)
    assert out.shape == batch.shape


def test_specaugment_modifies_some_samples():
    torch.manual_seed(0)
    batch = torch.randn(64, 64, 200)
    out = SpecAugment(probability=1.0)(batch)
    changed = [(not torch.equal(out[i], batch[i])) for i in range(batch.shape[0])]
    assert any(changed)


def test_specaugment_leaves_input_untouched():
    """It must not mutate the caller's tensor."""
    torch.manual_seed(1)
    batch = torch.randn(4, 64, 200)
    reference = batch.clone()
    SpecAugment(probability=1.0)(batch)
    assert torch.equal(batch, reference)


def test_specaugment_disabled_is_a_noop():
    batch = torch.randn(4, 64, 200)
    out = SpecAugment(probability=0.0)(batch)
    assert torch.equal(out, batch)
