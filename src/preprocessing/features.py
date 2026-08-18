"""Feature extractors aimed at the two failures measured in the ablations.

Both are responses to specific, quantified problems rather than speculative
additions.

**EEG memorises its training subjects.** The raw-signal encoder reached 99.75%
training accuracy against 42.17% validation. It is given 30 x 2500 = 75,000 raw
values per trial and 577,088 parameters to fit 2,800 samples from 28 people.
Two standard corrections apply:

* :func:`differential_entropy` reduces each trial to band-power features --- 150
  numbers instead of 75,000 --- which is the conventional representation for EEG
  emotion recognition and removes most of the capacity to memorise.
* :func:`euclidean_alignment` makes different subjects' signal distributions
  comparable, targeting the cross-subject transfer failure directly.

**Audio is carrying the entire system on 13 MFCCs.** :func:`log_mel_spectrogram`
supplies a 64-band representation instead, and :class:`SpecAugment` regularises
it during training.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

#: Standard EEG frequency bands, in Hz. Gamma stops at 45 Hz because the
#: preprocessing band-pass does.
EEG_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)

#: log(2*pi*e) / 2 -- the constant term of the differential entropy of a
#: Gaussian, precomputed.
_HALF_LOG_2PIE = 0.5 * np.log(2 * np.pi * np.e)


def differential_entropy(
    trial: np.ndarray,
    sample_rate: int,
    bands: Sequence[Tuple[str, float, float]] = EEG_BANDS,
    window_seconds: float = 1.0,
) -> np.ndarray:
    """Band-power differential entropy features for one EEG trial.

    For a band-limited signal that is approximately Gaussian, differential
    entropy reduces to ``0.5 * log(2*pi*e*sigma^2)`` --- that is, log band-power
    up to a constant. This is the feature used throughout the EEG
    emotion-recognition literature, and its compactness is the point: a
    ``(30, 2500)`` trial becomes ``(150, 20)``.

    Parameters
    ----------
    trial:
        ``(channels, time)`` signal.
    sample_rate:
        Sampling rate of ``trial``, in Hz.
    bands:
        ``(name, low_hz, high_hz)`` triples.
    window_seconds:
        Length of each non-overlapping analysis window. Some temporal structure
        is retained rather than collapsing the trial to a single vector.

    Returns
    -------
    np.ndarray
        ``(len(bands) * channels, n_windows)`` float32 array. Row ordering is
        band-major: all channels of band 0, then all channels of band 1, and so
        on.
    """
    from scipy.signal import butter, sosfiltfilt

    if trial.ndim != 2:
        raise ValueError(f"expected (channels, time), got shape {trial.shape}")

    n_channels, n_time = trial.shape
    window = int(round(window_seconds * sample_rate))
    if window < 2:
        raise ValueError(f"window_seconds={window_seconds} is too short for {sample_rate} Hz")
    n_windows = n_time // window
    if n_windows < 1:
        raise ValueError(f"trial of {n_time} samples is shorter than one window ({window})")

    nyquist = sample_rate / 2.0
    out = np.empty((len(bands) * n_channels, n_windows), dtype=np.float32)

    for band_idx, (_, low, high) in enumerate(bands):
        # Clamp to just under Nyquist so a band that reaches the anti-aliasing
        # edge does not produce an invalid filter.
        high_eff = min(high, nyquist * 0.99)
        if low >= high_eff:
            raise ValueError(f"band ({low}, {high}) is empty at {sample_rate} Hz")

        sos = butter(4, [low, high_eff], btype="bandpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, trial, axis=-1)

        # Reshape to windows and take per-window variance along time.
        trimmed = filtered[:, : n_windows * window]
        windows = trimmed.reshape(n_channels, n_windows, window)
        variance = windows.var(axis=-1)

        # Guard the log: a dead channel has zero variance.
        de = _HALF_LOG_2PIE + 0.5 * np.log(np.maximum(variance, 1e-12))
        out[band_idx * n_channels : (band_idx + 1) * n_channels] = de

    return out


def euclidean_alignment_matrix(trials: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute a subject's whitening matrix ``R^(-1/2)`` for Euclidean Alignment.

    Euclidean Alignment (He and Wu, 2020) whitens each subject's trials by the
    mean covariance of that subject's own data, so that after alignment every
    subject's trials have an identity mean covariance. This removes a large part
    of the between-subject distribution shift that the ablations identified as
    the reason the EEG representation fails to transfer.

    It is **unsupervised** --- it uses no labels --- and it is computed
    per-subject from that subject's own recordings only. Applying it to a
    held-out test subject is therefore legitimate: it corresponds to a brief
    unlabelled calibration recording, which any real deployment would have.

    Parameters
    ----------
    trials:
        ``(n_trials, channels, time)`` array for a single subject.
    eps:
        Ridge added to the covariance diagonal for numerical stability.

    Returns
    -------
    np.ndarray
        ``(channels, channels)`` symmetric whitening matrix.
    """
    if trials.ndim != 3:
        raise ValueError(f"expected (trials, channels, time), got {trials.shape}")

    n_trials, n_channels, n_time = trials.shape

    # Mean spatial covariance across the subject's trials.
    covariances = np.einsum("nct,ndt->ncd", trials, trials) / float(n_time)
    reference = covariances.mean(axis=0)
    reference += eps * np.trace(reference) / n_channels * np.eye(n_channels)

    # Symmetric inverse square root via eigendecomposition.
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    eigenvalues = np.maximum(eigenvalues, eps)
    whitening = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    return whitening.astype(np.float64)


def euclidean_alignment(trials: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply Euclidean Alignment to one subject's trials.

    Returns ``(n_trials, channels, time)`` with the subject's mean covariance
    whitened to identity.
    """
    whitening = euclidean_alignment_matrix(trials, eps=eps)
    aligned = np.einsum("cd,ndt->nct", whitening, trials.astype(np.float64))
    return np.ascontiguousarray(aligned, dtype=np.float32)


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Log-scaled mel spectrogram.

    Replaces the 13-coefficient MFCC representation. MFCCs apply a discrete
    cosine transform to the mel spectrum and keep only the lowest coefficients,
    which was a sensible compression under 1980s constraints but discards
    spectral detail that a convolutional encoder can use. With audio carrying
    effectively all of this system's signal, that discarded detail is the most
    likely place for additional accuracy to come from.

    Returns
    -------
    np.ndarray
        ``(n_mels, frames)`` float32 array in decibels.
    """
    import librosa

    if waveform.size == 0:
        raise ValueError("empty waveform")

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax if fmax is not None else sample_rate / 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return np.ascontiguousarray(log_mel, dtype=np.float32)


class SpecAugment:
    """Time and frequency masking for spectrogram inputs (Park et al., 2019).

    Applied to training batches only. This is the regulariser an earlier version
    of this project described but never actually applied to a working pipeline;
    here it addresses a measured 31-point train/validation gap on the audio
    model.

    Masks are filled with the batch mean rather than zero, so the masked region
    is uninformative rather than an artificial extreme value.
    """

    def __init__(
        self,
        n_freq_masks: int = 2,
        freq_mask_width: int = 8,
        n_time_masks: int = 2,
        time_mask_width: int = 40,
        probability: float = 0.5,
    ):
        self.n_freq_masks = n_freq_masks
        self.freq_mask_width = freq_mask_width
        self.n_time_masks = n_time_masks
        self.time_mask_width = time_mask_width
        self.probability = probability

    def __call__(self, batch):
        """Mask a ``(batch, n_mels, frames)`` tensor in place-safe fashion."""
        import torch

        if not self.probability:
            return batch

        out = batch.clone()
        n_batch, n_mels, n_frames = out.shape
        fill = out.mean()

        selected = torch.rand(n_batch, device=out.device) < self.probability
        for i in torch.nonzero(selected, as_tuple=False).flatten().tolist():
            for _ in range(self.n_freq_masks):
                width = int(torch.randint(0, self.freq_mask_width + 1, (1,)).item())
                if width:
                    start = int(torch.randint(0, max(1, n_mels - width), (1,)).item())
                    out[i, start : start + width, :] = fill
            for _ in range(self.n_time_masks):
                width = int(torch.randint(0, self.time_mask_width + 1, (1,)).item())
                if width:
                    start = int(torch.randint(0, max(1, n_frames - width), (1,)).item())
                    out[i, :, start : start + width] = fill
        return out
