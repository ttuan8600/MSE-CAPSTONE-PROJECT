"""Speech signal preprocessing utilities."""

import numpy as np


def compute_spectrogram(waveform: np.ndarray, fs: float) -> np.ndarray:
    """Compute a spectrogram from a raw waveform."""
    # placeholder using numpy or librosa later
    raise NotImplementedError


def detect_glottal_closure(waveform: np.ndarray, fs: float) -> np.ndarray:
    """Find glottal closure instants using zero frequency filtering (ZFF)."""
    raise NotImplementedError
