"""EEG preprocessing routines for EmoAI.

Contains functions for loading, filtering, segmenting EEG signals.
"""

import numpy as np


def load_eeg(file_path: str) -> np.ndarray:
    """Load EEG data from a file (placeholder)."""
    # TODO: implement actual I/O using MNE or h5py depending on dataset format
    raise NotImplementedError


def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Apply a bandpass filter to an EEG signal."""
    # placeholder implementation
    return signal
