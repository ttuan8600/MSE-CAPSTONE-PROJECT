"""Speech signal preprocessing utilities."""

import numpy as np


def compute_spectrogram(waveform: np.ndarray, fs: float) -> np.ndarray:
    """Compute a spectrogram from a raw waveform.

    Tries to use `librosa` if available for convenience; falls back to
    SciPy's `spectrogram` function otherwise. Returns magnitude in dB
    (shape: [n_freq_bins, n_frames]).
    """
    # prefer librosa if installed
    try:
        import librosa
        S = librosa.stft(waveform, n_fft=1024, hop_length=512, win_length=1024)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        return S_db
    except ImportError:
        # fallback to scipy
        from scipy import signal

        freqs, times, Sxx = signal.spectrogram(waveform, fs=fs, nperseg=1024,
                                               noverlap=512, window='hann')
        # convert to log scale (dB)
        S_db = 10 * np.log10(Sxx + 1e-10)
        return S_db


def detect_glottal_closure(waveform: np.ndarray, fs: float) -> np.ndarray:
    """Find glottal closure instants (GCIs) in a speech waveform.

    A simple placeholder implementation that uses the zero-crossing rate of
    the differentiated signal to approximate GCIs. Returns a boolean mask the
    same length as ``waveform`` with ``True`` at estimated closure points.
    For research-quality results, replace with a proper ZFF or DYPSA
    algorithm.
    """
    # differentiate signal
    diff = np.diff(waveform, prepend=waveform[0])
    # compute zero crossings
    zc = np.where(np.diff(np.signbit(diff)))[0]
    mask = np.zeros(len(waveform), dtype=bool)
    mask[zc] = True
    return mask
