"""Low-level readers for the EAV dataset's MATLAB files.

The important correction encoded here concerns the axis order of the EEG array.
The original loader assumed ``(n_segments, n_channels, time)`` and therefore read
``seg[0, :, :]`` as "the first segment". The array is in fact

    ``(time, channels, trials)`` == ``(10000, 30, 200)``

so ``seg[0, :, :]`` was a single 2 ms time-point spread across all 200 trials,
reinterpreted as a 200-step time series. See ``docs/DATA_CORRECTIONS.md``.

The recording is 500 Hz, so 10,000 samples is a 20 s trial. 30 channels are
recorded. Each subject contributes 200 trials: 100 *Listening* and 100
*Speaking*; only the Speaking trials have a matched audio recording.

The MATLAB variable holding the EEG is named ``seg`` for some subjects and
``seg1`` for others; both are accepted, but nothing else is, because guessing at
an unknown key is how the original silent-fallback bug started.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat

#: Native sampling rate of the EAV EEG recordings, in Hz.
EAV_SAMPLE_RATE = 500

#: Number of EEG channels in the EAV recordings.
EAV_N_CHANNELS = 30

#: Trials per subject (100 Listening + 100 Speaking).
EAV_N_TRIALS = 200

#: Accepted MATLAB variable names for the EEG array, in priority order.
EEG_KEY_CANDIDATES = ("seg", "seg1")

#: ``001_Trial_01_Listening_Neutral.mp4`` -> index 1, Listening, Neutral.
MEDIA_FILENAME_RE = re.compile(
    r"^(?P<idx>\d{3})_Trial_(?P<trial>\d+)_(?P<cond>Listening|Speaking)_(?P<emo>[A-Za-z]+)",
    re.IGNORECASE,
)


class EAVDataError(RuntimeError):
    """Raised when EAV data on disk does not match the documented structure.

    This is deliberately fatal. The predecessor of this module returned
    ``np.random.randn(...)`` when a file failed to load, which silently injected
    noise samples carrying real labels into training and evaluation runs.
    """


def resolve_eeg_key(mat: dict, source: Path) -> str:
    """Return the MATLAB variable holding the EEG array, or raise."""
    for key in EEG_KEY_CANDIDATES:
        if key in mat:
            return key
    available = sorted(k for k in mat if not k.startswith("__"))
    raise EAVDataError(
        f"{source}: no EEG variable found. Expected one of {EEG_KEY_CANDIDATES}, "
        f"available: {available}"
    )


def load_subject_segments(eeg_file: Path) -> np.ndarray:
    """Load a subject's EEG array as ``(trials, channels, time)``.

    The on-disk layout is ``(time, channels, trials)``; this function transposes
    it into the trial-major order the rest of the pipeline expects, so that
    ``segments[i]`` is the complete ``(channels, time)`` recording for trial ``i``.

    Raises
    ------
    EAVDataError
        If the file is missing, has no recognised EEG variable, is not 3-D, or
        contains non-finite values.
    """
    eeg_file = Path(eeg_file)
    if not eeg_file.exists():
        raise EAVDataError(f"EEG file not found: {eeg_file}")

    try:
        mat = loadmat(str(eeg_file))
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise EAVDataError(f"{eeg_file}: could not read MATLAB file ({exc})") from exc

    seg = mat[resolve_eeg_key(mat, eeg_file)]

    if seg.ndim != 3:
        raise EAVDataError(f"{eeg_file}: expected a 3-D EEG array, got shape {seg.shape}")

    n_time, n_channels, n_trials = seg.shape
    if n_channels != EAV_N_CHANNELS:
        raise EAVDataError(
            f"{eeg_file}: expected {EAV_N_CHANNELS} channels on axis 1, "
            f"got shape {seg.shape}. The array must be (time, channels, trials)."
        )
    if n_time < n_channels or n_time < n_trials:
        raise EAVDataError(
            f"{eeg_file}: axis 0 ({n_time}) should be the time axis and is "
            f"unexpectedly short for shape {seg.shape}"
        )

    segments = np.ascontiguousarray(seg.transpose(2, 1, 0), dtype=np.float32)

    if not np.isfinite(segments).all():
        n_bad = int((~np.isfinite(segments)).sum())
        raise EAVDataError(f"{eeg_file}: {n_bad} non-finite values in EEG array")

    return segments


def load_subject_labels(label_file: Path) -> np.ndarray:
    """Load a subject's raw ``(10, n_trials)`` one-hot label matrix."""
    label_file = Path(label_file)
    if not label_file.exists():
        raise EAVDataError(f"label file not found: {label_file}")
    try:
        mat = loadmat(str(label_file))
    except Exception as exc:  # noqa: BLE001
        raise EAVDataError(f"{label_file}: could not read MATLAB file ({exc})") from exc
    if "label" not in mat:
        available = sorted(k for k in mat if not k.startswith("__"))
        raise EAVDataError(f"{label_file}: no 'label' variable. Available: {available}")
    return mat["label"]


def subject_eeg_paths(subject_dir: Path) -> Tuple[Path, Path]:
    """Return ``(eeg_file, label_file)`` for a subject directory."""
    subject_dir = Path(subject_dir)
    name = subject_dir.name
    return (
        subject_dir / "EEG" / f"{name}_eeg.mat",
        subject_dir / "EEG" / f"{name}_eeg_label.mat",
    )


def list_subject_dirs(eav_root: Path, subjects: List[int] | None = None) -> List[Path]:
    """Return subject directories under ``eav_root``, sorted by subject number."""
    eav_root = Path(eav_root)
    if not eav_root.is_dir():
        raise EAVDataError(f"EAV root directory not found: {eav_root}")
    dirs = sorted(
        (d for d in eav_root.iterdir() if d.is_dir() and d.name.startswith("subject")),
        key=lambda p: int(p.name[len("subject"):]),
    )
    if subjects is not None:
        wanted = set(subjects)
        dirs = [d for d in dirs if int(d.name[len("subject"):]) in wanted]
    if not dirs:
        raise EAVDataError(f"no subject directories found under {eav_root}")
    return dirs


def parse_media_filename(name: str) -> Tuple[int, str, str]:
    """Parse ``001_Trial_01_Listening_Neutral.mp4`` into ``(1, 'Listening', 'Neutral')``.

    The leading three digits are the **1-based trial index** into the EEG array's
    trial axis. This is the link that makes EEG/audio pairing correct.
    """
    m = MEDIA_FILENAME_RE.match(name)
    if not m:
        raise EAVDataError(f"unparseable EAV media filename: {name!r}")
    return (
        int(m.group("idx")),
        m.group("cond").capitalize(),
        m.group("emo").capitalize(),
    )
