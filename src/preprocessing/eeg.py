"""EEG preprocessing routines for EmoAI.

Contains functions for loading, filtering, segmenting EEG signals.
"""

import numpy as np


def load_eeg(file_path: str) -> np.ndarray:
    """Load EEG data from a file.

    The routine currently supports NumPy ``.npy`` files and MATLAB ``.mat``
    files.  The interface is intentionally simple so that callers within the
    preprocessing pipeline can work with whatever format the raw EAV dataset
    happens to be stored in.  If you need to support additional formats
    (e.g. EDF, BrainVision, etc.) you can extend this function or replace it
    with an :mod:`mne`-based loader.

    Parameters
    ----------
    file_path : str
        Path to the file containing EEG data.

    Returns
    -------
    np.ndarray
        The raw EEG signal as a numpy array.

    Raises
    ------
    ValueError
        If the file extension is not recognized.
    FileNotFoundError
        If ``file_path`` does not exist.
    """
    import os
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".mat":
        # MATLAB files stored with scipy.  We attempt to return a sensible
        # array if a variable named ``data`` exists, otherwise we return the
        # entire dict turned into an array (which may not be what the caller
        # wants, but at least it's something).
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("scipy is required to load .mat files") from exc
        mat = loadmat(path)
        if "data" in mat:
            return mat["data"]
        # strip out meta entries that start with ``__``
        vals = [v for k, v in mat.items() if not k.startswith("__")]
        return np.asarray(vals)
    else:
        raise ValueError(f"unsupported EEG file extension '{ext}'")


def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Apply a bandpass filter to an EEG signal."""
    # placeholder implementation
    return signal


# ---------------------------------------------------------------------------
# helpers for working with the EAV dataset structure
# ---------------------------------------------------------------------------

def list_subject_folders(parent_directory: str) -> list[str]:
    """Return a sorted list of subject directory names under ``parent_directory``.

    The EAV dataset organizes each participant in a folder named ``subjectX``
    where ``X`` is an integer.  The helper mirrors the snippet found in the
    original example script:

    ```python
    subject_folders = [d for d in os.listdir(parent_directory)
                       if os.path.isdir(os.path.join(parent_directory, d))
                       and d.startswith("subject")]
    sorted_subjects = sorted(subject_folders, key=lambda s: int(s.replace("subject", "")))
    ```

    Parameters
    ----------
    parent_directory : str
        Path to the directory that contains the ``subject*`` subfolders.

    Returns
    -------
    list[str]
        Sorted list of folder names (not full paths).
    """
    import os

    if not os.path.isdir(parent_directory):
        raise FileNotFoundError(f"directory not found: {parent_directory}")

    subject_folders = [d for d in os.listdir(parent_directory)
                       if os.path.isdir(os.path.join(parent_directory, d))
                       and d.startswith("subject")]
    return sorted(subject_folders, key=lambda s: int(s.replace("subject", "")))


def iter_eeg_files(parent_directory: str):
    """Yield full paths to EEG files underneath each ``subject`` folder.

    This generator walks the typical EAV subdirectory structure and returns
    files found in ``EEG`` subfolders.  It is intended as a simple example for
    consumers of the dataset.

    Parameters
    ----------
    parent_directory : str
        Root of the raw EAV data (e.g. ``data/raw/EAV/EAV`` in this repo).

    Yields
    ------
    str
        Path to an individual EEG file.
    """
    from pathlib import Path

    base = Path(parent_directory)
    for subj in list_subject_folders(parent_directory):
        eeg_dir = base / subj / "EEG"
        if eeg_dir.is_dir():
            for file in eeg_dir.iterdir():
                if file.is_file():
                    yield str(file)


if __name__ == "__main__":
    # small script demonstrating how to walk the raw EAV directory.  This
    # mirrors the example found in the original research notebook.
    example_root = r"data/raw/EAV/EAV"
    try:
        subs = list_subject_folders(example_root)
        print("found subjects:", subs)
    except FileNotFoundError:
        print(f"example directory {example_root!r} not available")
