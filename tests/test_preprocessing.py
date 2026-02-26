import sys
from pathlib import Path

# add project root so imports of `src` work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from src.preprocessing import eeg, speech


def test_eeg_module_exists():
    # the import should succeed and the loader should raise appropriate errors
    with pytest.raises(FileNotFoundError):
        eeg.load_eeg("dummy")


def test_load_eeg_numpy(tmp_path):
    """Loading a .npy file returns the correct array."""
    arr = np.arange(12).reshape(3, 4)
    file = tmp_path / "test.npy"
    np.save(file, arr)

    loaded = eeg.load_eeg(str(file))
    assert isinstance(loaded, np.ndarray)
    np.testing.assert_array_equal(loaded, arr)


def test_list_subject_folders(tmp_path):
    # create unsorted subject directories
    (tmp_path / "subject10").mkdir()
    (tmp_path / "subject2").mkdir()
    (tmp_path / "subject1").mkdir()

    ordered = eeg.list_subject_folders(str(tmp_path))
    assert ordered == ["subject1", "subject2", "subject10"]

    # nonexistent path should raise
    with pytest.raises(FileNotFoundError):
        eeg.list_subject_folders(str(tmp_path / "nope"))


def test_speech_module_exists():
    with pytest.raises(NotImplementedError):
        speech.compute_spectrogram(None, 16000) # type: ignore
