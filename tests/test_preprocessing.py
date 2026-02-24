import pytest

from src.preprocessing import eeg, speech


def test_eeg_module_exists():
    # the functions currently raise NotImplementedError, but import should work
    with pytest.raises(NotImplementedError):
        eeg.load_eeg("dummy")


def test_speech_module_exists():
    with pytest.raises(NotImplementedError):
        speech.compute_spectrogram(None, 16000)
