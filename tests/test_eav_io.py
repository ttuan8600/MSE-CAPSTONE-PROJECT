"""Tests for the raw EAV MATLAB readers.

The central regression guarded here is the axis order. The on-disk array is
``(time, channels, trials)``; reading it as ``(segments, channels, time)`` --
which the original loader did -- silently yields a tensor of the wrong rank
semantics rather than an error, so only an explicit test catches it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from scipy.io import savemat

from src.preprocessing.eav_io import (
    EAV_N_CHANNELS,
    EAVDataError,
    list_subject_dirs,
    load_subject_labels,
    load_subject_segments,
    parse_media_filename,
    resolve_eeg_key,
    subject_eeg_paths,
)

N_TIME, N_TRIALS = 500, 8


def write_eeg(path: Path, key: str = "seg", array: np.ndarray | None = None):
    if array is None:
        # Distinct value per (time, channel, trial) so axis errors are detectable.
        array = np.arange(N_TIME * EAV_N_CHANNELS * N_TRIALS, dtype=np.float64)
        array = array.reshape(N_TIME, EAV_N_CHANNELS, N_TRIALS)
    savemat(str(path), {key: array})
    return array


def test_returns_trial_major_orientation(tmp_path):
    """``load_subject_segments`` must return (trials, channels, time)."""
    path = tmp_path / "subject1_eeg.mat"
    on_disk = write_eeg(path)

    segments = load_subject_segments(path)

    assert segments.shape == (N_TRIALS, EAV_N_CHANNELS, N_TIME)
    # Element-wise: segments[trial, channel, time] == on_disk[time, channel, trial]
    for trial in (0, 3, N_TRIALS - 1):
        for channel in (0, 17, EAV_N_CHANNELS - 1):
            np.testing.assert_allclose(
                segments[trial, channel, :], on_disk[:, channel, trial]
            )


def test_each_trial_is_a_distinct_recording(tmp_path):
    """Regression: the original loader gave every sample the same tensor."""
    path = tmp_path / "subject1_eeg.mat"
    write_eeg(path)
    segments = load_subject_segments(path)

    fingerprints = {segments[i].tobytes() for i in range(segments.shape[0])}
    assert len(fingerprints) == N_TRIALS


def test_accepts_seg1_key(tmp_path):
    """31 of the 42 real subjects store the array under 'seg1', not 'seg'."""
    path = tmp_path / "subject16_eeg.mat"
    write_eeg(path, key="seg1")
    assert load_subject_segments(path).shape == (N_TRIALS, EAV_N_CHANNELS, N_TIME)


def test_unknown_key_raises_rather_than_fabricating(tmp_path):
    """The original returned np.random.randn(...) here, keeping the real label."""
    path = tmp_path / "subject1_eeg.mat"
    savemat(str(path), {"unexpected_name": np.zeros((N_TIME, EAV_N_CHANNELS, N_TRIALS))})
    with pytest.raises(EAVDataError, match="no EEG variable found"):
        load_subject_segments(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(EAVDataError, match="not found"):
        load_subject_segments(tmp_path / "absent.mat")


def test_wrong_channel_count_raises(tmp_path):
    path = tmp_path / "subject1_eeg.mat"
    savemat(str(path), {"seg": np.zeros((N_TIME, 12, N_TRIALS))})
    with pytest.raises(EAVDataError, match="expected 30 channels"):
        load_subject_segments(path)


def test_transposed_input_is_rejected(tmp_path):
    """A (trials, channels, time) file would silently mis-load; it must raise."""
    path = tmp_path / "subject1_eeg.mat"
    savemat(str(path), {"seg": np.zeros((N_TRIALS, EAV_N_CHANNELS, N_TIME))})
    with pytest.raises(EAVDataError, match="unexpectedly short"):
        load_subject_segments(path)


def test_non_finite_values_are_fatal(tmp_path):
    path = tmp_path / "subject1_eeg.mat"
    array = np.zeros((N_TIME, EAV_N_CHANNELS, N_TRIALS))
    array[0, 0, 0] = np.nan
    savemat(str(path), {"seg": array})
    with pytest.raises(EAVDataError, match="non-finite"):
        load_subject_segments(path)


def test_resolve_eeg_key_prefers_seg():
    assert resolve_eeg_key({"seg": 1, "seg1": 2}, Path("x")) == "seg"
    assert resolve_eeg_key({"seg1": 2}, Path("x")) == "seg1"


def test_label_loader_requires_label_variable(tmp_path):
    path = tmp_path / "subject1_eeg_label.mat"
    savemat(str(path), {"not_label": np.zeros((10, 8))})
    with pytest.raises(EAVDataError, match="no 'label' variable"):
        load_subject_labels(path)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("001_Trial_01_Listening_Neutral.mp4", (1, "Listening", "Neutral")),
        ("002_Trial_02_Speaking_Neutral_Aud.wav", (2, "Speaking", "Neutral")),
        ("196_Trial_10_Speaking_Sadness_aud.wav", (196, "Speaking", "Sadness")),
        ("200_Trial_20_Speaking_Sadness.mp4", (200, "Speaking", "Sadness")),
    ],
)
def test_media_filename_parsing(name, expected):
    """The 3-digit prefix is the 1-based trial index -- the EEG/audio link."""
    assert parse_media_filename(name) == expected


def test_unparseable_filename_raises():
    with pytest.raises(EAVDataError, match="unparseable"):
        parse_media_filename("random_file.wav")


def test_subject_dirs_sort_numerically(tmp_path):
    for n in (10, 2, 1, 21):
        (tmp_path / f"subject{n}").mkdir()
    dirs = list_subject_dirs(tmp_path)
    assert [d.name for d in dirs] == ["subject1", "subject2", "subject10", "subject21"]


def test_subject_dirs_can_be_filtered(tmp_path):
    for n in (1, 2, 3):
        (tmp_path / f"subject{n}").mkdir()
    dirs = list_subject_dirs(tmp_path, subjects=[1, 3])
    assert [d.name for d in dirs] == ["subject1", "subject3"]


def test_subject_eeg_paths_naming(tmp_path):
    eeg, label = subject_eeg_paths(tmp_path / "subject7")
    assert eeg.name == "subject7_eeg.mat"
    assert label.name == "subject7_eeg_label.mat"
