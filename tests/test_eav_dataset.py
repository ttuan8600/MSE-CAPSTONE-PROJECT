"""Tests for the cache-backed EAV dataset.

Runs against a synthetic cache so the suite passes on a fresh clone with no data.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.preprocessing.eav_dataset import (
    EAVCacheMissing,
    EAVMultimodalDataset,
    eav_collate,
)
from src.preprocessing.eav_labels import EMOTION_TO_CLASS_INDEX

N_CHANNELS, N_TIME = 30, 250
N_MFCC, N_FRAMES = 13, 60
N_DE, N_WINDOWS = 150, 20      # 5 bands x 30 channels, 1 s windows
N_MELS, N_MEL_FRAMES = 64, 80
SAMPLES_PER_SUBJECT = 10


@pytest.fixture
def cache(tmp_path):
    """A miniature but structurally faithful preprocessing cache."""
    rng = np.random.default_rng(0)
    subjects = []
    emotions = list(EMOTION_TO_CLASS_INDEX)

    for sid in (1, 2, 3):
        n_trials = SAMPLES_PER_SUBJECT * 2
        eeg = rng.standard_normal((n_trials, N_CHANNELS, N_TIME)).astype(np.float32)
        de = rng.standard_normal((n_trials, N_DE, N_WINDOWS)).astype(np.float32)
        mfcc = rng.standard_normal(
            (SAMPLES_PER_SUBJECT, N_MFCC, N_FRAMES)
        ).astype(np.float32)
        mel = rng.standard_normal(
            (SAMPLES_PER_SUBJECT, N_MELS, N_MEL_FRAMES)
        ).astype(np.float32)
        np.save(tmp_path / f"subject{sid}_eeg.npy", eeg)
        np.save(tmp_path / f"subject{sid}_de.npy", de)
        np.save(tmp_path / f"subject{sid}_mfcc.npy", mfcc)
        np.save(tmp_path / f"subject{sid}_mel.npy", mel)

        samples = []
        for row in range(SAMPLES_PER_SUBJECT):
            emotion = emotions[row % len(emotions)]
            samples.append(
                {
                    "subject_id": sid,
                    "trial_index": row * 2 + 2,
                    "eeg_row": row * 2 + 1,
                    "mfcc_row": row,
                    "emotion": emotion,
                    "condition": "Speaking",
                    "label": EMOTION_TO_CLASS_INDEX[emotion],
                }
            )

        subjects.append(
            {
                "subject_id": sid,
                "name": f"subject{sid}",
                "eeg_file": f"subject{sid}_eeg.npy",
                "de_file": f"subject{sid}_de.npy",
                "mfcc_file": f"subject{sid}_mfcc.npy",
                "mel_file": f"subject{sid}_mel.npy",
                "n_eeg_trials": n_trials,
                "n_multimodal_samples": SAMPLES_PER_SUBJECT,
                "samples": samples,
            }
        )

    manifest = {
        "cache_version": 3,
        "source": "synthetic",
        "n_subjects": len(subjects),
        "n_multimodal_samples": SAMPLES_PER_SUBJECT * len(subjects),
        "emotion_to_class_index": EMOTION_TO_CLASS_INDEX,
        "subjects": subjects,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_loads_every_sample(cache):
    ds = EAVMultimodalDataset(cache_dir=cache)
    assert len(ds) == SAMPLES_PER_SUBJECT * 3
    assert ds.n_subjects == 3


def test_sample_shapes(cache):
    ds = EAVMultimodalDataset(cache_dir=cache)
    sample = ds[0]
    assert sample["eeg"].shape == (N_CHANNELS, N_TIME)
    assert sample["audio"].shape == (N_MFCC, N_FRAMES)
    assert isinstance(sample["emotion"], int)
    assert ds.eeg_shape == (N_CHANNELS, N_TIME)
    assert ds.audio_shape == (N_MFCC, N_FRAMES)


def test_every_sample_has_a_distinct_eeg_tensor(cache):
    """The core regression: the old loader returned one tensor per subject."""
    ds = EAVMultimodalDataset(cache_dir=cache, normalize_eeg=False)
    fingerprints = {ds[i]["eeg"].numpy().tobytes() for i in range(len(ds))}
    assert len(fingerprints) == len(ds)


def test_eeg_row_maps_to_the_paired_trial(cache):
    """Sample i must carry the EEG of its own trial, not an arbitrary one."""
    ds = EAVMultimodalDataset(cache_dir=cache, normalize_eeg=False)
    raw = np.load(cache / "subject1_eeg.npy")
    for i in range(SAMPLES_PER_SUBJECT):
        expected_row = ds.samples[i]["eeg_row"]
        np.testing.assert_allclose(ds[i]["eeg"].numpy(), raw[expected_row])


def test_normalization_produces_zero_mean_unit_variance_channels(cache):
    ds = EAVMultimodalDataset(cache_dir=cache, normalize_eeg=True)
    eeg = ds[0]["eeg"].numpy()
    np.testing.assert_allclose(eeg.mean(axis=1), 0, atol=1e-5)
    np.testing.assert_allclose(eeg.std(axis=1), 1, atol=1e-3)


def test_subject_filtering(cache):
    ds = EAVMultimodalDataset(cache_dir=cache, subjects=[2])
    assert ds.n_subjects == 1
    assert set(ds.subject_ids) == {2}
    assert len(ds) == SAMPLES_PER_SUBJECT


def test_subject_ids_and_labels_align_with_samples(cache):
    ds = EAVMultimodalDataset(cache_dir=cache)
    for i in range(len(ds)):
        assert ds.subject_ids[i] == ds[i]["subject_id"]
        assert ds.labels[i] == ds[i]["emotion"]


def test_eeg_only_mode_omits_audio(cache):
    ds = EAVMultimodalDataset(cache_dir=cache, load_audio=False)
    assert "audio" not in ds[0]
    assert ds.audio_shape is None


def test_missing_cache_raises_with_build_instructions(tmp_path):
    with pytest.raises(EAVCacheMissing, match="preprocess_eav"):
        EAVMultimodalDataset(cache_dir=tmp_path / "nonexistent")


def test_stale_cache_version_raises(cache):
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    manifest["cache_version"] = 1
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EAVCacheMissing, match="--force"):
        EAVMultimodalDataset(cache_dir=cache)


def test_de_features_selectable(cache):
    ds = EAVMultimodalDataset(cache_dir=cache, eeg_features="de")
    assert ds.eeg_shape == (N_DE, N_WINDOWS)
    assert ds[0]["eeg"].shape == (N_DE, N_WINDOWS)


def test_mel_features_selectable(cache):
    ds = EAVMultimodalDataset(cache_dir=cache, audio_features="mel")
    assert ds.audio_shape == (N_MELS, N_MEL_FRAMES)
    assert ds[0]["audio"].shape == (N_MELS, N_MEL_FRAMES)


def test_de_features_are_not_per_trial_zscored(cache):
    """Z-scoring DE per trial would erase the band-power magnitudes."""
    ds = EAVMultimodalDataset(cache_dir=cache, eeg_features="de")
    assert ds.normalize_eeg is False
    raw = np.load(cache / "subject1_de.npy")
    np.testing.assert_allclose(ds[0]["eeg"].numpy(), raw[ds.samples[0]["eeg_row"]])


def test_unknown_feature_type_raises(cache):
    with pytest.raises(ValueError, match="eeg_features"):
        EAVMultimodalDataset(cache_dir=cache, eeg_features="wavelet")
    with pytest.raises(ValueError, match="audio_features"):
        EAVMultimodalDataset(cache_dir=cache, audio_features="spectrogram")


def test_unknown_subject_selection_raises(cache):
    with pytest.raises(ValueError, match="no samples selected"):
        EAVMultimodalDataset(cache_dir=cache, subjects=[999])


def test_collate_builds_batched_tensors(cache):
    ds = EAVMultimodalDataset(cache_dir=cache)
    batch = eav_collate([ds[0], ds[1], ds[2]])
    assert batch["eeg"].shape == (3, N_CHANNELS, N_TIME)
    assert batch["audio"].shape == (3, N_MFCC, N_FRAMES)
    assert batch["emotion"].dtype == torch.long
    assert batch["emotion"].shape == (3,)


def test_collate_without_audio(cache):
    ds = EAVMultimodalDataset(cache_dir=cache, load_audio=False)
    batch = eav_collate([ds[0], ds[1]])
    assert "audio" not in batch
    assert batch["eeg"].shape == (2, N_CHANNELS, N_TIME)


def test_class_counts(cache):
    ds = EAVMultimodalDataset(cache_dir=cache)
    counts = ds.class_counts()
    assert sum(counts.values()) == len(ds)
    # The fixture cycles through emotions, so classes are balanced.
    assert len(set(counts.values())) == 1
