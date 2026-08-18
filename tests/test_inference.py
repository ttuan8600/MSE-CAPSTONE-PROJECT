"""Tests for the inference wrapper.

The defect these guard: the API's model class did not match the architecture in
the checkpoint, so ``load_state_dict`` could never succeed and the service ran
with no model at all. A round-trip test -- save from the real architecture, load
through the predictor -- is the only thing that catches that class of mismatch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.inference import (
    EmotionPredictor,
    EmotionRecognitionModel,
    ModelLoadError,
)

N_CHANNELS, N_TIME = 30, 250
N_MFCC, N_FRAMES = 13, 60


def save_unified(path: Path, modality: str = "multimodal") -> EmotionRecognitionModel:
    model = EmotionRecognitionModel(
        modality=modality, eeg_channels=N_CHANNELS, n_mfcc=N_MFCC
    )
    torch.save(
        {
            "model": model.state_dict(),
            "modality": modality,
            "eeg_channels": N_CHANNELS,
            "n_mfcc": N_MFCC,
            "epoch": 12,
            "val_acc": 0.41,
            "config": {"seed": 42},
        },
        path,
    )
    return model


def test_round_trip_multimodal_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    saved = save_unified(path)
    predictor = EmotionPredictor(str(path))

    # Weights must be identical, not merely shape-compatible.
    for (name, a), (_, b) in zip(
        saved.state_dict().items(), predictor.model.state_dict().items()
    ):
        assert torch.allclose(a, b), f"weight mismatch in {name}"


def test_predict_returns_a_normalised_distribution(tmp_path):
    path = tmp_path / "model.pt"
    save_unified(path)
    predictor = EmotionPredictor(str(path))

    result = predictor.predict(
        np.random.randn(N_CHANNELS, N_TIME), np.random.randn(N_MFCC, N_FRAMES)
    )
    assert result["emotion"] in EmotionPredictor.EMOTION_LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-5) == 1.0
    assert result["probabilities"][result["emotion"]] == result["confidence"]


def test_missing_audio_is_rejected_not_zero_filled(tmp_path):
    """The old code substituted zeros and returned a confident prediction."""
    path = tmp_path / "model.pt"
    save_unified(path)
    predictor = EmotionPredictor(str(path))

    with pytest.raises(ValueError, match="requires audio"):
        predictor.predict(np.random.randn(N_CHANNELS, N_TIME), None)


def test_missing_eeg_is_rejected(tmp_path):
    path = tmp_path / "model.pt"
    save_unified(path)
    predictor = EmotionPredictor(str(path))

    with pytest.raises(ValueError, match="requires EEG"):
        predictor.predict(None, np.random.randn(N_MFCC, N_FRAMES))


def test_wrong_channel_count_is_rejected(tmp_path):
    path = tmp_path / "model.pt"
    save_unified(path)
    predictor = EmotionPredictor(str(path))

    with pytest.raises(ValueError, match="must have 30 channels"):
        predictor.predict(
            np.random.randn(28, N_TIME), np.random.randn(N_MFCC, N_FRAMES)
        )


def test_non_finite_input_is_rejected(tmp_path):
    path = tmp_path / "model.pt"
    save_unified(path)
    predictor = EmotionPredictor(str(path))

    eeg = np.random.randn(N_CHANNELS, N_TIME)
    eeg[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        predictor.predict(eeg, np.random.randn(N_MFCC, N_FRAMES))


def test_eeg_only_checkpoint_needs_no_audio(tmp_path):
    path = tmp_path / "eeg_model.pt"
    save_unified(path, modality="eeg")
    predictor = EmotionPredictor(str(path))

    result = predictor.predict(np.random.randn(N_CHANNELS, N_TIME))
    assert result["emotion"] in EmotionPredictor.EMOTION_LABELS


def test_legacy_four_module_checkpoint_loads(tmp_path):
    """Historical checkpoints remain inspectable, flagged with a warning."""
    from src.models.attention_fusion import CrossModalAttentionFusion
    from src.models.eeg_encoder import AudioEncoder, EEGEncoder, EmotionClassifier

    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "encoder": EEGEncoder(in_channels=28, latent_dim=128).state_dict(),
            "audio_encoder": AudioEncoder(n_mfcc=13, latent_dim=128).state_dict(),
            "attention_fusion": CrossModalAttentionFusion(
                latent_dim=128, num_heads=4
            ).state_dict(),
            "classifier": EmotionClassifier(
                latent_dim=128, num_emotions=5
            ).state_dict(),
        },
        path,
    )

    predictor = EmotionPredictor(str(path))
    assert predictor.metadata["format"] == "legacy"
    assert "DATA_CORRECTIONS" in predictor.metadata["warning"]
    # Channel count inferred from the checkpoint, not assumed.
    assert predictor.metadata["eeg_channels"] == 28


def test_unrecognised_checkpoint_raises(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"something_else": torch.zeros(3)}, path)
    with pytest.raises(ModelLoadError, match="unrecognised checkpoint layout"):
        EmotionPredictor(str(path))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmotionPredictor(str(tmp_path / "absent.pt"))


def test_batch_predict_reports_per_sample_errors(tmp_path):
    path = tmp_path / "model.pt"
    save_unified(path)
    predictor = EmotionPredictor(str(path))

    results = predictor.batch_predict(
        eeg_list=[
            np.random.randn(N_CHANNELS, N_TIME),
            np.random.randn(7, N_TIME),          # wrong channel count
        ],
        audio_list=[
            np.random.randn(N_MFCC, N_FRAMES),
            np.random.randn(N_MFCC, N_FRAMES),
        ],
    )
    assert "emotion" in results[0]
    assert "error" in results[1]
    assert results[1]["sample_index"] == 1


def test_batch_predict_rejects_mismatched_list_lengths(tmp_path):
    path = tmp_path / "model.pt"
    save_unified(path)
    predictor = EmotionPredictor(str(path))

    with pytest.raises(ValueError, match="audio_list"):
        predictor.batch_predict(
            eeg_list=[np.random.randn(N_CHANNELS, N_TIME)] * 2,
            audio_list=[np.random.randn(N_MFCC, N_FRAMES)],
        )


def test_info_reports_the_loaded_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    save_unified(path)
    info = EmotionPredictor(str(path)).info()

    assert info["modality"] == "multimodal"
    assert info["eeg_channels"] == N_CHANNELS
    assert info["epoch"] == 12
    assert info["parameters"] > 0
    assert info["emotions"] == EmotionPredictor.EMOTION_LABELS
