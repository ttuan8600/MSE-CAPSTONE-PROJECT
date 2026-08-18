"""Inference wrapper for the trained emotion-recognition model.

The previous implementation instantiated a placeholder network -- an EEG encoder,
an audio encoder, and a ``Linear(256, 128)`` fusion layer -- and called
``load_state_dict`` on it with a checkpoint produced by the cross-modal attention
fusion architecture. The parameter names never matched, so the API could not
serve the model of record at all; ``create_app`` caught the exception and ran
with ``predictor = None``, returning 503 from every prediction endpoint.

This version reconstructs the architecture recorded *in the checkpoint* and
validates the input shapes against what the model was trained on.

Two checkpoint layouts are supported:

``{"model": state_dict, "modality": ..., "eeg_channels": ..., "n_mfcc": ...}``
    Written by the current ``scripts/train_attention_fusion.py``. Self-describing.

``{"encoder": ..., "audio_encoder": ..., "attention_fusion": ..., "classifier": ...}``
    Written by the pre-correction training scripts. Loadable so that historical
    checkpoints can be inspected, but note that those models were trained on
    mis-indexed EEG (see ``docs/DATA_CORRECTIONS.md``) and their predictions are
    not meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.attention_fusion import CrossModalAttentionFusion
from src.models.eeg_encoder import AudioEncoder, EEGEncoder, EmotionClassifier
from src.models.feature_encoders import build_audio_encoder, build_eeg_encoder
from src.preprocessing.eav_labels import EMOTION_NAMES

#: Sampling rate and channel count the deployed model expects, matching the
#: preprocessing cache (see scripts/preprocess_eav.py).
EXPECTED_EEG_RATE_HZ = 125
EXPECTED_N_MFCC = 13


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint cannot be mapped onto a known architecture."""


class EmotionRecognitionModel(nn.Module):
    """The deployed network: EEG encoder + audio encoder + cross-modal fusion."""

    def __init__(
        self,
        modality: str = "multimodal",
        eeg_channels: int = 30,
        n_mfcc: int = EXPECTED_N_MFCC,
        latent_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 5,
        eeg_features: str = "raw",
        audio_features: str = "mfcc",
    ):
        super().__init__()
        self.modality = modality
        self.eeg_channels = eeg_channels
        self.n_mfcc = n_mfcc
        self.eeg_features = eeg_features
        self.audio_features = audio_features

        if modality in ("multimodal", "eeg"):
            self.eeg_encoder = build_eeg_encoder(
                eeg_features, eeg_channels, latent_dim
            )
        if modality in ("multimodal", "audio"):
            self.audio_encoder = build_audio_encoder(
                audio_features, n_mfcc, latent_dim
            )
        if modality == "multimodal":
            self.fusion = CrossModalAttentionFusion(
                latent_dim=latent_dim, num_heads=num_heads
            )
        self.classifier = EmotionClassifier(
            latent_dim=latent_dim, num_emotions=num_classes
        )

    def forward(self, eeg=None, audio=None):
        if self.modality == "eeg":
            latent = self.eeg_encoder(eeg)
        elif self.modality == "audio":
            latent = self.audio_encoder(audio)
        else:
            latent = self.fusion(self.eeg_encoder(eeg), self.audio_encoder(audio))
        return self.classifier(latent)


def _load_legacy_checkpoint(checkpoint: dict) -> EmotionRecognitionModel:
    """Reassemble a model saved as four separate per-module state dicts."""
    encoder_sd = checkpoint["encoder"]
    # Infer channel count from the first conv layer's weight: (out, in, kernel).
    eeg_channels = int(encoder_sd["conv1.weight"].shape[1])
    n_mfcc = int(checkpoint["audio_encoder"]["conv1.weight"].shape[1])

    model = EmotionRecognitionModel(
        modality="multimodal", eeg_channels=eeg_channels, n_mfcc=n_mfcc
    )
    model.eeg_encoder.load_state_dict(encoder_sd)
    model.audio_encoder.load_state_dict(checkpoint["audio_encoder"])
    model.fusion.load_state_dict(checkpoint["attention_fusion"])
    model.classifier.load_state_dict(checkpoint["classifier"])
    return model


class EmotionPredictor:
    """High-level interface for emotion prediction."""

    EMOTION_LABELS: List[str] = EMOTION_NAMES

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.model_path = Path(model_path)
        self.metadata: Dict = {}
        self.model = self._load_model()

    def _load_model(self) -> EmotionRecognitionModel:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        if not isinstance(checkpoint, dict):
            raise ModelLoadError(
                f"{self.model_path}: expected a dict checkpoint, got {type(checkpoint)}"
            )

        if "model" in checkpoint:
            model = EmotionRecognitionModel(
                modality=checkpoint.get("modality", "multimodal"),
                eeg_channels=checkpoint.get("eeg_channels", 30),
                n_mfcc=checkpoint.get("n_mfcc", EXPECTED_N_MFCC),
                eeg_features=checkpoint.get("eeg_features", "raw"),
                audio_features=checkpoint.get("audio_features", "mfcc"),
            )
            missing, unexpected = model.load_state_dict(
                checkpoint["model"], strict=False
            )
            if missing or unexpected:
                raise ModelLoadError(
                    f"{self.model_path}: state dict does not match architecture. "
                    f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
                )
            self.metadata = {
                "format": "unified",
                "modality": model.modality,
                "epoch": checkpoint.get("epoch"),
                "val_accuracy": checkpoint.get("val_acc"),
                "config": checkpoint.get("config", {}),
            }
        elif {"encoder", "audio_encoder", "attention_fusion", "classifier"} <= set(checkpoint):
            model = _load_legacy_checkpoint(checkpoint)
            self.metadata = {
                "format": "legacy",
                "modality": "multimodal",
                "warning": (
                    "This checkpoint predates the EEG axis correction; its EEG "
                    "stream was trained on mis-indexed data. See "
                    "docs/DATA_CORRECTIONS.md."
                ),
            }
        else:
            raise ModelLoadError(
                f"{self.model_path}: unrecognised checkpoint layout. "
                f"Top-level keys: {sorted(checkpoint)[:10]}"
            )

        model.to(self.device)
        model.eval()
        self.metadata["eeg_channels"] = model.eeg_channels
        self.metadata["n_mfcc"] = model.n_mfcc
        self.metadata["eeg_features"] = model.eeg_features
        self.metadata["audio_features"] = model.audio_features
        self.metadata["parameters"] = sum(p.numel() for p in model.parameters())
        return model

    # -- prediction --------------------------------------------------------

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)

    def _prepare_inputs(
        self, eeg_data: Optional[np.ndarray], audio_data: Optional[np.ndarray]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Validate and normalise inputs.

        Shape mismatches raise. The previous implementation substituted a zero
        tensor for missing audio, which silently produced a confident-looking
        prediction from half a model.
        """
        eeg_tensor = audio_tensor = None

        if self.model.modality in ("multimodal", "eeg"):
            if eeg_data is None:
                raise ValueError(
                    f"model modality '{self.model.modality}' requires EEG input"
                )
            eeg = np.asarray(eeg_data, dtype=np.float32)
            if eeg.ndim != 2:
                raise ValueError(f"EEG must be 2-D (features, time), got {eeg.shape}")
            if eeg.shape[0] != self.model.eeg_channels:
                unit = (
                    "band-power features" if self.model.eeg_features == "de"
                    else "channels"
                )
                raise ValueError(
                    f"EEG must have {self.model.eeg_channels} {unit}, "
                    f"got {eeg.shape[0]}"
                )
            if not np.isfinite(eeg).all():
                raise ValueError("EEG contains non-finite values")
            # Band-power features are already log-scaled and Euclidean-aligned;
            # the encoder's input BatchNorm standardises them, matching training.
            prepared = eeg if self.model.eeg_features == "de" else self._zscore(eeg)
            eeg_tensor = (
                torch.from_numpy(np.ascontiguousarray(prepared))
                .unsqueeze(0)
                .to(self.device)
            )

        if self.model.modality in ("multimodal", "audio"):
            unit = "log-mel bands" if self.model.audio_features == "mel" else "MFCC coefficients"
            if audio_data is None:
                raise ValueError(
                    f"model modality '{self.model.modality}' requires audio input "
                    f"({self.model.n_mfcc} {unit})"
                )
            audio = np.asarray(audio_data, dtype=np.float32)
            if audio.ndim != 2:
                raise ValueError(
                    f"Audio must be 2-D (features, frames), got {audio.shape}"
                )
            if audio.shape[0] != self.model.n_mfcc:
                raise ValueError(
                    f"Audio must have {self.model.n_mfcc} {unit}, "
                    f"got {audio.shape[0]}"
                )
            if not np.isfinite(audio).all():
                raise ValueError("Audio contains non-finite values")
            audio_tensor = (
                torch.from_numpy(np.ascontiguousarray(self._zscore(audio)))
                .unsqueeze(0)
                .to(self.device)
            )

        return eeg_tensor, audio_tensor

    def predict(
        self,
        eeg_data: Optional[np.ndarray] = None,
        audio_data: Optional[np.ndarray] = None,
    ) -> Dict:
        """Predict the emotion class for one sample.

        Returns a dict with the predicted label, its id, the confidence, the full
        probability vector and the input shapes that were accepted.
        """
        eeg_tensor, audio_tensor = self._prepare_inputs(eeg_data, audio_data)

        with torch.no_grad():
            logits = self.model(eeg=eeg_tensor, audio=audio_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        emotion_id = int(np.argmax(probs))
        return {
            "emotion": self.EMOTION_LABELS[emotion_id],
            "emotion_id": emotion_id,
            "confidence": float(probs[emotion_id]),
            "probabilities": {
                label: float(probs[i]) for i, label in enumerate(self.EMOTION_LABELS)
            },
            "input_shapes": {
                "eeg": list(np.shape(eeg_data)) if eeg_data is not None else None,
                "audio": list(np.shape(audio_data)) if audio_data is not None else None,
            },
        }

    def batch_predict(
        self, eeg_list: Optional[list] = None, audio_list: Optional[list] = None
    ) -> list:
        """Predict for several samples. Per-sample errors are returned, not raised."""
        if eeg_list is not None:
            n = len(eeg_list)
        elif audio_list is not None:
            n = len(audio_list)
        else:
            raise ValueError("provide eeg_list, audio_list, or both")
        if eeg_list is not None and audio_list is not None and len(audio_list) != n:
            raise ValueError(
                f"eeg_list has {n} entries but audio_list has {len(audio_list)}"
            )

        results = []
        for i in range(n):
            try:
                results.append(
                    self.predict(
                        eeg_list[i] if eeg_list is not None else None,
                        audio_list[i] if audio_list is not None else None,
                    )
                )
            except (ValueError, RuntimeError) as exc:
                results.append({"error": str(exc), "sample_index": i})
        return results

    def info(self) -> Dict:
        """Describe the loaded model, for the API's ``/model-info`` endpoint."""
        return {
            "checkpoint": str(self.model_path),
            "device": str(self.device),
            "emotions": self.EMOTION_LABELS,
            "expected_eeg_rate_hz": EXPECTED_EEG_RATE_HZ,
            **self.metadata,
        }
