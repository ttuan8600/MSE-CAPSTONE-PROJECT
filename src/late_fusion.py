"""Late fusion of two unimodal models by averaging their output distributions.

This is the project's best-performing configuration: **69.55%** under 7-fold
subject-wise cross-validation, against 64.48% for audio alone (+5.07pp, 95% CI
$[+2.81, +7.40]$, $p < 0.0001$) and 64.12% for the trained cross-modal attention
fusion model.

It has **no parameters of its own**. Two independently trained models each emit a
probability vector and the two are averaged. Every parameterised alternative
tested -- a single fitted weight, per-class weights, per-sample confidence
gating, an 850,417-parameter learned attention module, and a sequence-level
attention module -- scored lower. On a corpus where the failure mode is transfer
to unseen people, a combiner with free parameters fits the training subjects'
modality preferences and does not carry them to new ones.

The EEG model is trained with subject-adversarial gradient reversal, which is
worth about +0.6pp on its own but roughly +1.5pp inside this fusion: making the
EEG representation subject-invariant is what makes it combinable.

See ``docs/CHANGELOG.md`` for the full comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.inference import EmotionPredictor
from src.preprocessing.eav_labels import EMOTION_NAMES


class LateFusionPredictor:
    """Average the probability vectors of an EEG model and an audio model.

    Parameters
    ----------
    eeg_model_path, audio_model_path:
        Checkpoints written by ``scripts/train_attention_fusion.py``. The first
        must be an EEG-modality model, the second an audio-modality model;
        passing them the wrong way round raises rather than producing a
        confidently wrong prediction.
    weight:
        Weight on the audio distribution. Defaults to 0.5 --- the equal average
        that measured best. It is exposed for reproducing the weighted-rule
        ablation, not because tuning it is recommended: every fitted weight
        tested transferred worse than 0.5.
    """

    EMOTION_LABELS: List[str] = EMOTION_NAMES

    def __init__(
        self,
        eeg_model_path: str | Path,
        audio_model_path: str | Path,
        weight: float = 0.5,
        device: str = "cpu",
    ):
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.weight = float(weight)

        self.eeg = EmotionPredictor(str(eeg_model_path), device=device)
        self.audio = EmotionPredictor(str(audio_model_path), device=device)

        if self.eeg.model.modality != "eeg":
            raise ValueError(
                f"{eeg_model_path} is a '{self.eeg.model.modality}' model, "
                f"expected 'eeg' -- are the two checkpoints swapped?"
            )
        if self.audio.model.modality != "audio":
            raise ValueError(
                f"{audio_model_path} is a '{self.audio.model.modality}' model, "
                f"expected 'audio' -- are the two checkpoints swapped?"
            )

    # -- prediction --------------------------------------------------------

    def predict(self, eeg_data: np.ndarray, audio_data: np.ndarray) -> Dict:
        """Predict from one EEG trial and its matching audio clip.

        Both modalities are required. A late fusion given only one stream is not
        a fusion, and substituting a uniform distribution for the missing one
        would silently halve the model while still returning a confident-looking
        answer.
        """
        if eeg_data is None or audio_data is None:
            raise ValueError(
                "late fusion requires both EEG and audio; use EmotionPredictor "
                "with a unimodal checkpoint if only one stream is available"
            )

        eeg_result = self.eeg.predict(eeg_data=eeg_data)
        audio_result = self.audio.predict(audio_data=audio_data)

        eeg_probs = np.array(
            [eeg_result["probabilities"][k] for k in self.EMOTION_LABELS]
        )
        audio_probs = np.array(
            [audio_result["probabilities"][k] for k in self.EMOTION_LABELS]
        )
        fused = self.weight * audio_probs + (1.0 - self.weight) * eeg_probs

        emotion_id = int(np.argmax(fused))
        return {
            "emotion": self.EMOTION_LABELS[emotion_id],
            "emotion_id": emotion_id,
            "confidence": float(fused[emotion_id]),
            "probabilities": {
                label: float(fused[i]) for i, label in enumerate(self.EMOTION_LABELS)
            },
            "components": {
                "eeg": {
                    "emotion": eeg_result["emotion"],
                    "confidence": eeg_result["confidence"],
                },
                "audio": {
                    "emotion": audio_result["emotion"],
                    "confidence": audio_result["confidence"],
                },
            },
        }

    def info(self) -> Dict:
        return {
            "type": "late_fusion",
            "rule": f"{self.weight:.2f}*audio + {1 - self.weight:.2f}*eeg",
            "fusion_parameters": 0,
            "emotions": self.EMOTION_LABELS,
            "eeg": self.eeg.info(),
            "audio": self.audio.info(),
        }
