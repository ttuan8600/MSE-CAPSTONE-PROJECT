"""Model definitions for EmoAI (CNN-LSTM, CMA, classifiers, etc.)."""

from .eeg_encoder import EEGEncoder, EEGEncoderLSTM, EmotionClassifier

__all__ = [
    "EEGEncoder",
    "EEGEncoderLSTM",
    "EmotionClassifier",
]