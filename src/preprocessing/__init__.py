"""Preprocessing utilities for EmoAI framework.

This package will contain functions for:
- EEG segmentation and artifact removal
- Speech signal processing (spectrograms, ZFF, etc.)
- Data loading for FACED and EAV datasets
"""

from .eeg import *
from .speech import *
from .eav_labels import (
    EMOTION_NAMES,
    EMOTION_TO_CLASS_INDEX,
    decode_label_matrix,
    emotion_class_indices,
)
from .eav_io import EAVDataError, load_subject_labels, load_subject_segments
from .data_loader import (
    FAEDDataset,
    EAVCacheMissing,
    EAVMultimodalDataset,
    eav_collate,
    create_faced_dataloader,
    create_eav_dataloader,
)

__all__ = [
    "FAEDDataset",
    "EAVMultimodalDataset",
    "EAVCacheMissing",
    "EAVDataError",
    "eav_collate",
    "create_faced_dataloader",
    "create_eav_dataloader",
    "EMOTION_NAMES",
    "EMOTION_TO_CLASS_INDEX",
    "decode_label_matrix",
    "emotion_class_indices",
    "load_subject_labels",
    "load_subject_segments",
]
