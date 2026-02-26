"""Preprocessing utilities for EmoAI framework.

This package will contain functions for:
- EEG segmentation and artifact removal
- Speech signal processing (spectrograms, ZFF, etc.)
- Data loading for FACED and EAV datasets
"""

from .eeg import *
from .speech import *
from .data_loader import (
    FAEDDataset,
    EAVMultimodalDataset,
    create_faced_dataloader,
    create_eav_dataloader,
)

__all__ = [
    "FAEDDataset",
    "EAVMultimodalDataset",
    "create_faced_dataloader",
    "create_eav_dataloader",
]
