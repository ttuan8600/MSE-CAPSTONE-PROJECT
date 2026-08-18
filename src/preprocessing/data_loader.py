"""Data loaders for FACED and EAV emotion recognition datasets.

The EAV dataset class lives in :mod:`src.preprocessing.eav_dataset` and is
re-exported here for backwards compatibility with existing imports. It is now
backed by the preprocessing cache built by ``scripts/preprocess_eav.py``; the
previous version read the raw ``.mat`` files with an incorrect axis assumption.
See ``docs/DATA_CORRECTIONS.md``.
"""

import os
import pickle
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# Import EEG utilities
from .eeg import load_eeg
from .eav_dataset import EAVCacheMissing, EAVMultimodalDataset, eav_collate

__all__ = [
    "FAEDDataset",
    "EAVMultimodalDataset",
    "EAVCacheMissing",
    "eav_collate",
    "create_faced_dataloader",
    "create_eav_dataloader",
]


class FAEDDataset(Dataset):
    """PyTorch Dataset for FACED EEG pre-training.

    FACED dataset structure:
    - Each subject file: (28 channels, 32 emotion trials, 7500 time steps)
    - Segmented into windows for training

    Parameters
    ----------
    data_dir : str
        Root directory containing subject pickle files.
    window_size : int, default=512
        Time steps per window.
    stride : int, default=256
        Stride between windows (for overlapping segments).
    subjects : List[int], optional
        Specific subject indices to load. If None, loads all.
    normalize : bool, default=True
        Apply z-score normalization per channel.
    contrastive : bool, default=False
        When True, dataset returns a pair of augmented views for contrastive learning.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 512,
        stride: int = 256,
        subjects: Optional[List[int]] = None,
        normalize: bool = True,
        contrastive: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.contrastive = contrastive

        # Find all pickle files. The FACED dataset is typically stored in a
        # nested ``Processed_data/Processed_data`` folder, but we allow a few
        # common layouts for convenience.
        if self.data_dir.is_dir():
            pkl_files = sorted(self.data_dir.glob("sub*.pkl"))
            if not pkl_files:
                # Support a nested 'Processed_data' folder layout
                nested = self.data_dir / "Processed_data"
                if nested.is_dir():
                    self.data_dir = nested
                    pkl_files = sorted(self.data_dir.glob("sub*.pkl"))
        else:
            raise FileNotFoundError(f"FACED dataset directory not found: {self.data_dir!r}")

        # Fallback: search recursively in case the folder structure differs
        if not pkl_files:
            pkl_files = sorted(self.data_dir.rglob("sub*.pkl"))

        if subjects is not None:
            # Filter to specific subjects
            pkl_files = [f for f in pkl_files
                         if int(f.stem[3:]) in subjects]

        if not pkl_files:
            raise FileNotFoundError(
                f"No FACED .pkl files found under {self.data_dir!r}. "
                "Expecting files named like sub000.pkl, sub001.pkl, etc."
            )

        self.pkl_files = pkl_files
        self.windows = []
        self.subject_labels = []  # Subject ID label per window (pre-training target)
        # Pre-compute windows for fast access
        self._build_windows()
    
    def _build_windows(self):
        """Segment each subject file into windows."""
        for pkl_file in self.pkl_files:
            subject_id = int(pkl_file.stem[3:])
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)  # Shape: (28, 32, 7500)

            # Process each trial
            for trial_idx in range(data.shape[1]):
                trial_data = data[:, trial_idx, :]  # (28, 7500)

                # Create sliding windows
                for start in range(0, trial_data.shape[1] - self.window_size, self.stride):
                    end = start + self.window_size
                    window = trial_data[:, start:end]  # (28, window_size)

                    if self.normalize:
                        # Z-score normalization per channel
                        window = (window - window.mean(axis=1, keepdims=True)) / \
                                (window.std(axis=1, keepdims=True) + 1e-8)

                    self.windows.append(window)
                    self.subject_labels.append(subject_id)

    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.windows)
    
    def _augment_window(self, window: np.ndarray) -> np.ndarray:
        """Apply simple augmentations to an EEG window for contrastive learning."""
        # Add small Gaussian noise
        noise = np.random.normal(scale=0.01, size=window.shape).astype(np.float32)
        aug = window + noise

        # Random time shift (circular)
        shift = np.random.randint(-10, 11)
        if shift != 0:
            aug = np.roll(aug, shift, axis=1)

        return aug

    def __getitem__(self, idx: int):
        """Get a window.

        Returns
        -------
        (window, subject_id) or (view1, view2)
            If ``contrastive`` is False, returns a single window and the subject id.
            If ``contrastive`` is True, returns two augmented views for contrastive loss.
        """
        window = self.windows[idx].astype(np.float32)
        subject_label = int(self.subject_labels[idx])

        if self.contrastive:
            view1 = self._augment_window(window)
            view2 = self._augment_window(window)
            return (
                torch.from_numpy(view1),
                torch.from_numpy(view2),
            )

        return (
            torch.from_numpy(window),
            subject_label
        )


def create_faced_dataloader(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 512,
    stride: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    subjects: Optional[List[int]] = None,
    val_split: float = 0.0,
    seed: int = 42,
    contrastive: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], FAEDDataset]:
    """Create DataLoaders for FACED pre-training.

    Parameters
    ----------
    data_dir : str
        Path to FACED Processed_data folder.
    batch_size : int
        Batch size.
    window_size : int
        Time steps per window.
    stride : int
        Stride between windows.
    shuffle : bool
        Whether to shuffle the data.
    num_workers : int
        Number of data loading workers.
    subjects : List[int], optional
        Specific subjects to load.
    val_split : float, default=0.0
        Fraction of the dataset to reserve for validation.
    seed : int, default=42
        Random seed for deterministic train/val split.
    contrastive : bool, default=False
        If True, returns augmented view pairs for contrastive training.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader or None
    dataset : FAEDDataset
    """
    dataset = FAEDDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        subjects=subjects,
        contrastive=contrastive,
    )

    pin_memory = torch.cuda.is_available()

    # Optionally split train/validation
    val_loader = None
    if val_split and 0.0 < val_split < 1.0:
        generator = torch.Generator().manual_seed(seed)
        train_size = int(len(dataset) * (1.0 - val_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, dataset


def create_eav_dataloader(
    cache_dir: str = "data/processed/eav",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    subjects: Optional[List[int]] = None,
    load_audio: bool = True,
) -> Tuple[DataLoader, EAVMultimodalDataset]:
    """Create a DataLoader over the preprocessed EAV cache.

    Parameters
    ----------
    cache_dir : str
        Directory produced by ``scripts/preprocess_eav.py``.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle the data.
    num_workers : int
        Number of data loading workers.
    subjects : List[int], optional
        Restrict to these subject ids. If None, loads all 42.
    load_audio : bool
        Whether to include the audio modality.

    Returns
    -------
    dataloader : DataLoader
    dataset : EAVMultimodalDataset
    """
    dataset = EAVMultimodalDataset(
        cache_dir=cache_dir,
        subjects=subjects,
        load_audio=load_audio,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=eav_collate,
    )

    return dataloader, dataset
