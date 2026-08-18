"""Multimodal EAV dataset backed by the preprocessing cache.

This replaces an earlier implementation that had three defects, all of which
invalidated the results computed with it:

1. **Wrong EEG axis.** It read ``seg[0, :, :]`` believing the array to be
   ``(segments, channels, time)``. The array is ``(time, channels, trials)``, so
   it extracted one 2 ms time-point and reinterpreted the trial axis as time.

2. **One EEG tensor shared by 100 samples.** It paired each subject's single
   ``.mat`` file with every one of that subject's 100 ``.wav`` files. Because the
   EEG read was deterministic, all 100 samples received a byte-identical EEG
   tensor while carrying five different emotion labels. The EEG stream therefore
   could not contribute any label information, and every reported "multimodal
   fusion" result was in fact audio-only.

3. **Silent noise substitution.** Any EEG load failure returned
   ``np.random.randn(28, 200)`` while keeping the real label; audio failures
   became zero tensors. Corrupt inputs were indistinguishable from valid ones and
   the proportion of noise in any run was unknowable.

The corrected dataset indexes trials on the correct axis, takes labels from the
dataset's own ``label`` matrix rather than from parsed filenames, pairs each
audio clip with *its own* EEG trial via the filename's trial index, and treats
every load failure as fatal.

See ``docs/DATA_CORRECTIONS.md`` for the full account.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .eav_labels import EMOTION_NAMES, EMOTION_TO_CLASS_INDEX

DEFAULT_CACHE_DIR = Path("data/processed/eav")
EXPECTED_CACHE_VERSION = 3

#: EEG representations. ``raw`` is the filtered 125 Hz time series; ``de`` is
#: Euclidean-aligned differential-entropy band power, which exists because the
#: raw representation memorised its training subjects (99.75% train / 42.17% val).
EEG_FEATURES = ("raw", "de")

#: Audio representations. ``mfcc`` is the original 13 coefficients; ``mel`` is a
#: 64-band log-mel spectrogram.
AUDIO_FEATURES = ("mfcc", "mel")

_EEG_FILE_KEY = {"raw": "eeg_file", "de": "de_file"}
_AUDIO_FILE_KEY = {"mfcc": "mfcc_file", "mel": "mel_file"}


class EAVCacheMissing(FileNotFoundError):
    """Raised when the preprocessing cache has not been built."""


class EAVMultimodalDataset(Dataset):
    """EEG + audio samples from the EAV dataset, one sample per Speaking trial.

    Each sample is a genuine 20 s EEG recording paired with the audio actually
    recorded during that same trial.

    Parameters
    ----------
    cache_dir:
        Directory written by ``scripts/preprocess_eav.py``.
    subjects:
        Restrict to these subject ids. ``None`` loads all 42.
    normalize_eeg, normalize_audio:
        Apply per-channel z-scoring to each trial.
    load_audio:
        When ``False``, the audio tensor is omitted from the returned dict. Used
        for the EEG-only ablation. Note this is *not* the same as substituting
        zeros -- a caller that wants an EEG-only model must build one.

    Attributes
    ----------
    subject_ids : np.ndarray
        Subject id per sample, in dataset order. Pass this to
        ``src.data.splits.subject_independent_split``.
    labels : np.ndarray
        Class index per sample, in dataset order.
    """

    EMOTION_MAP = EMOTION_TO_CLASS_INDEX
    EMOTION_NAMES = EMOTION_NAMES

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        subjects: Optional[List[int]] = None,
        normalize_eeg: bool = True,
        normalize_audio: bool = True,
        load_audio: bool = True,
        eeg_features: str = "raw",
        audio_features: str = "mfcc",
    ):
        if eeg_features not in EEG_FEATURES:
            raise ValueError(f"eeg_features must be one of {EEG_FEATURES}")
        if audio_features not in AUDIO_FEATURES:
            raise ValueError(f"audio_features must be one of {AUDIO_FEATURES}")

        self.cache_dir = Path(cache_dir)
        self.normalize_eeg = normalize_eeg
        self.normalize_audio = normalize_audio
        self.load_audio = load_audio
        self.eeg_features = eeg_features
        self.audio_features = audio_features

        # Band-power features are already log-scaled and Euclidean-aligned;
        # per-trial z-scoring would remove the band-power magnitudes that carry
        # the signal. The encoder normalises them with an input BatchNorm.
        if eeg_features == "de":
            self.normalize_eeg = False

        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise EAVCacheMissing(
                f"EAV cache not found at {manifest_path}.\n"
                f"Build it once with:  python scripts/preprocess_eav.py"
            )

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = self.manifest.get("cache_version")
        if version != EXPECTED_CACHE_VERSION:
            raise EAVCacheMissing(
                f"EAV cache at {self.cache_dir} is version {version}, expected "
                f"{EXPECTED_CACHE_VERSION}. Rebuild with: "
                f"python scripts/preprocess_eav.py --force"
            )

        wanted = set(subjects) if subjects is not None else None

        self.samples: List[dict] = []
        self._eeg_arrays: Dict[int, np.ndarray] = {}
        self._mfcc_arrays: Dict[int, np.ndarray] = {}

        for entry in self.manifest["subjects"]:
            sid = entry["subject_id"]
            if wanted is not None and sid not in wanted:
                continue
            # memory-mapped: the cache is several GB and never fully resident.
            self._eeg_arrays[sid] = np.load(
                self.cache_dir / entry[_EEG_FILE_KEY[eeg_features]], mmap_mode="r"
            )
            if self.load_audio:
                self._mfcc_arrays[sid] = np.load(
                    self.cache_dir / entry[_AUDIO_FILE_KEY[audio_features]],
                    mmap_mode="r",
                )
            self.samples.extend(entry["samples"])

        if not self.samples:
            raise ValueError(
                f"no samples selected from {self.cache_dir} "
                f"(subjects={subjects!r})"
            )

        self.subject_ids = np.array([s["subject_id"] for s in self.samples])
        self.labels = np.array([s["label"] for s in self.samples], dtype=np.int64)

    # -- introspection -----------------------------------------------------

    @property
    def n_subjects(self) -> int:
        return int(np.unique(self.subject_ids).size)

    @property
    def eeg_shape(self) -> tuple:
        """``(channels, time)`` of a single EEG trial."""
        first = next(iter(self._eeg_arrays.values()))
        return tuple(first.shape[1:])

    @property
    def audio_shape(self) -> Optional[tuple]:
        """``(n_mfcc, frames)`` of a single audio clip, or ``None``."""
        if not self._mfcc_arrays:
            return None
        first = next(iter(self._mfcc_arrays.values()))
        return tuple(first.shape[1:])

    def class_counts(self) -> Dict[str, int]:
        counts = np.bincount(self.labels, minlength=len(EMOTION_NAMES))
        return {EMOTION_NAMES[i]: int(counts[i]) for i in range(len(EMOTION_NAMES))}

    def describe(self) -> str:
        lines = [
            f"EAVMultimodalDataset({self.cache_dir})",
            f"  samples   : {len(self)}",
            f"  subjects  : {self.n_subjects}",
            f"  eeg       : {self.eeg_features} {self.eeg_shape}",
            f"  audio     : {self.audio_features} {self.audio_shape}",
            f"  classes   : {self.class_counts()}",
        ]
        return "\n".join(lines)

    # -- torch Dataset protocol -------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        """Z-score each channel of a ``(channels, time)`` array."""
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        return (x - mean) / (std + 1e-8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sid = sample["subject_id"]

        # np.array (not asarray) forces a writable copy out of the memory map,
        # which torch.from_numpy requires.
        eeg = np.array(self._eeg_arrays[sid][sample["eeg_row"]], dtype=np.float32)
        if self.normalize_eeg:
            eeg = self._zscore(eeg)

        item: Dict[str, torch.Tensor] = {
            "eeg": torch.from_numpy(eeg),
            "emotion": int(sample["label"]),
            "subject_id": int(sid),
            "trial_index": int(sample["trial_index"]),
        }

        if self.load_audio:
            mfcc = np.array(
                self._mfcc_arrays[sid][sample["mfcc_row"]], dtype=np.float32
            )
            if self.normalize_audio:
                mfcc = self._zscore(mfcc)
            item["audio"] = torch.from_numpy(mfcc)

        return item


def eav_collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Collate EAV samples into batched tensors.

    Unlike the previous implementation this does not swallow exceptions. Every
    sample in the cache has identical shape, so a stacking failure means the
    cache is corrupt and must surface rather than silently degrade the batch.
    """
    out: Dict[str, torch.Tensor] = {
        "eeg": torch.stack([b["eeg"] for b in batch]),
        "emotion": torch.tensor([b["emotion"] for b in batch], dtype=torch.long),
        "subject_id": torch.tensor([b["subject_id"] for b in batch], dtype=torch.long),
        "trial_index": torch.tensor([b["trial_index"] for b in batch], dtype=torch.long),
    }
    if "audio" in batch[0]:
        out["audio"] = torch.stack([b["audio"] for b in batch])
    return out
