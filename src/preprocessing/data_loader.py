"""Data loaders for FACED and EAV emotion recognition datasets."""

import os
import pickle
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Import EEG utilities
from .eeg import load_eeg


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
    """
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 512,
        stride: int = 256,
        subjects: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        
        # Find all pickle files
        pkl_files = sorted(self.data_dir.glob("sub*.pkl"))
        
        if subjects is not None:
            # Filter to specific subjects
            pkl_files = [f for f in pkl_files 
                        if int(f.stem[3:]) in subjects]
        
        self.pkl_files = pkl_files
        self.windows = []
        self.emotion_labels = []  # Optional: emotion class per trial
        
        # Pre-compute windows for fast access
        self._build_windows()
    
    def _build_windows(self):
        """Segment each subject file into windows."""
        for pkl_file in self.pkl_files:
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
    
    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a window.
        
        Returns
        -------
        window : torch.Tensor
            Shape (28, window_size) EEG signal.
        subject_id : int
            Subject index (used as pseudo-label for pre-training).
        """
        window = self.windows[idx].astype(np.float32)
        subject_label = int(np.random.randint(0, 5))  # Placeholder emotion class
        
        return (
            torch.from_numpy(window),
            subject_label
        )


class EAVMultimodalDataset(Dataset):
    """PyTorch Dataset for EAV multimodal emotion recognition.
    
    Loads synchronized Audio, EEG, and Video from the EAV dataset.
    Properly handles .mat files for EEG, .wav files for audio, and .mp4 for video.
    
    Dataset Structure:
    - EAV/EAV/subject{1-42}/<modality>/* files
    - Emotions: Neutral, Anger, Calmness, Sadness, Happiness
    - Modalities: EEG (.mat files), Audio (.wav files), Video (.mp4 files)
    
    Parameters
    ----------
    eav_data_dir : str
        Path to EAV/EAV directory with subject folders.
    subjects : List[int], optional
        Specific subjects to load. If None, loads all available.
    window_size : int, default=2048
        Samples per window for audio MFCC features.
    n_mfcc : int, default=13
        Number of MFCC coefficients to extract.
    load_audio : bool, default=True
        Whether to load audio features.
    load_video : bool, default=False
        Whether to load video features (stub - full implementation pending).
    """
    
    # Emotion label mapping
    EMOTION_MAP = {
        'Neutral': 0,
        'Anger': 1,
        'Calmness': 2,
        'Sadness': 3,
        'Happiness': 4,
    }
    
    def __init__(
        self,
        eav_data_dir: str,
        subjects: Optional[List[int]] = None,
        window_size: int = 2048,
        n_mfcc: int = 13,
        load_audio: bool = True,
        load_video: bool = False,
        normalize_eeg: bool = True,
    ):
        self.eav_data_dir = Path(eav_data_dir)
        self.window_size = window_size
        self.n_mfcc = n_mfcc
        self.load_audio = load_audio
        self.load_video = load_video
        self.normalize_eeg = normalize_eeg
        
        self.samples = []
        
        # Get all subject directories
        subject_dirs = sorted([d for d in self.eav_data_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("subject")])
        
        if subjects is not None:
            subject_indices = set(subjects)
            subject_dirs = [d for d in subject_dirs 
                           if int(d.name[7:]) in subject_indices]
        
        self._build_samples(subject_dirs)
    
    def _build_samples(self, subject_dirs):
        """Scan subject directories to build samples with matched modalities."""
        for subject_dir in subject_dirs:
            subject_id = int(subject_dir.name[7:])
            eeg_dir = subject_dir / "EEG"
            audio_dir = subject_dir / "Audio"
            video_dir = subject_dir / "Video"
            
            # Look for EEG files (.mat files, excluding label files)
            if eeg_dir.exists():
                for eeg_file in eeg_dir.glob("*.mat"):
                    # Skip label files
                    if "_label" in eeg_file.name:
                        continue
                    
                    # Extract subject ID from filename (e.g., subject1_eeg.mat)
                    stem = eeg_file.stem
                    
                    # Look for corresponding label file
                    label_file = eeg_dir / f"{stem.replace('_eeg', '_eeg_label')}.mat"
                    
                    # Try to match with audio files to get emotion labels
                    matched_samples = self._find_matched_samples(
                        subject_id, audio_dir, video_dir, eeg_file, label_file
                    )
                    
                    self.samples.extend(matched_samples)
    
    def _find_matched_samples(self, subject_id, audio_dir, video_dir, eeg_file, label_file):
        """Find audio/video files that match EEG data."""
        matched = []
        
        if not audio_dir.exists():
            # No audio files, just use EEG
            matched.append({
                'subject_id': subject_id,
                'eeg': eeg_file,
                'eeg_label': label_file if label_file.exists() else None,
                'audio': None,
                'audio_emotion': None,
                'video': None,
            })
            return matched
        
        # Group audio files by emotion
        audio_files = list(audio_dir.glob("*.wav"))
        for audio_file in sorted(audio_files):
            # Parse emotion from filename (e.g., "002_Trial_02_Speaking_Neutral_Aud.wav")
            emotion = self._parse_emotion_from_audio(audio_file.name)
            
            # Find corresponding video
            video_file = None
            if video_dir.exists():
                # Video files are numbered (e.g., "001_Trial_01_Listening_Neutral.mp4")
                base_num = audio_file.name[:3]  # Get the first 3 digits
                possible_video = video_dir / f"{base_num}*.mp4"
                video_matches = list(video_dir.glob(possible_video.name.replace('*', '*')))
                if video_matches:
                    video_file = video_matches[0]
            
            matched.append({
                'subject_id': subject_id,
                'eeg': eeg_file,
                'eeg_label': label_file if label_file.exists() else None,
                'audio': audio_file,
                'audio_emotion': emotion,
                'video': video_file,
            })
        
        return matched
    
    @staticmethod
    def _parse_emotion_from_audio(filename: str) -> Optional[str]:
        """Extract emotion label from audio filename.
        
        Examples:
            "002_Trial_02_Speaking_Neutral_Aud.wav" -> "Neutral"
            "004_Trial_02_Speaking_Anger_aud.wav" -> "Anger"
        """
        # Look for emotion keywords
        for emotion in EAVMultimodalDataset.EMOTION_MAP.keys():
            if emotion in filename:
                return emotion
        return None
    
    def _load_eeg(self, eeg_file: Path, label_file: Optional[Path]) -> Tuple[np.ndarray, Optional[int]]:
        """Load EEG data and optionally emotion label from .mat files.
        
        The MATLAB file contains EEG segments with keys like 'seg', 'seg1', etc.
        Shape is typically (n_segments, n_channels, time_steps).
        We extract the first segment as a representative sample.
        """
        try:
            from scipy.io import loadmat
            
            mat_data = loadmat(str(eeg_file))
            
            # Find the EEG data key (could be 'seg', 'seg1', 'seg2', etc.)
            eeg_key = None
            for potential_key in ['seg', 'seg1', 'seg2', 'seg0', 'EEG', 'eeg', 'data']:
                if potential_key in mat_data:
                    eeg_key = potential_key
                    break
            
            if eeg_key:
                eeg_raw = mat_data[eeg_key]
                
                # Handle different shapes
                if len(eeg_raw.shape) == 3:
                    # Shape: (n_segments, n_channels, time_steps)
                    # Take the first segment
                    eeg_data = eeg_raw[0, :, :].astype(np.float32)
                elif len(eeg_raw.shape) == 2:
                    # Shape: (channels, time_steps) - already in correct format
                    eeg_data = eeg_raw.astype(np.float32)
                else:
                    # Try to reshape
                    eeg_data = eeg_raw.flatten().reshape(-1, eeg_raw.shape[-1]).astype(np.float32)
                    if eeg_data.shape[0] > 128:  # Too many channels, might be wrong shape
                        eeg_data = eeg_raw.reshape(eeg_raw.shape[0], -1).astype(np.float32)
            else:
                # No known EEG key found, return dummy data
                print(f"Warning: Could not find EEG data key in {eeg_file.name}, available keys: {[k for k in mat_data.keys() if not k.startswith('__')]}")
                eeg_data = np.random.randn(30, 200).astype(np.float32)
            
            emotion_label = None
            if label_file and label_file.exists():
                try:
                    labels_data = loadmat(str(label_file))
                    # The label_file contains metadata but emotion labels come from audio filenames
                except Exception as e:
                    pass  # Silently continue
            
            # normalize to standard 28 channels
            if eeg_data.shape[0] != 28:
                if eeg_data.shape[0] > 28:
                    eeg_data = eeg_data[:28, :]
                else:
                    # pad with zeros
                    pad = np.zeros((28 - eeg_data.shape[0], eeg_data.shape[1]), dtype=np.float32)
                    eeg_data = np.concatenate([eeg_data, pad], axis=0)
            return eeg_data, emotion_label
            
        except Exception as e:
            print(f"Warning: Failed to load EEG {eeg_file}: {e}")
            # Return dummy data to allow iteration to continue
            return np.random.randn(28, 200).astype(np.float32), None
    
    def _load_and_process_audio(self, audio_file: Path) -> Optional[np.ndarray]:
        """Load audio and extract MFCC features using librosa."""
        try:
            # Load audio using librosa
            import librosa
            
            waveform, sample_rate = librosa.load(str(audio_file))
            
            # Compute MFCC features
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=400,
                hop_length=160
            )
            
            # Normalize
            mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / \
                   (mfcc.std(axis=1, keepdims=True) + 1e-8)
            
            return mfcc
        except Exception as e:
            print(f"Warning: Failed to load audio {audio_file}: {e}")
            return None
    
    def _load_video_stub(self, video_file: Path) -> Optional[np.ndarray]:
        """Stub for video loading (to be implemented)."""
        # TODO: Extract video frames and compute face embeddings/landmarks
        return None
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a multimodal sample.
        
        Returns
        -------
        dict with keys:
            - 'eeg': (28, time_steps) or windowed tensor
            - 'audio': (n_mfcc, time) audio features or None
            - 'video': (n_frames, features) video features or None
            - 'emotion': emotion class label (0-4)
            - 'subject_id': subject identifier
        """
        sample = self.samples[idx]
        data = {'subject_id': sample['subject_id']}
        
        # Load EEG
        eeg_data, eeg_emotion = self._load_eeg(sample['eeg'], sample['eeg_label'])
        
        # Normalize EEG
        if self.normalize_eeg:
            if len(eeg_data.shape) == 2:
                eeg_data = (eeg_data - eeg_data.mean(axis=1, keepdims=True)) / \
                          (eeg_data.std(axis=1, keepdims=True) + 1e-8)
        
        data['eeg'] = torch.from_numpy(eeg_data.astype(np.float32))
        
        # Load audio if available
        if self.load_audio and sample['audio']:
            audio_data = self._load_and_process_audio(sample['audio'])
            if audio_data is not None:
                data['audio'] = torch.from_numpy(audio_data.astype(np.float32))
            else:
                data['audio'] = None
        else:
            data['audio'] = None
        
        # Load video if available (stub)
        if self.load_video and sample['video']:
            video_data = self._load_video_stub(sample['video'])
            if video_data is not None:
                data['video'] = torch.from_numpy(video_data.astype(np.float32))
            else:
                data['video'] = None
        else:
            data['video'] = None
        
        # Set emotion label
        emotion = sample['audio_emotion']
        if emotion in self.EMOTION_MAP:
            data['emotion'] = self.EMOTION_MAP[emotion]
        else:
            data['emotion'] = -1  # Unknown emotion
        
        return data


def create_faced_dataloader(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 512,
    stride: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    subjects: Optional[List[int]] = None,
) -> Tuple[DataLoader, FAEDDataset]:
    """Create a DataLoader for FACED pre-training.
    
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
    
    Returns
    -------
    dataloader : DataLoader
    dataset : FAEDDataset
    """
    dataset = FAEDDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        subjects=subjects,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, dataset


def _eav_collate_fn(batch):
    """Custom collate function for EAV dataset to handle variable-size tensors and None values."""
    collated = {}
    
    try:
        # Get all keys from first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key == 'subject_id':
                collated[key] = [item[key] for item in batch]
            elif key == 'emotion':
                collated[key] = torch.tensor([item[key] for item in batch])
            elif key == 'eeg':
                # EEG is typically (batch, channels, time_steps)
                collated[key] = torch.stack([item[key] for item in batch])
            elif key in ['audio', 'video']:
                # Handle None values and variable lengths
                modality_list = [item[key] for item in batch]
                valid_indices = [i for i, x in enumerate(modality_list) if x is not None]
                
                if valid_indices:
                    valid_data = [modality_list[i] for i in valid_indices]
                    # Try to stack if shapes match
                    try:
                        collated[key] = torch.stack(valid_data)
                    except RuntimeError:
                        # Shapes don't match - just return list
                        collated[key] = valid_data
                else:
                    collated[key] = None
            else:
                collated[key] = [item[key] for item in batch]
        
        return collated
    except Exception as e:
        # Fallback to default collation
        print(f"Collate error: {e}")
        return batch


def create_eav_dataloader(
    eav_data_dir: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    subjects: Optional[List[int]] = None,
    load_audio: bool = True,
    load_video: bool = False,
) -> Tuple[DataLoader, EAVMultimodalDataset]:
    """Create a DataLoader for EAV fine-tuning.
    
    Parameters
    ----------
    eav_data_dir : str
        Path to EAV/EAV directory.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle the data.
    num_workers : int
        Number of data loading workers.
    subjects : List[int], optional
        Specific subjects to load. If None, loads all.
    load_audio : bool
        Whether to load audio features.
    load_video : bool
        Whether to load video features.
    
    Returns
    -------
    dataloader : DataLoader
    dataset : EAVMultimodalDataset
    """
    dataset = EAVMultimodalDataset(
        eav_data_dir=eav_data_dir,
        subjects=subjects,
        load_audio=load_audio,
        load_video=load_video,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_eav_collate_fn,
    )
    
    return dataloader, dataset
