"""Model inference utility for emotion recognition.

Handles model loading, preprocessing, and prediction for EEG+Audio data.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class EmotionPredictor:
    """High-level interface for emotion prediction."""
    
    EMOTION_LABELS = ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize emotion predictor.
        
        Parameters
        ----------
        model_path : str
            Path to trained model checkpoint (.pt file)
        device : str
            Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Import model components
        from src.models.eeg_encoder import EEGEncoder, AudioEncoder
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Recreate model (adjust architecture if needed)
        self.model = EmotionRecognitionModel()
        
        # Load weights
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {self.model_path}")
    
    def predict(self, eeg_data: np.ndarray, audio_data: Optional[np.ndarray] = None) -> Dict:
        """
        Predict emotion from EEG and optional audio data.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG signal of shape (28, time_steps)
        audio_data : np.ndarray, optional
            Audio MFCC features of shape (13, time_steps)
        
        Returns
        -------
        dict
            {
                'emotion': str,           # Predicted emotion label
                'emotion_id': int,        # Emotion ID (0-4)
                'confidence': float,      # Confidence score (0-1)
                'probabilities': dict,    # Per-class probabilities
                'input_shapes': dict      # Input data shapes for verification
            }
        """
        # Validate and prepare inputs
        eeg_tensor, audio_tensor = self._prepare_inputs(eeg_data, audio_data)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(eeg_tensor, audio_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Extract prediction
        emotion_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        emotion_label = self.EMOTION_LABELS[emotion_id]
        
        # Create output
        probabilities = {
            self.EMOTION_LABELS[i]: float(probs[i])
            for i in range(len(self.EMOTION_LABELS))
        }
        
        return {
            'emotion': emotion_label,
            'emotion_id': emotion_id,
            'confidence': confidence,
            'probabilities': probabilities,
            'input_shapes': {
                'eeg': tuple(eeg_data.shape),
                'audio': tuple(audio_data.shape) if audio_data is not None else None
            }
        }
    
    def _prepare_inputs(self, eeg_data: np.ndarray, audio_data: Optional[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare and validate input tensors."""
        # Validate EEG
        if not isinstance(eeg_data, np.ndarray):
            eeg_data = np.array(eeg_data)
        
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        if eeg_data.shape[0] != 28:
            raise ValueError(f"EEG must have 28 channels, got {eeg_data.shape[0]}")
        
        # Normalize EEG
        eeg_data = (eeg_data - eeg_data.mean(axis=1, keepdims=True)) / \
                   (eeg_data.std(axis=1, keepdims=True) + 1e-8)
        
        eeg_tensor = torch.from_numpy(eeg_data.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # Handle audio
        if audio_data is None:
            # Create dummy audio tensor
            audio_tensor = torch.zeros(1, 13, 128, dtype=torch.float32).to(self.device)
        else:
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            if audio_data.shape[0] != 13:
                raise ValueError(f"Audio must have 13 MFCC channels, got {audio_data.shape[0]}")
            
            # Normalize audio
            audio_data = (audio_data - audio_data.mean(axis=1, keepdims=True)) / \
                        (audio_data.std(axis=1, keepdims=True) + 1e-8)
            
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0).to(self.device)
        
        return eeg_tensor, audio_tensor
    
    def batch_predict(self, eeg_list: list, audio_list: Optional[list] = None) -> list:
        """
        Predict emotions for multiple samples.
        
        Parameters
        ----------
        eeg_list : list
            List of EEG arrays
        audio_list : list, optional
            List of audio arrays (if None, uses dummy audio)
        
        Returns
        -------
        list
            List of prediction dictionaries
        """
        results = []
        for i, eeg in enumerate(eeg_list):
            audio = audio_list[i] if audio_list is not None else None
            result = self.predict(eeg, audio)
            results.append(result)
        
        return results


class EmotionRecognitionModel(torch.nn.Module):
    """Placeholder model for loading attention fusion architecture."""
    
    def __init__(self):
        super().__init__()
        from src.models.eeg_encoder import EEGEncoder, AudioEncoder
        
        self.eeg_encoder = EEGEncoder(in_channels=28, latent_dim=128)
        self.audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128)
        
        # Simple fusion (can be replaced with attention fusion)
        self.fusion_fc = torch.nn.Linear(256, 128)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 5)
        )
    
    def forward(self, eeg, audio):
        """Forward pass."""
        eeg_feat = self.eeg_encoder(eeg)
        audio_feat = self.audio_encoder(audio)
        
        fused = torch.cat([eeg_feat, audio_feat], dim=1)
        fused = torch.relu(self.fusion_fc(fused))
        
        logits = self.classifier(fused)
        return logits
