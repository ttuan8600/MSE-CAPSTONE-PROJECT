import sys
from pathlib import Path

# ensure the root of the project is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.eeg_encoder import (
    EEGEncoder,
    AudioEncoder,
    MultimodalFusion,
    EmotionClassifier,
)


def test_audio_encoder_output_shape():
    model = AudioEncoder(n_mfcc=13, latent_dim=128)
    dummy = torch.randn(4, 13, 500)  # batch of 4, 13 coeffs, 500 timesteps
    out = model(dummy)
    assert out.shape == (4, 128)


def test_fusion_combines_features():
    eeg_feat = torch.randn(3, 128)
    audio_feat = torch.randn(3, 128)
    fusion = MultimodalFusion(latent_dim=128)
    fused = fusion(eeg_feat, audio_feat)
    assert fused.shape == (3, 128)


def test_classifier_accepts_fused():
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5)
    inp = torch.randn(2, 128)
    logits = classifier(inp)
    assert logits.shape == (2, 5)
