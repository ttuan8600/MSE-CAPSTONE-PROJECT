"""Quick test to verify models work."""

import sys
sys.path.insert(0, r"c:\Users\ttuan8600\Documents\My Projects\MSE-CAPSTONE-PROJECT")

import torch
from src.models import EEGEncoder, EEGEncoderLSTM, EmotionClassifier

# Test EEG Encoder
encoder = EEGEncoder(in_channels=28, latent_dim=128)
x = torch.randn(4, 28, 512)  # batch of 4, 28 channels, 512 time steps
output = encoder(x)

print("EEGEncoder Test:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == torch.Size([4, 128]), f"Expected [4, 128], got {output.shape}"
print("  ✓ Pass")

# Test EEG Encoder LSTM
encoder_lstm = EEGEncoderLSTM(in_channels=28, hidden_dim=64, latent_dim=128)
output_lstm = encoder_lstm(x)

print("\nEEGEncoderLSTM Test:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output_lstm.shape}")
assert output_lstm.shape == torch.Size([4, 128]), f"Expected [4, 128], got {output_lstm.shape}"
print("  ✓ Pass")

# Test Emotion Classifier
classifier = EmotionClassifier(latent_dim=128, num_emotions=5)
logits = classifier(output)

print("\nEmotionClassifier Test:")
print(f"  Input shape: {output.shape}")
print(f"  Output shape: {logits.shape}")
assert logits.shape == torch.Size([4, 5]), f"Expected [4, 5], got {logits.shape}"
print("  ✓ Pass")

print("\n✅ All models instantiate and forward pass works!")
