"""Baseline experiment comparing EEG-only vs EEG+audio fine-tuning.

This standalone script fabricates small random datasets so you can quickly
verify that the fusion module is wired correctly and see whether adding the
audio branch provides any benefit (or at least doesn't crash).

Usage:
    python scripts/baseline_experiment.py [--use-audio]

By default it runs 3 epochson 1k synthetic samples, reports training loss/
accuracy and prints final classifier logits statistics.

No real EAV data required.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, MultimodalFusion, EmotionClassifier


class SyntheticEegAudioDataset(Dataset):
    """Generates random EEG (and optionally audio) samples with random labels."""

    def __init__(self, size: int = 1000, eeg_dim=(28, 512), audio_dim=(13, 500),
                 use_audio: bool = False, num_classes: int = 5):
        super().__init__()
        self.size = size
        self.eeg_dim = eeg_dim
        self.audio_dim = audio_dim
        self.use_audio = use_audio
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        eeg = torch.randn(self.eeg_dim, dtype=torch.float32)
        sample = {"eeg": eeg}
        if self.use_audio:
            audio = torch.randn(self.audio_dim, dtype=torch.float32)
            sample["audio"] = audio
        label = torch.randint(0, self.num_classes, (1,))
        sample["label"] = label
        return sample


def collate_fn(batch):
    """Simple stack collate for synthetic data."""
    eeg = torch.stack([b["eeg"] for b in batch])
    out = {"eeg": eeg}
    if "audio" in batch[0]:
        audio = torch.stack([b["audio"] for b in batch])
        out["audio"] = audio
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out, labels


def run_experiment(use_audio: bool = False, fusion_mode: str = "concat"):
    """Run a single experiment (EEG-only or EEG+audio).

    Parameters
    ----------
    use_audio : bool
        Whether to include audio modality.
    fusion_mode : str
        Fusion strategy ("concat", "cross_attention", "gated").
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Running experiment: use_audio={use_audio}, fusion_mode={fusion_mode}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create tiny dataset
    dataset = SyntheticEegAudioDataset(size=512, use_audio=use_audio)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Initialize models
    encoder = EEGEncoder(in_channels=28, latent_dim=128).to(device)
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
    
    params = [*encoder.parameters(), *classifier.parameters()]
    
    audio_encoder = None
    fusion = None
    
    if use_audio:
        audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128).to(device)
        fusion = MultimodalFusion(latent_dim=128, mode=fusion_mode).to(device)
        params.extend(audio_encoder.parameters())
        params.extend(fusion.parameters())

    optimizer = optim.Adam(params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch, labels in dataloader:
            eeg = batch["eeg"].to(device)
            labels = labels.to(device)

            # Forward pass
            eeg_latent = encoder(eeg)

            if use_audio and audio_encoder is not None and fusion is not None:
                audio = batch["audio"].to(device)
                audio_latent = audio_encoder(audio)
                fused = fusion(eeg_latent, audio_latent)
            else:
                fused = eeg_latent

            logits = classifier(fused)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item() * eeg.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += eeg.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    # Get final logits from last batch for statistics
    logits = None
    with torch.no_grad():
        for batch, labels in dataloader:
            eeg = batch["eeg"].to(device)
            eeg_latent = encoder(eeg)
            if use_audio and audio_encoder is not None and fusion is not None:
                audio = batch["audio"].to(device)
                audio_latent = audio_encoder(audio)
                fused = fusion(eeg_latent, audio_latent)
            else:
                fused = eeg_latent
            logits = classifier(fused)
            break

    if logits is not None:
        print(f"\nFinal classifier logits shape: {logits.shape}")
        print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline EEG ± audio experiment")
    parser.add_argument("--use-audio", action="store_true", help="Include audio modality")
    parser.add_argument("--fusion-mode", type=str, default="concat",
                        choices=["concat","cross_attention","gated"],
                        help="Fusion strategy when audio is used")
    args = parser.parse_args()
    
    # Run EEG-only baseline
    run_experiment(use_audio=False, fusion_mode=args.fusion_mode)

    # Run EEG+audio experiment
    run_experiment(use_audio=args.use_audio, fusion_mode=args.fusion_mode)
    print("  • EEG-only: baseline single-modality branch")
    print("  • EEG+audio: fusion concatenates both latents")


if __name__ == "__main__":
    main()
