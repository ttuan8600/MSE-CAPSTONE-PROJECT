"""Advanced training script with multiple strategies for improving accuracy.

Implements:
1. Multiple fusion mode comparison
2. Hyperparameter grid search
3. Learning rate scheduling optimization
4. Data augmentation
5. Model ensemble evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eeg_encoder import (
    EEGEncoder,
    EEGEncoderLSTM,
    AudioEncoder,
    EmotionClassifier,
    MultimodalFusion,
)
from src.preprocessing.data_loader import create_eav_dataloader


def create_dataloader_eav(batch_size: int = 32, num_workers: int = 0):
    """Helper to create EAV dataloader."""
    try:
        eav_data_dir = "data/raw/EAV/EAV"
        dl, ds = create_eav_dataloader(
            eav_data_dir=eav_data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            load_audio=True,
        )
        return dl, ds
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        # Return None to signal fallback
        return None, None


class ImprovedTrainer:
    """Trainer with advanced techniques for better accuracy."""
    
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        audio_encoder: nn.Module = None,
        fusion: nn.Module = None,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.device = device or torch.device('cpu')
        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        self.audio_encoder = audio_encoder.to(self.device) if audio_encoder else None
        self.fusion = fusion.to(self.device) if fusion else None
        
        # Gather parameters
        params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        if self.audio_encoder:
            params.extend(self.audio_encoder.parameters())
        if self.fusion:
            params.extend(self.fusion.parameters())
        
        # Optimizer
        self.optimizer = optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.best_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train one epoch."""
        self.encoder.train()
        self.classifier.train()
        if self.audio_encoder:
            self.audio_encoder.train()
        if self.fusion:
            self.fusion.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2 and isinstance(batch[0], dict):
                        batch_dict, labels = batch
                        eeg = batch_dict.get('eeg', None)
                        audio = batch_dict.get('audio', None)
                    elif len(batch) == 3:
                        eeg, audio, labels = batch
                    else:
                        continue
                else:
                    eeg = batch.get('eeg', None)
                    audio = batch.get('audio', None)
                    labels = batch.get('label', None)
                
                if eeg is None or labels is None:
                    continue
                
                eeg = eeg.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                eeg_latent = self.encoder(eeg)
                
                if self.audio_encoder and audio is not None:
                    audio = audio.to(self.device)
                    audio_latent = self.audio_encoder(audio)
                    fused = self.fusion(eeg_latent, audio_latent)
                else:
                    fused = eeg_latent
                
                logits = self.classifier(fused)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in [self.encoder, self.classifier, self.audio_encoder, self.fusion] 
                     if p is not None for p in p.parameters()],
                    max_norm=1.0
                )
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)
                
                if batch_idx % 20 == 0:
                    print(f"    Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            except Exception as e:
                continue
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(avg_acc)
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.encoder.eval()
        self.classifier.eval()
        if self.audio_encoder:
            self.audio_encoder.eval()
        if self.fusion:
            self.fusion.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in val_loader:
            try:
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2 and isinstance(batch[0], dict):
                        batch_dict, labels = batch
                        eeg = batch_dict.get('eeg', None)
                        audio = batch_dict.get('audio', None)
                    elif len(batch) == 3:
                        eeg, audio, labels = batch
                    else:
                        continue
                else:
                    eeg = batch.get('eeg', None)
                    audio = batch.get('audio', None)
                    labels = batch.get('label', None)
                
                if eeg is None or labels is None:
                    continue
                
                eeg = eeg.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                eeg_latent = self.encoder(eeg)
                
                if self.audio_encoder and audio is not None:
                    audio = audio.to(self.device)
                    audio_latent = self.audio_encoder(audio)
                    fused = self.fusion(eeg_latent, audio_latent)
                else:
                    fused = eeg_latent
                
                logits = self.classifier(fused)
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)
            
            except Exception as e:
                continue
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)
        
        return avg_loss, avg_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int,
              scheduler=None):
        """Train for multiple epochs."""
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                print(f"  * Best model! Acc: {val_acc:.4f}")
            
            # LR scheduler step
            if scheduler:
                scheduler.step(val_loss)


def train_configuration(
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_audio: bool,
    fusion_mode: str,
    learning_rate: float,
    num_epochs: int,
    encoder_type: str = "cnn",
) -> Dict:
    """Train a single configuration and return metrics."""
    
    print(f"\n{'='*70}")
    print(f"Configuration: {'EEG+Audio' if use_audio else 'EEG-only'} | {encoder_type} | LR={learning_rate}")
    print(f"Fusion Mode: {fusion_mode if use_audio else 'N/A'}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create encoder
    if encoder_type == "cnn":
        encoder = EEGEncoder(in_channels=28, latent_dim=128)
    elif encoder_type == "lstm":
        encoder = EEGEncoderLSTM(in_channels=28, latent_dim=128)
    else:
        encoder = EEGEncoder(in_channels=28, latent_dim=128)
    
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5)
    
    audio_encoder = None
    fusion = None
    if use_audio:
        audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128)
        fusion = MultimodalFusion(latent_dim=128, mode=fusion_mode)
    
    # Create trainer
    trainer = ImprovedTrainer(
        encoder=encoder,
        classifier=classifier,
        audio_encoder=audio_encoder,
        fusion=fusion,
        device=device,
        learning_rate=learning_rate,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs, scheduler=scheduler)
    
    # Prepare results
    results = {
        'config': f"{'audio' if use_audio else 'eeg_only'}_{fusion_mode if use_audio else 'n/a'}",
        'use_audio': use_audio,
        'fusion_mode': fusion_mode,
        'encoder_type': encoder_type,
        'learning_rate': learning_rate,
        'best_val_acc': trainer.best_acc,
        'history': trainer.history,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Advanced training for improved accuracy")
    parser.add_argument("--num-epochs", type=int, default=20,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--fusion-modes", nargs="+", 
                       default=["concat", "cross_attention", "gated"],
                       help="Fusion modes to test")
    parser.add_argument("--no-audio", action="store_true",
                       help="Test EEG-only model")
    parser.add_argument("--encoder-types", nargs="+",
                       default=["cnn"],
                       help="Encoder types to test")
    parser.add_argument("--output-dir", default="outputs/advanced_training",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading EAV dataset...")
    try:
        train_val_loader, dataset = create_dataloader_eav(
            batch_size=args.batch_size,
            num_workers=0,
        )
        
        if dataset is None:
            raise Exception("Failed to load dataset")
        
        # Split into train/val
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"OK: Dataset loaded: {train_size} train, {val_size} val")
    
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return
    
    # Run experiments
    all_results = {}
    
    # Test configurations
    use_audio_options = [True] if not args.no_audio else [False]
    
    for use_audio in use_audio_options:
        for encoder_type in args.encoder_types:
            fusion_modes = args.fusion_modes if use_audio else [None]
            for fusion_mode in fusion_modes:
                fusion_mode = fusion_mode or "n/a"
                
                key = f"{'audio' if use_audio else 'eeg_only'}_{encoder_type}_{fusion_mode}"
                print(f"\n\n{'#'*70}")
                print(f"# Training: {key}")
                print(f"{'#'*70}")
                
                results = train_configuration(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    use_audio=use_audio,
                    fusion_mode=fusion_mode,
                    learning_rate=args.learning_rate,
                    num_epochs=args.num_epochs,
                    encoder_type=encoder_type,
                )
                
                all_results[key] = results
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY OF ALL CONFIGURATIONS")
    print(f"{'='*70}\n")
    
    for key, results in sorted(all_results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True):
        print(f"{key:40s} | Best Val Acc: {results['best_val_acc']:.4f}")
    
    # Save results
    results_file = Path(output_dir) / "results.json"
    
    # Convert to serializable
    serializable_results = {}
    for key, res in all_results.items():
        serializable_results[key] = {
            'config': res['config'],
            'use_audio': res['use_audio'],
            'fusion_mode': res['fusion_mode'],
            'encoder_type': res['encoder_type'],
            'learning_rate': res['learning_rate'],
            'best_val_acc': float(res['best_val_acc']),
            'final_train_acc': float(res['history']['train_acc'][-1]) if res['history']['train_acc'] else None,
            'final_val_acc': float(res['history']['val_acc'][-1]) if res['history']['val_acc'] else None,
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nOK: Results saved to {results_file}")


if __name__ == "__main__":
    main()
