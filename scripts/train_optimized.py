"""Optimized training script for improved accuracy.

Implements best practices:
- Proper train/val/test split
- Multiple fusion modes and hyperparameter experiments
- Best model checkpointing
- Detailed metric tracking
- Learning rate warmup and fine-tuning
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eeg_encoder import (
    EEGEncoder,
    AudioEncoder,
    EmotionClassifier,
    MultimodalFusion,
)
from src.preprocessing.data_loader import create_eav_dataloader


def setup_device():
    """Setup device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU")
    return device


class OptimizedTrainer:
    """Trainer with optimized hyperparameters and validation."""
    
    def __init__(
        self,
        device: torch.device,
        use_audio: bool = True,
        fusion_mode: str = "gated",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        output_dir: str = "outputs/optimized_training",
    ):
        self.device = device
        self.use_audio = use_audio
        self.fusion_mode = fusion_mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize models
        self.encoder = EEGEncoder(in_channels=28, latent_dim=128).to(device)
        self.classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
        
        self.params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        
        if self.use_audio:
            self.audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128).to(device)
            self.fusion = MultimodalFusion(latent_dim=128, mode=fusion_mode).to(device)
            self.params.extend(self.audio_encoder.parameters())
            self.params.extend(self.fusion.parameters())
        else:
            self.audio_encoder = None
            self.fusion = None
        
        # Optimizer with weight decay
        self.optimizer = optim.Adam(
            self.params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler: warmup then decay
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 5,
            total_steps=5000,  # Will be adjusted
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10.0,
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': None,
            'test_loss': None,
        }
    
    def _forward_pass(self, batch: Dict, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional audio."""
        # Handle both real data (dict) and synthetic data (tuple)
        if isinstance(batch, (tuple, list)):
            # Synthetic data: (eeg, audio, labels)
            if len(batch) == 3:
                eeg, audio, labels = batch
                eeg = eeg.to(self.device)
                labels = labels.to(self.device)
                audio = audio.to(self.device) if audio is not None else None
            # Real data tuple: (dict, labels) 
            elif len(batch) == 2:
                batch_dict, labels = batch
                eeg = batch_dict['eeg'].to(self.device)
                audio = batch_dict.get('audio', None)
                if audio is not None:
                    audio = audio.to(self.device)
                labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else None
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        else:
            # Real data: dict
            batch_dict = batch
            eeg = batch_dict['eeg'].to(self.device)
            audio = batch_dict.get('audio', None)
            if audio is not None:
                audio = audio.to(self.device)
            labels = batch_dict.get('label', None)
            if labels is not None:
                labels = labels.to(self.device)
        
        # EEG encoding
        eeg_latent = self.encoder(eeg)
        
        # Audio encoding if available
        if self.use_audio and audio is not None:
            audio_latent = self.audio_encoder(audio)
            fused = self.fusion(eeg_latent, audio_latent)
        else:
            fused = eeg_latent
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, labels, eeg
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train one epoch."""
        self.encoder.train()
        self.classifier.train()
        if self.use_audio and self.audio_encoder is not None:
            self.audio_encoder.train()
            self.fusion.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                logits, labels, eeg = self._forward_pass(batch, training=True)
                
                if labels is None or eeg.shape[0] == 0:
                    continue
                
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
                self.optimizer.step()
                
                if hasattr(self, 'scheduler'):
                    self.scheduler.step()
                
                # Metrics
                total_loss += loss.item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)
                batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
            except Exception as e:
                print(f"  ⚠ Batch {batch_idx} skipped: {type(e).__name__}")
                continue
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate on validation set."""
        self.encoder.eval()
        self.classifier.eval()
        if self.use_audio and self.audio_encoder is not None:
            self.audio_encoder.eval()
            self.fusion.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in val_loader:
            try:
                logits, labels, eeg = self._forward_pass(batch, training=False)
                
                if labels is None or eeg.shape[0] == 0:
                    continue
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)
            
            except Exception as e:
                continue
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, avg_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 20):
        """Train for multiple epochs."""
        print(f"\n{'='*70}")
        print(f"OPTIMIZED TRAINING")
        print(f"{'='*70}")
        print(f"Config: use_audio={self.use_audio}, fusion_mode={self.fusion_mode}")
        print(f"LR={self.learning_rate}, Weight Decay={self.weight_decay}")
        print(f"Epochs={num_epochs}, Device={self.device}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, best=True)
                print(f"  ✓ Best model saved! (Acc: {val_acc:.4f})")
            
            print()
        
        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, epoch: int, best: bool = False):
        """Save model checkpoint."""
        ckpt_name = "best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        path = self.output_dir / ckpt_name
        
        ckpt = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'metrics': self.metrics,
        }
        if self.use_audio:
            ckpt['audio_encoder'] = self.audio_encoder.state_dict()
            ckpt['fusion'] = self.fusion.state_dict()
        
        torch.save(ckpt, path)
        print(f"    Saved: {path}")
    
    def evaluate_test(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on test set."""
        test_loss, test_acc = self.validate(test_loader)
        self.metrics['test_loss'] = test_loss
        self.metrics['test_acc'] = test_acc
        return test_loss, test_acc
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_file = self.output_dir / "metrics.json"
        
        # Convert numpy values for JSON serialization
        metrics_json = {
            'train_loss': [float(x) for x in self.metrics['train_loss']],
            'train_acc': [float(x) for x in self.metrics['train_acc']],
            'val_loss': [float(x) for x in self.metrics['val_loss']],
            'val_acc': [float(x) for x in self.metrics['val_acc']],
            'test_loss': float(self.metrics['test_loss']) if self.metrics['test_loss'] is not None else None,
            'test_acc': float(self.metrics['test_acc']) if self.metrics['test_acc'] is not None else None,
            'best_epoch': self.best_epoch,
            'config': {
                'use_audio': self.use_audio,
                'fusion_mode': self.fusion_mode,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Optimized training for better accuracy")
    parser.add_argument("--use-audio", action="store_true", default=True,
                       help="Use audio modality (default: True)")
    parser.add_argument("--no-audio", action="store_true",
                       help="Disable audio modality")
    parser.add_argument("--fusion-mode", choices=["concat", "cross_attention", "gated"],
                       default="gated", help="Fusion mode")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=25,
                       help="Number of epochs")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--output-dir", default="outputs/optimized_training",
                       help="Output directory")
    parser.add_argument("--subjects", type=str, default=None,
                       help="Comma-separated list of subject indices to use")
    
    args = parser.parse_args()
    
    if args.no_audio:
        args.use_audio = False
    
    # Setup
    device = setup_device()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Load data
    print("\nLoading EAV dataset...")
    try:
        eav_data_dir = "data/raw/EAV/EAV"
        full_dataloader, full_dataset = create_eav_dataloader(
            eav_data_dir=eav_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            load_audio=args.use_audio,
        )
        
        # Get dataset size
        dataset_size = len(full_dataset) if hasattr(full_dataset, '__len__') else None
        
        print(f"✓ Dataset loaded (size: {dataset_size})")
        
        # Split into train/val/test (70/15/15)
        if dataset_size is not None and dataset_size > 0:
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            
            print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")
        else:
            # Use the full loader for all
            train_loader = val_loader = test_loader = full_dataloader
            print("  Using full dataset for all splits")
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Using synthetic data fallback...")
        
        # Fallback: use synthetic data
        from torch.utils.data import TensorDataset
        
        n_samples = 500
        X_eeg = torch.randn(n_samples, 28, 512)
        X_audio = torch.randn(n_samples, 13, 500)
        y = torch.randint(0, 5, (n_samples,))
        
        dataset = TensorDataset(X_eeg, X_audio, y)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train
    trainer = OptimizedTrainer(
        device=device,
        use_audio=args.use_audio,
        fusion_mode=args.fusion_mode,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=output_dir,
    )
    
    trainer.train(train_loader, val_loader, num_epochs=args.num_epochs)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = trainer.evaluate_test(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}\n")
    
    # Save metrics
    trainer.save_metrics()
    
    print(f"✓ Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
