"""Pre-training and fine-tuning pipeline for EEG-based emotion recognition.

This script implements the two-stage training approach:
1. Pre-train EEG encoder on massive FACED dataset (123 subjects)
2. Fine-tune on EAV multimodal dataset with synchronized audio/EEG/video
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models.eeg_encoder import EEGEncoder, EEGEncoderLSTM, EmotionClassifier
from src.preprocessing.data_loader import (
    create_faced_dataloader,
    create_eav_dataloader,
)


class PretrainingTrainer:
    """Trainer for pre-training EEG encoder on FACED dataset."""
    
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        output_dir: str = "outputs/pretraining",
    ):
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.CrossEntropyLoss()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.output_dir)
        self.global_step = 0
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.encoder.train()
        self.classifier.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (eeg, labels) in enumerate(dataloader):
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            latent = self.encoder(eeg)
            logits = self.classifier(latent)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.classifier.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * eeg.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += eeg.size(0)
            
            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")
            
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}\n")
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        self.writer.add_scalar("train/epoch_accuracy", avg_acc, epoch)
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """Validate on a dataset."""
        self.encoder.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for eeg, labels in dataloader:
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)
            
            latent = self.encoder(eeg)
            logits = self.classifier(latent)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * eeg.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += eeg.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        print(f"Validation | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")
        self.writer.add_scalar("val/epoch_loss", avg_loss, epoch)
        self.writer.add_scalar("val/epoch_accuracy", avg_acc, epoch)
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, best=False):
        """Save model checkpoint."""
        ckpt_name = "best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        path = self.output_dir / ckpt_name
        
        torch.save({
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        
        print(f"Saved checkpoint: {path}")


class FineTuningTrainer:
    """Trainer for fine-tuning on EAV multimodal data."""
    
    def __init__(
        self,
        encoder: nn.Module,
        pretrained_path: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        output_dir: str = "outputs/finetuning",
    ):
        self.encoder = encoder.to(device)
        self.device = device
        
        # Load pre-trained weights
        if os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location=device)
            self.encoder.load_state_dict(ckpt['encoder'])
            print(f"Loaded pre-trained encoder from {pretrained_path}")
        
        # Emotion classifier for fine-tuning
        self.classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
        
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.output_dir)
        self.global_step = 0
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch on EAV data."""
        self.encoder.train()
        self.classifier.train()
        
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            eeg = batch['eeg'].to(self.device)
            
            # Dummy labels (replace with actual emotion labels from EAV metadata)
            labels = torch.randint(0, 5, (eeg.size(0),)).to(self.device)
            
            # Forward pass
            latent = self.encoder(eeg)
            logits = self.classifier(latent)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.classifier.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item() * eeg.size(0)
            total_samples += eeg.size(0)
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
            
            self.writer.add_scalar("finetune/loss", loss.item(), self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}\n")
        self.writer.add_scalar("finetune/epoch_loss", avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch):
        """Save fine-tuned model."""
        path = self.output_dir / f"finetuned_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
        }, path)
        
        print(f"Saved fine-tuned model: {path}")


def pretrain(args):
    """Run pre-training on FACED."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader
    print("Loading FACED dataset...")
    dataloader, dataset = create_faced_dataloader(
        data_dir=args.faced_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
    )
    print(f"Loaded {len(dataset)} windows from FACED")
    
    # Create model
    encoder = EEGEncoder(in_channels=28, latent_dim=128)
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5)
    
    # Create trainer
    trainer = PretrainingTrainer(
        encoder=encoder,
        classifier=classifier,
        device=device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss, train_acc = trainer.train_epoch(dataloader, epoch)
        
        if train_loss < best_loss:
            best_loss = train_loss
            trainer.save_checkpoint(epoch, best=True)
        
        trainer.scheduler.step()


def finetune(args):
    """Run fine-tuning on EAV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader
    print("Loading EAV dataset...")
    dataloader, dataset = create_eav_dataloader(
        eav_data_dir=args.eav_dir,
        batch_size=args.batch_size,
    )
    print(f"Loaded {len(dataset)} samples from EAV")
    
    # Create model and load pre-trained encoder
    encoder = EEGEncoder(in_channels=28, latent_dim=128)
    
    # Create trainer
    trainer = FineTuningTrainer(
        encoder=encoder,
        pretrained_path=args.pretrained_path,
        device=device,
        learning_rate=args.finetune_lr,
        output_dir=args.output_dir,
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        trainer.train_epoch(dataloader, epoch)
        trainer.scheduler.step()
        
        if (epoch + 1) % 2 == 0:
            trainer.save_checkpoint(epoch)


def main():
    parser = argparse.ArgumentParser(description="Pre-train and fine-tune EEG encoder")
    
    # Common arguments
    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True,
                       help="Training mode")
    parser.add_argument("--num-epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for checkpoints")
    
    # Pre-training arguments
    parser.add_argument("--faced-dir", type=str,
                       default="data/raw/Processed_data/Processed_data",
                       help="Path to FACED dataset")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate for pre-training")
    parser.add_argument("--window-size", type=int, default=512,
                       help="EEG window size for pre-training")
    
    # Fine-tuning arguments
    parser.add_argument("--eav-dir", type=str,
                       default="data/raw/EAV/EAV",
                       help="Path to EAV dataset")
    parser.add_argument("--pretrained-path", type=str,
                       default="outputs/pretraining/best_model.pt",
                       help="Path to pre-trained encoder")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                       help="Learning rate for fine-tuning")
    
    args = parser.parse_args()
    
    # Add timestamp to output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}/{args.mode}_{timestamp}"
    
    if args.mode == "pretrain":
        pretrain(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
