"""Pre-training and fine-tuning pipeline for EEG-based emotion recognition.

This script implements the two-stage training approach:
1. Pre-train EEG encoder on massive FACED dataset (123 subjects)
2. Fine-tune on EAV multimodal dataset with synchronized audio/EEG/video
"""

import os
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None
    print("Warning: TensorBoard not installed. Install with: pip install tensorboard")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eeg_encoder import (
    EEGEncoder,
    EEGEncoderLSTM,
    EmotionClassifier,
    AudioEncoder,
    MultimodalFusion,
)
from src.preprocessing.data_loader import (
    create_faced_dataloader,
    create_eav_dataloader,
)


class PretrainingTrainer:
    """Trainer for pre-training EEG encoder on FACED dataset."""

    def __init__(
        self,
        encoder: nn.Module,
        classifier: Optional[nn.Module],
        device: torch.device,
        learning_rate: float = 1e-3,
        output_dir: str = "outputs/pretraining",
        task: str = "subject_classification",
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        temperature: float = 0.5,
        projection_dim: int = 128,
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.task = task

        # Create projection / classification heads depending on task
        if task == "contrastive":
            # Simple projection head for contrastive learning
            self.projection = nn.Sequential(
                nn.Linear(encoder.latent_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            ).to(device)
            self.criterion = self._nt_xent_loss
        else:
            # Subject classification
            assert classifier is not None, "Classifier must be provided for subject classification task."
            self.classifier = classifier.to(device)
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        params = list(self.encoder.parameters())
        if task == "contrastive":
            params += list(self.projection.parameters())
        else:
            params += list(self.classifier.parameters())

        self.optimizer = optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        if scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        elif scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            self.scheduler = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.output_dir) if HAS_TENSORBOARD else None
        self.global_step = 0
        self.temperature = temperature

    def _nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

        Implements the formulation from SimCLR:
        - Each instance has two augmented views (z_i, z_j)
        - Positive pair: (i, i + batch_size)
        - Negative pairs: all other combinations
        """
        # z_i / z_j shape: (batch, dim)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        batch_size = z_i.size(0)
        representations = torch.cat([z_i, z_j], dim=0)  # (2*B, dim)

        # Similarity matrix (2B x 2B)
        similarity_matrix = torch.matmul(representations, representations.T)
        similarity_matrix = similarity_matrix / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Targets: for i in [0, B), positive is i+B; for i in [B, 2B), positive is i-B.
        targets = torch.arange(batch_size, device=z_i.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        loss = F.cross_entropy(similarity_matrix, targets)
        return loss
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.encoder.train()
        if self.task == "contrastive":
            self.projection.train()
        else:
            self.classifier.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if self.task == "contrastive":
                x1, x2 = batch
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                z1 = self.encoder(x1)
                z2 = self.encoder(x2)

                p1 = self.projection(z1)
                p2 = self.projection(z2)

                loss = self.criterion(p1, p2)

                # Metrics
                total_loss += loss.item() * x1.size(0)
                total_samples += x1.size(0)

            else:
                eeg, labels = batch
                eeg = eeg.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                latent = self.encoder(eeg)
                logits = self.classifier(latent)
                loss = self.criterion(logits, labels)

                # Metrics
                total_loss += loss.item() * eeg.size(0)
                pred = logits.argmax(dim=1)
                total_correct += (pred == labels).sum().item()
                total_samples += eeg.size(0)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + (list(self.projection.parameters()) if self.task == "contrastive" else list(self.classifier.parameters())),
                max_norm=1.0
            )
            self.optimizer.step()

            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

            if self.writer:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.global_step += 1

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples if self.task != "contrastive" else None

        if self.task == "contrastive":
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}\n")
        else:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}\n")

        if self.writer:
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            if avg_acc is not None:
                self.writer.add_scalar("train/epoch_accuracy", avg_acc, epoch)

        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """Validate on a dataset."""
        if self.task == "contrastive":
            # Contrastive pre-training typically does not use a standard validation set.
            return None, None

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
        if self.writer:
            self.writer.add_scalar("val/epoch_loss", avg_loss, epoch)
            self.writer.add_scalar("val/epoch_accuracy", avg_acc, epoch)

        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, best=False):
        """Save model checkpoint."""
        ckpt_name = "best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        path = self.output_dir / ckpt_name

        ckpt = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.task == "contrastive":
            ckpt['projection'] = self.projection.state_dict()
        else:
            ckpt['classifier'] = self.classifier.state_dict()

        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")


class FineTuningTrainer:
    """Trainer for fine-tuning on EAV data, optionally multimodal (EEG + audio)."""
    
    def __init__(
        self,
        encoder: nn.Module,
        pretrained_path: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        output_dir: str = "outputs/finetuning",
        use_audio: bool = False,
        fusion_mode: str = "concat",
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.use_audio = use_audio
        self.fusion_mode = fusion_mode
        
        # Load pre-trained weights
        if os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location=device)
            self.encoder.load_state_dict(ckpt['encoder'])
            print(f"Loaded pre-trained encoder from {pretrained_path}")
        
        # build audio encoder and fusion if needed
        if self.use_audio:
            self.audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128).to(device)
            self.fusion = MultimodalFusion(latent_dim=128, mode=self.fusion_mode).to(device)
        else:
            self.audio_encoder = None
            self.fusion = None
        
        # Emotion classifier for fine-tuning (takes fused features)
        self.classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
        
        params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        if self.use_audio:
            params += list(self.audio_encoder.parameters()) + list(self.fusion.parameters())
        
        self.optimizer = optim.Adam(
            params,
            lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.output_dir) if HAS_TENSORBOARD else None
        self.global_step = 0
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch on EAV data."""
        self.encoder.train()
        self.classifier.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            eeg = batch['eeg'].to(self.device)

            # Use actual emotion labels when available
            if 'emotion' in batch:
                labels = batch['emotion'].to(self.device)
                # Filter out samples with unknown labels
                mask = labels >= 0
                if mask.sum().item() == 0:
                    continue
                eeg = eeg[mask]
                labels = labels[mask]
            else:
                labels = torch.randint(0, 5, (eeg.size(0),)).to(self.device)

            # Forward pass
            eeg_latent = self.encoder(eeg)
            if self.use_audio and batch.get('audio') is not None:
                audio = batch['audio']
                if isinstance(audio, torch.Tensor):
                    audio = audio[mask]
                elif isinstance(audio, list):
                    audio = [a for a, m in zip(audio, mask.tolist()) if m]

                if audio is not None:
                    audio = audio.to(self.device) if isinstance(audio, torch.Tensor) else None
                audio_latent = self.audio_encoder(audio) if audio is not None else None
                fused = self.fusion(eeg_latent, audio_latent) if audio_latent is not None else eeg_latent
            else:
                fused = eeg_latent

            logits = self.classifier(fused)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.classifier.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
            
            if self.writer:
                self.writer.add_scalar("finetune/loss", loss.item(), self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / total_samples if total_samples else float('nan')
        avg_acc = total_correct / total_samples if total_samples else None
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}\n")
        if self.writer:
            self.writer.add_scalar("finetune/epoch_loss", avg_loss, epoch)
            if avg_acc is not None:
                self.writer.add_scalar("finetune/epoch_accuracy", avg_acc, epoch)
        
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """Validate on EAV data."""
        self.encoder.eval()
        self.classifier.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            eeg = batch['eeg'].to(self.device)
            if 'emotion' in batch:
                labels = batch['emotion'].to(self.device)
                mask = labels >= 0
                if mask.sum().item() == 0:
                    continue
                eeg = eeg[mask]
                labels = labels[mask]
            else:
                labels = torch.randint(0, 5, (eeg.size(0),)).to(self.device)

            eeg_latent = self.encoder(eeg)
            if self.use_audio and batch.get('audio') is not None:
                audio = batch['audio']
                if isinstance(audio, torch.Tensor):
                    audio = audio[mask]
                elif isinstance(audio, list):
                    audio = [a for a, m in zip(audio, mask.tolist()) if m]
                if audio is not None:
                    audio = audio.to(self.device) if isinstance(audio, torch.Tensor) else None
                audio_latent = self.audio_encoder(audio) if audio is not None else None
                fused = self.fusion(eeg_latent, audio_latent) if audio_latent is not None else eeg_latent
            else:
                fused = eeg_latent

            logits = self.classifier(fused)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples if total_samples else float('nan')
        avg_acc = total_correct / total_samples if total_samples else None
        print(f"Validation | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")
        if self.writer:
            self.writer.add_scalar("val/epoch_loss", avg_loss, epoch)
            if avg_acc is not None:
                self.writer.add_scalar("val/epoch_accuracy", avg_acc, epoch)
        return avg_loss, avg_acc
    
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
    
    # Create data loaders
    print("Loading FACED dataset...")
    train_loader, val_loader, dataset = create_faced_dataloader(
        data_dir=args.faced_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        val_split=args.val_split,
        seed=args.seed,
        contrastive=(args.pretrain_task == "contrastive"),
    )
    print(f"Using FACED data directory: {dataset.data_dir}")
    print(f"Loaded {len(dataset)} windows from FACED")

    # Create model
    encoder = EEGEncoder(in_channels=28, latent_dim=128)

    classifier = None
    if args.pretrain_task == "subject_classification":
        num_subjects = len(set(dataset.subject_labels))
        classifier = EmotionClassifier(latent_dim=128, num_emotions=num_subjects)

    # Create trainer
    trainer = PretrainingTrainer(
        encoder=encoder,
        classifier=classifier,
        device=device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        task=args.pretrain_task,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
    )

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)

        val_loss, val_acc = trainer.validate(val_loader, epoch) if val_loader is not None else (None, None)

        # Determine whether to checkpoint
        metric_loss = val_loss if val_loss is not None else train_loss
        if metric_loss is not None and metric_loss < best_loss:
            best_loss = metric_loss
            trainer.save_checkpoint(epoch, best=True)

        if trainer.scheduler is not None:
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
        load_audio=args.use_audio,
        load_video=False,
    )
    print(f"Loaded {len(dataset)} samples from EAV")

    # Optionally split into train/validation
    val_loader = None
    train_loader = dataloader
    if args.val_split and 0.0 < args.val_split < 1.0:
        generator = torch.Generator().manual_seed(args.seed)
        train_size = int(len(dataset) * (1.0 - args.val_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        pin_memory = torch.cuda.is_available()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=dataloader.collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=dataloader.collate_fn,
        )
        print(f"Split EAV into {len(train_dataset)} train and {len(val_dataset)} val samples")

    # Create model and load pre-trained encoder
    encoder = EEGEncoder(in_channels=28, latent_dim=128)
    
    # Create trainer
    trainer = FineTuningTrainer(
        encoder=encoder,
        pretrained_path=args.pretrained_path,
        device=device,
        learning_rate=args.finetune_lr,
        output_dir=args.output_dir,
        use_audio=args.use_audio,
        fusion_mode=args.fusion_mode,
    )
    
    # Resume logic
    start_epoch = 0
    if args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        trainer.encoder.load_state_dict(checkpoint['encoder'])
        trainer.classifier.load_state_dict(checkpoint['classifier'])
        if hasattr(trainer, 'optimizer') and 'optimizer' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed at epoch {start_epoch}")

    # Training loop
    best_loss = float('inf')
    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = (
            trainer.validate(val_loader, epoch) if val_loader is not None else (None, None)
        )

        # Save best model by validation loss if available
        metric_loss = val_loss if val_loss is not None else train_loss
        if metric_loss is not None and metric_loss < best_loss:
            best_loss = metric_loss
            trainer.save_checkpoint(epoch)

        if (epoch + 1) % 2 == 0:
            trainer.save_checkpoint(epoch)

        trainer.scheduler.step()


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
                       default="data/raw/Processed_data",
                       help="Path to FACED dataset (contains subXXX.pkl files)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate for pre-training")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                       help="Weight decay (L2 regularization) for optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "none"],
                       help="Learning rate scheduler type")
    parser.add_argument("--val-split", type=float, default=0.0,
                       help="Fraction of FACED dataset to use for validation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/val split")
    parser.add_argument("--pretrain-task", type=str, default="subject_classification",
                       choices=["subject_classification", "contrastive"],
                       help="Pre-training objective")
    parser.add_argument("--temperature", type=float, default=0.5,
                       help="Temperature for contrastive loss")
    parser.add_argument("--projection-dim", type=int, default=128,
                       help="Projection head dimension for contrastive training")
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
    parser.add_argument("--use-audio", action="store_true",
                       help="Enable audio modality during fine-tuning")
    parser.add_argument("--fusion-mode", type=str, default="concat",
                       choices=["concat", "cross_attention", "gated"],
                       help="Fusion strategy to use when audio is enabled")    
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                       help="Path to a fine-tuning checkpoint to resume from (e.g., finetuned_epoch_7.pt)")
    args = parser.parse_args()
    
    # Add timestamp to output dir (unless resuming)
    if args.resume_checkpoint is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.output_dir}/{args.mode}_{timestamp}"
    else:
        # Use the directory of the checkpoint as output_dir
        args.output_dir = str(Path(args.resume_checkpoint).parent)
    
    if args.mode == "pretrain":
        pretrain(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
