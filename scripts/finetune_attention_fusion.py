#!/usr/bin/env python3
"""
Fine-tune Attention Fusion with Data Augmentation
Uses the best checkpoint and trains with augmentation for potential +1-3% improvement
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, EmotionClassifier
from src.models.attention_fusion import CrossModalAttentionFusion
from src.preprocessing.data_loader import create_eav_dataloader


class AugmentedEAVDataset(Dataset):
    """EAV dataset with augmentation"""
    
    def __init__(self, base_dataset, augment=True, mixup_alpha=0.2):
        self.base_dataset = base_dataset
        self.augment = augment
        self.mixup_alpha = mixup_alpha
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _spec_augment_audio(self, audio):
        """SpecAugment: time/frequency masking on audio"""
        if not self.augment:
            return audio
        
        # Random frequency masking
        if np.random.rand() > 0.5:
            freq_mask_width = np.random.randint(1, 3)
            freq_mask_start = np.random.randint(0, max(1, audio.shape[0] - freq_mask_width))
            audio[freq_mask_start:freq_mask_start + freq_mask_width, :] = 0
        
        # Random time masking
        if np.random.rand() > 0.5:
            time_mask_width = np.random.randint(1, 8)
            time_mask_start = np.random.randint(0, max(1, audio.shape[1] - time_mask_width))
            audio[:, time_mask_start:time_mask_start + time_mask_width] = 0
        
        return audio
    
    def _eeg_jitter(self, eeg):
        """Add small noise to EEG"""
        if not self.augment or np.random.rand() > 0.5:
            return eeg
        
        noise = np.random.normal(0, 0.01, eeg.shape)
        return eeg + noise
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        eeg = sample['eeg'].numpy() if isinstance(sample['eeg'], torch.Tensor) else sample['eeg']
        audio = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
        
        # Handle both 'label' and 'emotion' key names
        if 'label' in sample:
            label = sample['label']
        else:
            label = sample['emotion']
        
        # Apply augmentation
        if self.augment:
            eeg = self._eeg_jitter(eeg)
            audio = self._spec_augment_audio(audio.copy())
        
        return {
            'eeg': torch.FloatTensor(eeg),
            'audio': torch.FloatTensor(audio),
            'emotion': label,
            'label': label
        }


def train_epoch(model_dict, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    encoder, audio_encoder, attention_fusion, classifier = (
        model_dict['encoder'],
        model_dict['audio_encoder'],
        model_dict['attention_fusion'],
        model_dict['classifier']
    )
    
    for model in [encoder, audio_encoder, attention_fusion, classifier]:
        model.train()
    
    total_loss = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        eeg = batch['eeg'].to(device)
        audio = batch['audio'].to(device)
        
        # Handle both 'label' and 'emotion' key names
        if 'label' in batch:
            labels = batch['label'].to(device)
        else:
            labels = batch['emotion'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        eeg_feat = encoder(eeg)
        audio_feat = audio_encoder(audio)
        fused = attention_fusion(eeg_feat, audio_feat)
        logits = classifier(fused)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(audio_encoder.parameters()) +
            list(attention_fusion.parameters()) +
            list(classifier.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        total_loss += loss.item() * labels.shape[0]
        total_samples += labels.shape[0]
        
        if (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / total_samples
            print(f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
    
    return total_loss / total_samples


def validate(model_dict, val_loader, criterion, device):
    """Validate model"""
    encoder, audio_encoder, attention_fusion, classifier = (
        model_dict['encoder'],
        model_dict['audio_encoder'],
        model_dict['attention_fusion'],
        model_dict['classifier']
    )
    
    for model in [encoder, audio_encoder, attention_fusion, classifier]:
        model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            audio = batch['audio'].to(device)
            
            # Handle both 'label' and 'emotion' key names
            if 'label' in batch:
                labels = batch['label'].to(device)
            else:
                labels = batch['emotion'].to(device)
            
            eeg_feat = encoder(eeg)
            audio_feat = audio_encoder(audio)
            fused = attention_fusion(eeg_feat, audio_feat)
            logits = classifier(fused)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    
    return total_loss / total, correct / total


def main():
    print("=" * 70)
    print("🔧 FINE-TUNE ATTENTION FUSION WITH DATA AUGMENTATION")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    # Load checkpoint
    print("\n📥 Loading checkpoint...")
    checkpoint_path = Path('outputs/attention_fusion_model_best.pt')
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✅ Checkpoint loaded")
    
    # Create models
    print("\n🏗️  Creating models...")
    encoder = EEGEncoder().to(device)
    audio_encoder = AudioEncoder().to(device)
    attention_fusion = CrossModalAttentionFusion().to(device)
    classifier = EmotionClassifier().to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder'])
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    attention_fusion.load_state_dict(checkpoint['attention_fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    print(f"✅ Models initialized with pretrained weights")
    
    # Load real EAV data
    print("\n📊 Loading real EAV dataset...")
    
    try:
        eav_data_dir = "data/raw/EAV/EAV"
        
        # Load dataset
        full_loader, full_dataset = create_eav_dataloader(
            eav_data_dir=eav_data_dir,
            batch_size=32,
            num_workers=0,
            shuffle=True,
            load_audio=True,
        )
        
        dataset_size = len(full_dataset)
        print(f"  ✅ Loaded {dataset_size} samples from {eav_data_dir}")
        
        # Split data
        train_size = int(0.70 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Wrap training data with augmentation
        print("  Adding augmentation to training data...")
        augmented_train = AugmentedEAVDataset(train_ds, augment=True)
        
        train_loader = DataLoader(
            augmented_train,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"✅ Data split: {train_size} training, {val_size} validation, {test_size} test")
    
    except Exception as e:
        print(f"⚠️  Error loading real data: {e}")
        print("  Falling back to synthetic data...")
        
        num_train_samples = 200
        num_val_samples = 50
        
        class SyntheticDataset(Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return {
                    'eeg': torch.randn(28, 512),
                    'audio': torch.randn(13, 44),
                    'emotion': idx % 5,
                    'label': idx % 5
                }
        
        train_dataset = SyntheticDataset(num_train_samples)
        val_dataset = SyntheticDataset(num_val_samples)
        augmented_train = AugmentedEAVDataset(train_dataset, augment=True)
        train_loader = DataLoader(augmented_train, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        print(f"✅ Synthetic data created: {len(augmented_train)} training, {len(val_dataset)} validation")
    
    # Set up training
    model_dict = {
        'encoder': encoder,
        'audio_encoder': audio_encoder,
        'attention_fusion': attention_fusion,
        'classifier': classifier
    }
    
    # Focal Loss with alpha weighting
    class_weights = torch.tensor([1.0, 1.0, 1.5, 1.5, 1.0])
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, logits, labels):
            ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
            probs = torch.exp(-ce_loss)
            focal_loss = self.alpha[labels] * (1 - probs) ** self.gamma * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(class_weights.to(device))
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(audio_encoder.parameters()) +
        list(attention_fusion.parameters()) +
        list(classifier.parameters()),
        lr=1e-4,  # Lower than 2e-4 used in initial training
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("🚀 TRAINING")
    print("=" * 70)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 5
    num_epochs = 20
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📍 Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model_dict, train_loader, optimizer, criterion, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc = validate(model_dict, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            best_checkpoint = {
                'encoder': encoder.state_dict(),
                'audio_encoder': audio_encoder.state_dict(),
                'attention_fusion': attention_fusion.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }
            
            best_path = Path('outputs/attention_fusion_finetuned_best.pt')
            torch.save(best_checkpoint, best_path)
            print(f"  ✅ Best checkpoint saved ({val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break
        
        scheduler.step(val_acc)
    
    # Load best model and test
    print("\n" + "=" * 70)
    print("✅ FINE-TUNING COMPLETE")
    print("=" * 70)
    
    print(f"\n🎯 Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"\nImprovement vs Baseline (78.57%): {(best_val_acc - 0.7857)*100:+.2f}pp")
    
    # Compare results
    if best_val_acc > 0.7857:
        print(f"✅ SUCCESS! Fine-tuning improved accuracy by {(best_val_acc - 0.7857)*100:.2f}pp")
    else:
        print(f"⚠️  Fine-tuning did not improve validation accuracy")
    
    print(f"\n💾 Best checkpoint: outputs/attention_fusion_finetuned_best.pt")


if __name__ == '__main__':
    main()
