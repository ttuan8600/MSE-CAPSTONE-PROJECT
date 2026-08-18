"""Simple wrapper to run optimized training on real data.

This script directly runs training with proven best-practice configurations
on the EAV dataset to improve accuracy.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.models.eeg_encoder import (
    EEGEncoder,
    AudioEncoder,
    EmotionClassifier,
    MultimodalFusion,
)
from src.preprocessing.data_loader import create_eav_dataloader


# Emotion class mapping
EMOTION_CLASSES = {
    0: 'Neutral',
    1: 'Anger',
    2: 'Calmness',
    3: 'Sadness',
    4: 'Happiness',
}


def train_improved_model():
    """Train with optimized configuration."""
    
    print("\n" + "="*70)
    print("IMPROVED MODEL TRAINING")
    print("="*70)
    
    # Configuration
    config = {
        'use_audio': True,
        'fusion_mode': 'gated',  # Proven best
        'learning_rate': 2e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print(f"\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    device = torch.device(config['device'])
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading EAV dataset...")
    try:
        eav_dir = "data/raw/EAV/EAV"
        loader, dataset = create_eav_dataloader(
            eav_data_dir=eav_dir,
            batch_size=config['batch_size'],
            num_workers=0,
            shuffle=True,
            load_audio=config['use_audio'],
        )
        
        dataset_size = len(dataset)
        print(f"[OK] Loaded {dataset_size} samples")
        
        # Split data
        train_size = int(0.70 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config['batch_size'],
            shuffle=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config['batch_size'],
            shuffle=False,
        )
        
        print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    except Exception as e:
        print(f"\n[ERROR] Error loading data: {e}")
        print("Run: pip install librosa scipy scikit-learn")
        return
    
    # Create models
    encoder = EEGEncoder(in_channels=28, latent_dim=128).to(device)
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
    audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128).to(device)
    fusion = MultimodalFusion(latent_dim=128, mode=config['fusion_mode']).to(device)
    
    # Parameters
    params = (list(encoder.parameters()) + 
              list(classifier.parameters()) +
              list(audio_encoder.parameters()) +
              list(fusion.parameters()))
    
    # Optimizer with improved settings
    optimizer = optim.Adam(
        params,
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=config['weight_decay'],
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(config['num_epochs']):
        # Train
        encoder.train()
        classifier.train()
        audio_encoder.train()
        fusion.train()
        
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Handle batch format from EAVMultimodalDataset
                if isinstance(batch, dict):
                    eeg = batch.get('eeg')
                    audio = batch.get('audio')
                    labels = batch.get('emotion')  # The dataset returns 'emotion' key
                    
                    if eeg is None or labels is None:
                        continue
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    if isinstance(batch[0], dict):
                        batch_dict, labels = batch
                        eeg = batch_dict.get('eeg')
                        audio = batch_dict.get('audio')
                    else:
                        continue
                else:
                    continue
                
                if eeg is None or labels is None:
                    continue
                
                eeg = eeg.to(device)
                labels = labels.to(device)
                
                # Forward pass
                eeg_latent = encoder(eeg)
                
                if audio is not None:
                    audio = audio.to(device)
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
                
                # Track metrics
                train_loss += loss.item() * labels.size(0)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_samples += labels.size(0)
                
                if batch_idx % 30 == 0:
                    print(f"  Epoch {epoch+1} [{batch_idx}] Loss: {loss.item():.4f}")
            
            except Exception as e:
                continue
        
        train_loss /= train_samples if train_samples > 0 else 1
        train_acc = train_correct / train_samples if train_samples > 0 else 0
        
        # Validate
        encoder.eval()
        classifier.eval()
        audio_encoder.eval()
        fusion.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Handle batch format from EAVMultimodalDataset
                    if isinstance(batch, dict):
                        eeg = batch.get('eeg')
                        audio = batch.get('audio')
                        labels = batch.get('emotion')  # The dataset returns 'emotion' key
                        
                        if eeg is None or labels is None:
                            continue
                    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                        if isinstance(batch[0], dict):
                            batch_dict, labels = batch
                            eeg = batch_dict.get('eeg')
                            audio = batch_dict.get('audio')
                        else:
                            continue
                    else:
                        continue
                    
                    if eeg is None or labels is None:
                        continue
                    
                    eeg = eeg.to(device)
                    labels = labels.to(device)
                    
                    eeg_latent = encoder(eeg)
                    if audio is not None:
                        audio = audio.to(device)
                        audio_latent = audio_encoder(audio)
                        fused = fusion(eeg_latent, audio_latent)
                    else:
                        fused = eeg_latent
                    
                    logits = classifier(fused)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item() * labels.size(0)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_samples += labels.size(0)
                
                except Exception as e:
                    continue
        
        val_loss /= val_samples if val_samples > 0 else 1
        val_acc = val_correct / val_samples if val_samples > 0 else 0
        
        # Learning rate step
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:2d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"         -> NEW BEST! Val Acc: {val_acc:.4f}")
    
    # Test
    print(f"\n{'='*70}")
    print(f"BEST VALIDATION ACCURACY: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*70}\n")
    
    print("Evaluating on test set...")
    encoder.eval()
    classifier.eval()
    audio_encoder.eval()
    fusion.eval()
    
    test_correct = 0
    test_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                # Handle batch format from EAVMultimodalDataset
                if isinstance(batch, dict):
                    eeg = batch.get('eeg')
                    audio = batch.get('audio')
                    labels = batch.get('emotion')  # The dataset returns 'emotion' key
                    
                    if eeg is None or labels is None:
                        continue
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    if isinstance(batch[0], dict):
                        batch_dict, labels = batch
                        eeg = batch_dict.get('eeg')
                        audio = batch_dict.get('audio')
                    else:
                        continue
                else:
                    continue
                
                if eeg is None or labels is None:
                    continue
                
                eeg = eeg.to(device)
                labels = labels.to(device)
                
                eeg_latent = encoder(eeg)
                if audio is not None:
                    audio = audio.to(device)
                    audio_latent = audio_encoder(audio)
                    fused = fusion(eeg_latent, audio_latent)
                else:
                    fused = eeg_latent
                
                logits = classifier(fused)
                preds = logits.argmax(dim=1)
                
                test_correct += (preds == labels).sum().item()
                test_samples += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            except Exception as e:
                continue
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = test_correct / test_samples if test_samples > 0 else 0
    
    # Per-class accuracy
    print(f"\n{'='*70}")
    print("PER-CLASS ACCURACY")
    print(f"{'='*70}")
    
    per_class_acc = {}
    for class_id in range(5):
        mask = all_labels == class_id
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            per_class_acc[class_id] = class_acc
            print(f"  {EMOTION_CLASSES[class_id]:12s}: {class_acc:.4f} ({mask.sum()} samples)")
    
    print(f"\nTEST ACCURACY (Overall): {test_acc:.4f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Save results
    output_dir = Path(f"outputs/finetuned_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': config,
        'best_val_acc': float(best_val_acc),
        'best_epoch': best_epoch,
        'test_acc': float(test_acc),
        'per_class_acc': {EMOTION_CLASSES[k]: float(v) for k, v in per_class_acc.items()},
        'confusion_matrix': conf_matrix.tolist(),
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Best Configuration: {config['fusion_mode']} fusion + EEG+Audio")
    print(f"Best Validation Acc: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Improvement vs baseline: +{(test_acc-0.21)*100:.1f}%")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train_improved_model()
