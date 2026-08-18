"""
Train CNN baseline and save checkpoint for ensemble evaluation.
Replicates train_final.py but with checkpoint saving.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.eeg_encoder import (
    EEGEncoder, AudioEncoder, EmotionClassifier, MultimodalFusion
)
from src.preprocessing.data_loader import EAVMultimodalDataset


EMOTION_CLASSES = {
    0: 'Neutral',
    1: 'Anger',
    2: 'Calmness',
    3: 'Sadness',
    4: 'Happiness',
}


def train_cnn_baseline():
    """Train CNN baseline with checkpoint saving."""
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading EAV dataset...")
    eav_data_dir = Path('data/raw/EAV/EAV')
    dataset = EAVMultimodalDataset(
        str(eav_data_dir),
        load_audio=True,
        load_video=False,
        normalize_eeg=True
    )
    print(f"Total samples: {len(dataset)}")
    
    # Split into train/val/test (70/15/15)
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Build models
    print("\n" + "="*70)
    print("TRAINING CNN BASELINE")
    print("="*70)
    
    encoder = EEGEncoder(in_channels=28, latent_dim=128).to(device)
    audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128).to(device)
    fusion = MultimodalFusion(mode='gated', latent_dim=128).to(device)
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(audio_encoder.parameters()) +
        list(fusion.parameters()) +
        list(classifier.parameters()),
        lr=2e-4,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop - simplified (20 epochs for speed)
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(20):
        # Train
        encoder.train()
        audio_encoder.train()
        fusion.train()
        classifier.train()
        
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            eeg = batch['eeg'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['emotion'].to(device)
            
            eeg_feat = encoder(eeg)
            audio_feat = audio_encoder(audio)
            fused = fusion(eeg_feat, audio_feat)
            logits = classifier(fused)
            
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) +
                list(audio_encoder.parameters()) +
                list(fusion.parameters()) +
                list(classifier.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        encoder.eval()
        audio_encoder.eval()
        fusion.eval()
        classifier.eval()
        
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['emotion'].to(device)
                
                eeg_feat = encoder(eeg)
                audio_feat = audio_encoder(audio)
                fused = fusion(eeg_feat, audio_feat)
                logits = classifier(fused)
                
                preds = torch.argmax(logits, dim=1)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:2d}/{20} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = 'outputs/cnn_baseline_model_best.pt'
            torch.save({
                'encoder': encoder.state_dict(),
                'audio_encoder': audio_encoder.state_dict(),
                'fusion': fusion.state_dict(),
                'classifier': classifier.state_dict(),
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(test_loader)}", end='\r')
            
            eeg = batch['eeg'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['emotion'].to(device)
            
            eeg_feat = encoder(eeg)
            audio_feat = audio_encoder(audio)
            fused = fusion(eeg_feat, audio_feat)
            logits = classifier(fused)
            probs = torch.softmax(logits, dim=1)
            
            preds = torch.argmax(probs, dim=1)
            test_preds.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_probs.append(probs.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    test_probs = np.concatenate(test_probs)
    
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\n✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Per-class metrics
    print(f"\nPer-Class Accuracy:")
    per_class_acc = {}
    conf_matrix = confusion_matrix(test_labels, test_preds, labels=list(range(5)))
    
    for class_idx in range(5):
        class_acc = conf_matrix[class_idx, class_idx] / conf_matrix[class_idx].sum()
        per_class_acc[EMOTION_CLASSES[class_idx]] = class_acc
        print(f"  {EMOTION_CLASSES[class_idx]:12s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Save results
    results = {
        'test_acc': float(test_acc),
        'best_val_acc': float(best_val_acc),
        'best_epoch': best_epoch,
        'per_class_acc': per_class_acc,
        'confusion_matrix': conf_matrix.tolist(),
        'model': 'CNN_Baseline',
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = f'outputs/cnn_baseline_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Checkpoint saved to: outputs/cnn_baseline_model_best.pt")
    
    return test_acc, per_class_acc


if __name__ == '__main__':
    train_cnn_baseline()
