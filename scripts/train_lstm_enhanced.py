"""Enhanced training with LSTM encoder, extended epochs, and class weighting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import models
from src.models.eeg_encoder import EEGEncoderLSTM, AudioEncoder, MultimodalFusion, EmotionClassifier
from src.preprocessing.data_loader import EAVMultimodalDataset

# Emotion mapping
EMOTION_CLASSES = ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']

def compute_class_weights(dataset, num_classes=5):
    """Compute inverse frequency weights for class balancing."""
    class_counts = np.zeros(num_classes)
    
    for sample in dataset.samples:
        emotion = sample.get('audio_emotion')
        if emotion in EMOTION_CLASSES:
            idx = EMOTION_CLASSES.index(emotion)
            class_counts[idx] += 1
    
    # Inverse frequency: classes with fewer samples get higher weight
    weights = np.zeros(num_classes)
    total = class_counts.sum()
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total / (num_classes * class_counts[i])
        else:
            weights[i] = 1.0
    
    # Normalize to average 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_with_lstm():
    """Train with LSTM encoder, class weights, and extended epochs."""
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Dataset
    eav_data_dir = Path('data/raw/EAV/EAV')
    print(f"\nLoading EAV dataset from {eav_data_dir}...")
    
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
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Compute class weights for loss function
    class_weights = compute_class_weights(dataset, num_classes=5)
    print(f"\nClass weights: {class_weights.numpy()}")
    
    # Models
    print("\n" + "="*80)
    print("BUILDING MODELS WITH LSTM ENCODER")
    print("="*80)
    
    encoder = EEGEncoderLSTM(
        in_channels=28,
        hidden_dim=128,
        num_layers=2,
        latent_dim=128
    ).to(device)
    
    audio_encoder = AudioEncoder(
        n_mfcc=13,
        latent_dim=128
    ).to(device)
    
    fusion = MultimodalFusion(latent_dim=128, mode='gated').to(device)
    
    classifier = EmotionClassifier(
        latent_dim=128,
        num_emotions=5
    ).to(device)
    
    # Optimizer with weight decay
    params = list(encoder.parameters()) + \
             list(audio_encoder.parameters()) + \
             list(fusion.parameters()) + \
             list(classifier.parameters())
    
    optimizer = optim.Adam(params, lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Loss with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Audio encoder params: {sum(p.numel() for p in audio_encoder.parameters()):,}")
    print(f"Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")
    print(f"Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")
    
    total_params = sum(p.numel() for p in (list(encoder.parameters()) + 
                                           list(audio_encoder.parameters()) + 
                                           list(fusion.parameters()) + 
                                           list(classifier.parameters())))
    print(f"Total params: {total_params:,}")
    
    # Training loop - EXTENDED TO 50 EPOCHS
    num_epochs = 50
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    max_patience = 5
    
    print(f"\n" + "="*80)
    print(f"TRAINING WITH LSTM (50 epochs, class-weighted loss)")
    print("="*80)
    
    for epoch in range(num_epochs):
        # Training
        encoder.train()
        audio_encoder.train()
        fusion.train()
        classifier.train()
        
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                if isinstance(batch, dict):
                    eeg = batch.get('eeg')
                    audio = batch.get('audio')
                    labels = batch.get('emotion')
                    
                    if eeg is None or labels is None:
                        continue
                else:
                    continue
                
                eeg = eeg.to(device)
                audio = audio.to(device) if audio is not None else None
                labels = labels.to(device)
                
                # Forward pass
                eeg_latent = encoder(eeg)
                
                if audio is not None:
                    audio_latent = audio_encoder(audio)
                    fused = fusion(eeg_latent, audio_latent)
                else:
                    fused = eeg_latent
                
                logits = classifier(fused)
                loss = loss_fn(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_samples += labels.size(0)
                
            except Exception as e:
                print(f"[Epoch {epoch+1}] Skip batch: {str(e)[:50]}")
                continue
        
        train_acc = train_correct / train_samples if train_samples > 0 else 0
        train_loss /= max(1, train_samples)
        
        # Validation
        encoder.eval()
        audio_encoder.eval()
        fusion.eval()
        classifier.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    if isinstance(batch, dict):
                        eeg = batch.get('eeg')
                        audio = batch.get('audio')
                        labels = batch.get('emotion')
                        
                        if eeg is None or labels is None:
                            continue
                    else:
                        continue
                    
                    eeg = eeg.to(device)
                    audio = audio.to(device) if audio is not None else None
                    labels = labels.to(device)
                    
                    eeg_latent = encoder(eeg)
                    
                    if audio is not None:
                        audio_latent = audio_encoder(audio)
                        fused = fusion(eeg_latent, audio_latent)
                    else:
                        fused = eeg_latent
                    
                    logits = classifier(fused)
                    loss = loss_fn(logits, labels)
                    
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += labels.size(0)
                    
                except Exception as e:
                    continue
        
        val_acc = val_correct / val_samples if val_samples > 0 else 0
        val_loss /= max(1, val_samples)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save({
                'encoder': encoder.state_dict(),
                'audio_encoder': audio_encoder.state_dict(),
                'fusion': fusion.state_dict(),
                'classifier': classifier.state_dict(),
            }, f'outputs/lstm_model_best.pt')
        else:
            patience_counter += 1
        
        # Print progress
        status = "[OK]"
        if epoch % 5 == 0 or epoch < 10:
            print(f"[Epoch {epoch+1:2d}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= max_patience:
            print(f"\n[EARLY STOP] Epoch {epoch+1}: validation acc plateaued for {max_patience} epochs")
            break
    
    # Test evaluation
    print(f"\n" + "="*80)
    print(f"TESTING (Best model from epoch {best_epoch})")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load('outputs/lstm_model_best.pt')
    encoder.load_state_dict(checkpoint['encoder'])
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    fusion.load_state_dict(checkpoint['fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    encoder.eval()
    audio_encoder.eval()
    fusion.eval()
    classifier.eval()
    
    test_correct = 0
    test_samples = 0
    all_preds = []
    all_labels = []
    per_class_correct = np.zeros(5)
    per_class_total = np.zeros(5)
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                if isinstance(batch, dict):
                    eeg = batch.get('eeg')
                    audio = batch.get('audio')
                    labels = batch.get('emotion')
                    
                    if eeg is None or labels is None:
                        continue
                else:
                    continue
                
                eeg = eeg.to(device)
                audio = audio.to(device) if audio is not None else None
                labels = labels.to(device)
                
                eeg_latent = encoder(eeg)
                
                if audio is not None:
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
                
                # Per-class metrics
                for i in range(5):
                    mask = labels == i
                    if mask.sum() > 0:
                        per_class_correct[i] += (preds[mask] == i).sum().item()
                        per_class_total[i] += mask.sum().item()
                
            except Exception as e:
                continue
    
    test_acc = test_correct / test_samples if test_samples > 0 else 0
    
    print(f"\n======================================================================")
    print(f"BEST VALIDATION ACCURACY: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"======================================================================")
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    print(f"\n======================================================================")
    print(f"PER-CLASS ACCURACY (LSTM Enhanced)")
    print(f"======================================================================")
    
    per_class_acc = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        if per_class_total[i] > 0:
            acc = per_class_correct[i] / per_class_total[i]
            per_class_acc[emotion] = acc
            print(f"  {emotion:12s}: {acc:.4f} ({int(per_class_total[i])} samples)")
        else:
            per_class_acc[emotion] = 0.0
            print(f"  {emotion:12s}: N/A")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(5)))
    
    print(f"\n======================================================================")
    print(f"SUMMARY")
    print(f"======================================================================")
    print(f"Best Validation Acc: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    improvement = (test_acc - 0.20) * 100
    print(f"Improvement vs baseline: +{improvement:.1f}%")
    
    # Save results
    results_dir = Path(f"outputs/lstm_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'arch': 'LSTM_encoder',
        'fusion_mode': 'gated',
        'num_epochs': num_epochs,
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'per_class_acc': {k: float(v) for k, v in per_class_acc.items()},
        'confusion_matrix': conf_matrix.tolist(),
        'class_weights': class_weights.cpu().numpy().tolist(),
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Output directory: {results_dir}")


if __name__ == '__main__':
    train_with_lstm()
