"""Enhanced training with Focal Loss for hard-example mining."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, MultimodalFusion, EmotionClassifier
from src.preprocessing.data_loader import EAVMultimodalDataset


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    This loss down-weights easy examples and focuses training on hard negatives,
    which is ideal for the Calmness/Neutral confusion problem.
    
    Parameters
    ----------
    alpha : float or list of floats
        Weighting factor in range (0,1) to balance classes. Higher values 
        give more weight to rare classes.
    gamma : float
        Exponent of the modulating factor (1 - p_t)^gamma to balance 
        easy vs hard examples. gamma=0 is CrossEntropyLoss.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """Compute focal loss.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch_size, num_classes). Logits from classifier.
        targets : torch.LongTensor
            Shape (batch_size,). Ground truth class labels.
        
        Returns
        -------
        torch.Tensor
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_with_focal_loss():
    """Train with Focal Loss and extended epochs (40 epochs)."""
    
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
    
    # Models - using CNN (the better performer)
    print("\n" + "="*80)
    print("BUILDING MODELS WITH CNN ENCODER + FOCAL LOSS")
    print("="*80)
    
    encoder = EEGEncoder(
        in_channels=28,
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
    
    # Optimizer
    params = list(encoder.parameters()) + \
             list(audio_encoder.parameters()) + \
             list(fusion.parameters()) + \
             list(classifier.parameters())
    
    optimizer = optim.Adam(params, lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Focal Loss: alpha weights for class balancing, gamma=2 for hard example mining
    # alpha=[1.0, 1.0, 1.5, 1.5, 1.0] gives more weight to Calmness(2) and Sadness(3)
    focal_loss = FocalLoss(
        alpha=[1.0, 1.0, 1.5, 1.5, 1.0],  # Focus on weak classes
        gamma=2.0,  # Hard example mining factor
        reduction='mean'
    )
    
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Audio encoder params: {sum(p.numel() for p in audio_encoder.parameters()):,}")
    print(f"Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")
    print(f"Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")
    
    total_params = sum(p.numel() for p in params)
    print(f"Total params: {total_params:,}")
    
    # Training loop - 40 epochs with Focal Loss
    num_epochs = 40
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    max_patience = 5
    
    print(f"\n" + "="*80)
    print(f"TRAINING WITH FOCAL LOSS (40 epochs, gamma=2.0, focus on hard examples)")
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
        hard_examples = 0
        
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
                loss = focal_loss(logits, labels)
                
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
                
                # Count hard examples (high loss cases)
                with torch.no_grad():
                    ce_loss = F.cross_entropy(logits, labels, reduction='none')
                    hard_examples += (ce_loss > 1.0).sum().item()
                
            except Exception as e:
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
                    loss = focal_loss(logits, labels)
                    
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
            }, f'outputs/focal_loss_model_best.pt')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 5 == 0 or epoch < 10:
            print(f"[Epoch {epoch+1:2d}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Hard Ex: {hard_examples:4d}, Loss: {train_loss:.4f}")
        
        if patience_counter >= max_patience:
            print(f"\n[EARLY STOP] Epoch {epoch+1}: validation acc plateaued for {max_patience} epochs")
            break
    
    # Test evaluation
    print(f"\n" + "="*80)
    print(f"TESTING (Best model from epoch {best_epoch})")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load('outputs/focal_loss_model_best.pt')
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
    print(f"\nTest Accuracy (Focal Loss): {test_acc:.4f}")
    
    print(f"\n======================================================================")
    print(f"PER-CLASS ACCURACY (FOCAL LOSS)")
    print(f"======================================================================")
    
    emotion_names = ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']
    per_class_acc = {}
    for i, emotion in enumerate(emotion_names):
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
    print(f"\nComparison to CNN Baseline (52.22%):")
    improvement = (test_acc - 0.5222) * 100
    if improvement > 0:
        print(f"  ✓ IMPROVED by +{improvement:.2f}%")
    else:
        print(f"  ✗ Decreased by {improvement:.2f}%")
    
    # Save results
    results_dir = Path(f"outputs/focal_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'arch': 'CNN_encoder',
        'loss_fn': 'FocalLoss',
        'gamma': 2.0,
        'fusion_mode': 'gated',
        'num_epochs': num_epochs,
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'per_class_acc': {k: float(v) for k, v in per_class_acc.items()},
        'confusion_matrix': conf_matrix.tolist(),
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Output directory: {results_dir}")


if __name__ == '__main__':
    train_with_focal_loss()
