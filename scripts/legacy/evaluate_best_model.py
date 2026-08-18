"""
Evaluate the best Focal Loss model on test set.

This script loads and evaluates the Focal Loss model which achieved 63.02% accuracy.
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
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.eeg_encoder import (
    EEGEncoder,
    AudioEncoder,
    EmotionClassifier,
    MultimodalFusion,
)
from src.preprocessing.data_loader import EAVMultimodalDataset


EMOTION_CLASSES = {
    0: 'Neutral',
    1: 'Anger',
    2: 'Calmness',
    3: 'Sadness',
    4: 'Happiness',
}


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Initialize models
    encoder = EEGEncoder()
    audio_encoder = AudioEncoder()
    fusion = MultimodalFusion(mode='gated')
    classifier = EmotionClassifier(num_emotions=5)
    
    # Move to device
    encoder.to(device)
    audio_encoder.to(device)
    fusion.to(device)
    classifier.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    fusion.load_state_dict(checkpoint['fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    return encoder, audio_encoder, fusion, classifier


def evaluate_model(encoder, audio_encoder, fusion, classifier, test_loader, device):
    """Evaluate model on test set."""
    encoder.eval()
    audio_encoder.eval()
    fusion.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}", end='\r')
            
            # Extract batch data
            eeg = batch['eeg'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['emotion'].to(device)
            
            # Forward pass
            eeg_feat = encoder(eeg)
            audio_feat = audio_encoder(audio)
            fused = fusion(eeg_feat, audio_feat)
            logits = classifier(fused)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    return all_labels, all_preds, all_probs


def main():
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    test_indices = indices[train_size + val_size:]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test set: {len(test_indices)} samples")
    
    # Load model
    print("\n" + "="*80)
    print("LOADING FOCAL LOSS MODEL")
    print("="*80)
    
    checkpoint_path = 'outputs/focal_loss_model_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"✗ Error: Checkpoint not found at {checkpoint_path}")
        print(f"\nAvailable checkpoints in outputs/:")
        for f in os.listdir('outputs'):
            if f.endswith('.pt'):
                print(f"  - {f}")
        return
    
    print(f"Loading: {checkpoint_path}")
    encoder, audio_encoder, fusion, classifier = load_model(checkpoint_path, device)
    print("✓ Model loaded successfully")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    print(f"\nEvaluating on test set...")
    all_labels, all_preds, all_probs = evaluate_model(
        encoder, audio_encoder, fusion, classifier, 
        test_loader, device
    )
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'─'*80}")
    print(f"Overall Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'─'*80}")
    
    # Per-class metrics
    print(f"\nPer-Class Accuracy:")
    print(f"{'─'*80}")
    
    per_class_acc = {}
    confusion_mat = confusion_matrix(all_labels, all_preds, labels=list(range(5)))
    
    for class_idx in range(5):
        class_acc = confusion_mat[class_idx, class_idx] / confusion_mat[class_idx].sum()
        per_class_acc[class_idx] = class_acc
        class_name = EMOTION_CLASSES[class_idx]
        class_count = confusion_mat[class_idx].sum()
        print(f"  {class_name:12s}: {class_acc:7.4f} ({class_acc*100:5.2f}%) - {class_count:3d} samples")
    
    # Confusion matrix visualization
    print(f"\n{'─'*80}")
    print("Generating confusion matrix visualization...")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[EMOTION_CLASSES[i] for i in range(5)],
        yticklabels=[EMOTION_CLASSES[i] for i in range(5)],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Focal Loss Model - Confusion Matrix (Test Accuracy: {test_acc*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/focal_loss_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/focal_loss_confusion_matrix.png")
    plt.close()
    
    # Classification report
    print(f"\n{'─'*80}")
    print("Detailed Classification Report:")
    print(f"{'─'*80}")
    print(classification_report(
        all_labels, all_preds,
        target_names=[EMOTION_CLASSES[i] for i in range(5)],
        digits=4
    ))
    
    # Save results
    results = {
        'test_acc': float(test_acc),
        'per_class_acc': {
            EMOTION_CLASSES[k]: float(v) for k, v in per_class_acc.items()
        },
        'confusion_matrix': confusion_mat.tolist(),
        'model': 'Focal Loss CNN',
        'checkpoint': checkpoint_path,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = 'outputs/focal_loss_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {output_file}")
    
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"Model: Focal Loss CNN")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Best Class: {max(per_class_acc.items(), key=lambda x: x[1])[0]} - {EMOTION_CLASSES[max(per_class_acc.items(), key=lambda x: x[1])[0]]}")
    print(f"Weakest Class: {min(per_class_acc.items(), key=lambda x: x[1])[0]} - {EMOTION_CLASSES[min(per_class_acc.items(), key=lambda x: x[1])[0]]}")
    print(f"="*80)


if __name__ == '__main__':
    main()
