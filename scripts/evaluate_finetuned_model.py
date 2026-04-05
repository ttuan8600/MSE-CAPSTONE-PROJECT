"""
Comprehensive evaluation of finetuned vs baseline Attention Fusion model
Compares test set accuracy, per-class metrics, and generates comparison report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, EmotionClassifier
from src.models.attention_fusion import CrossModalAttentionFusion
from src.preprocessing.data_loader import create_eav_dataloader


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    model_dict = {
        'encoder': EEGEncoder(),
        'audio_encoder': AudioEncoder(),
        'attention_fusion': CrossModalAttentionFusion(),
        'classifier': EmotionClassifier(),
    }
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key, model in model_dict.items():
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
        model.to(device)
        model.eval()
    
    return model_dict


def evaluate_model(model_dict, test_loader, device, model_name):
    """Evaluate model on test set"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUATING {model_name}")
    print(f"{'='*70}\n")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    emotion_names = {0: 'Neutral', 1: 'Anger', 2: 'Calmness', 3: 'Sadness', 4: 'Happiness'}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            eeg = batch['eeg'].to(device)
            audio = batch['audio'].to(device)
            
            # Handle both 'label' and 'emotion' keys
            if 'label' in batch:
                labels = batch['label'].cpu().numpy()
            else:
                labels = batch['emotion'].cpu().numpy()
            
            # Forward pass
            eeg_features = model_dict['encoder'](eeg)
            audio_features = model_dict['audio_encoder'](audio)
            fused_features = model_dict['attention_fusion'](eeg_features, audio_features)
            logits = model_dict['classifier'](fused_features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)} processed")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_acc = {}
    for emotion_id in range(5):
        mask = all_labels == emotion_id
        if mask.sum() > 0:
            per_class_acc[emotion_names[emotion_id]] = accuracy_score(
                all_labels[mask], all_preds[mask]
            ) * 100
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(5))
    
    print(f"✅ Test Set Results:")
    print(f"  Overall Accuracy:  {test_acc*100:.2f}%")
    print(f"  Macro Precision:   {precision*100:.2f}%")
    print(f"  Macro Recall:      {recall*100:.2f}%")
    print(f"  Macro F1-Score:    {f1:.4f}")
    print(f"\n📊 Per-Class Accuracy:")
    for emotion, acc in per_class_acc.items():
        print(f"  {emotion:12s}: {acc:.2f}%")
    
    print(f"\n🔍 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=[emotion_names[i] for i in range(5)]))
    
    return {
        'overall_acc': test_acc * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test data
    print("\n📂 Loading dataset...")
    
    try:
        eav_data_dir = "data/raw/EAV/EAV"
        
        # Load full dataset
        full_loader, full_dataset = create_eav_dataloader(
            eav_data_dir=eav_data_dir,
            batch_size=32,
            num_workers=0,
            shuffle=False,
            load_audio=True,
        )
        
        dataset_size = len(full_dataset)
        print(f"  ✅ Loaded {dataset_size} samples from {eav_data_dir}")
        
        # Split data (same split as training)
        train_size = int(0.70 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create test dataloader
        test_loader = DataLoader(
            test_ds,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"✅ Test set loaded: {len(test_loader)} batches ({test_size} samples)")
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Evaluate baseline model
    print("\n1️⃣  BASELINE MODEL")
    baseline_results = evaluate_model(
        load_model('outputs/attention_fusion_model_best.pt', device),
        test_loader,
        device,
        'BASELINE MODEL (78.57% validation)'
    )
    
    # Evaluate finetuned model
    print("\n2️⃣  FINETUNED MODEL")
    finetuned_results = evaluate_model(
        load_model('outputs/attention_fusion_finetuned_best.pt', device),
        test_loader,
        device,
        'FINETUNED MODEL (82.06% validation)'
    )
    
    # Comparison
    print(f"\n{'='*70}")
    print("📈 COMPARISON: FINETUNED vs BASELINE")
    print(f"{'='*70}\n")
    
    improvement = finetuned_results['overall_acc'] - baseline_results['overall_acc']
    print(f"Test Accuracy Improvement: {improvement:+.2f}pp")
    print(f"  Baseline:  {baseline_results['overall_acc']:.2f}%")
    print(f"  Finetuned: {finetuned_results['overall_acc']:.2f}%")
    
    print(f"\nPer-Class Improvements:")
    for emotion in baseline_results['per_class_acc'].keys():
        baseline_acc = baseline_results['per_class_acc'][emotion]
        finetuned_acc = finetuned_results['per_class_acc'][emotion]
        delta = finetuned_acc - baseline_acc
        print(f"  {emotion:12s}: {baseline_acc:6.2f}% → {finetuned_acc:6.2f}% ({delta:+6.2f}pp)")
    
    # Save comparison results
    comparison_report = {
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_results,
        'finetuned': finetuned_results,
        'improvement_pp': improvement,
        'status': 'SUCCESS' if improvement > 0 else 'UNCHANGED'
    }
    
    output_path = f"outputs/comparison_finetuned_vs_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print(f"\n💾 Comparison report saved: {output_path}")
    
    # Deployment recommendation
    print(f"\n{'='*70}")
    print("🚀 DEPLOYMENT RECOMMENDATION")
    print(f"{'='*70}\n")
    
    if improvement >= 0.5:
        print("✅ RECOMMENDED: Deploy finetuned model")
        print(f"   Reason: {improvement:.2f}pp improvement over baseline")
        print(f"   New accuracy: {finetuned_results['overall_acc']:.2f}%")
        print(f"\n   Actions:")
        print(f"   1. Backup baseline: cp outputs/attention_fusion_model_best.pt outputs/attention_fusion_model_baseline_backup.pt")
        print(f"   2. Replace with finetuned: cp outputs/attention_fusion_finetuned_best.pt outputs/attention_fusion_model_best.pt")
        print(f"   3. Update production deployment")
    else:
        print("⚠️  MARGINALLY IMPROVED: Baseline remains preferred")
        print(f"   Reason: Only {improvement:.2f}pp improvement (minimal)")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
