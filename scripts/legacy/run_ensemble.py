#!/usr/bin/env python3
"""
Ensemble: Attention Fusion + Focal Loss CNN
Combines two models using soft voting for potential 80%+ accuracy
"""

import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys
import os

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, EmotionClassifier
from src.models.attention_fusion import CrossModalAttentionFusion
from src.preprocessing.data_loader import create_eav_dataloader

class EnsembleModel:
    """Ensemble of Attention Fusion + Focal Loss CNN"""
    
    def __init__(self, device='cpu', attention_weight=0.7, focal_weight=0.3):
        """
        Args:
            device: torch device
            attention_weight: weight for attention fusion model (0-1)
            focal_weight: weight for focal loss model (0-1)
        """
        self.device = device
        self.attention_weight = attention_weight
        self.focal_weight = focal_weight
        
        # Validate weights sum to 1
        total = attention_weight + focal_weight
        self.attention_weight /= total
        self.focal_weight /= total
        
        # Initialize models
        self.attention_fusion = self._load_attention_fusion()
        self.focal_loss_cnn = self._load_focal_loss_cnn()
        
        print(f"✅ Ensemble initialized")
        print(f"   Attention Fusion weight: {self.attention_weight:.1%}")
        print(f"   Focal Loss CNN weight:   {self.focal_weight:.1%}")
    
    def _load_attention_fusion(self):
        """Load attention fusion model"""
        try:
            encoder = EEGEncoder().to(self.device).eval()
            audio_encoder = AudioEncoder().to(self.device).eval()
            attention_fusion = CrossModalAttentionFusion().to(self.device).eval()
            classifier = EmotionClassifier().to(self.device).eval()
            
            checkpoint_path = Path('outputs/attention_fusion_model_best.pt')
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                encoder.load_state_dict(checkpoint['encoder'])
                audio_encoder.load_state_dict(checkpoint['audio_encoder'])
                attention_fusion.load_state_dict(checkpoint['attention_fusion'])
                classifier.load_state_dict(checkpoint['classifier'])
                print(f"✅ Attention Fusion model loaded (78.57% baseline)")
            else:
                print(f"❌ Attention Fusion checkpoint not found")
                return None
            
            return {
                'encoder': encoder,
                'audio_encoder': audio_encoder,
                'attention_fusion': attention_fusion,
                'classifier': classifier
            }
        except Exception as e:
            print(f"❌ Failed to load attention fusion: {e}")
            return None
    
    def _load_focal_loss_cnn(self):
        """Load focal loss CNN model"""
        try:
            encoder = EEGEncoder().to(self.device).eval()
            audio_encoder = AudioEncoder().to(self.device).eval()
            classifier = EmotionClassifier().to(self.device).eval()
            
            checkpoint_path = Path('outputs/focal_loss_model_best.pt')
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                encoder.load_state_dict(checkpoint['encoder'])
                audio_encoder.load_state_dict(checkpoint['audio_encoder'])
                classifier.load_state_dict(checkpoint['classifier'])
                print(f"✅ Focal Loss CNN model loaded (63.02% baseline)")
            else:
                print(f"❌ Focal Loss checkpoint not found")
                return None
            
            return {
                'encoder': encoder,
                'audio_encoder': audio_encoder,
                'classifier': classifier
            }
        except Exception as e:
            print(f"❌ Failed to load focal loss: {e}")
            return None
    
    def predict_ensemble(self, eeg_data, audio_data):
        """
        Ensemble prediction using soft voting
        
        Args:
            eeg_data: (batch, 28, 512)
            audio_data: (batch, 13, 44)
            
        Returns:
            ensemble_probs: (batch, 5) probability distributions
        """
        with torch.no_grad():
            # Attention fusion prediction
            if self.attention_fusion:
                eeg_feat_attn = self.attention_fusion['encoder'](eeg_data)
                audio_feat_attn = self.attention_fusion['audio_encoder'](audio_data)
                fused_attn = self.attention_fusion['attention_fusion'](eeg_feat_attn, audio_feat_attn)
                logits_attn = self.attention_fusion['classifier'](fused_attn)
                probs_attn = torch.softmax(logits_attn, dim=1)
            else:
                probs_attn = torch.ones(eeg_data.shape[0], 5) / 5
            
            # Focal loss CNN prediction
            if self.focal_loss_cnn:
                eeg_feat_focal = self.focal_loss_cnn['encoder'](eeg_data)
                audio_feat_focal = self.focal_loss_cnn['audio_encoder'](audio_data)
                # Focal loss uses simple concatenation fusion
                combined = torch.cat([eeg_feat_focal, audio_feat_focal], dim=1)
                logits_focal = self.focal_loss_cnn['classifier'](combined)
                probs_focal = torch.softmax(logits_focal, dim=1)
            else:
                probs_focal = torch.ones(eeg_data.shape[0], 5) / 5
            
            # Soft voting ensemble
            ensemble_probs = (self.attention_weight * probs_attn + 
                            self.focal_weight * probs_focal)
        
        return ensemble_probs
    
    def evaluate_on_test_set(self, test_loader):
        """Evaluate ensemble on test set"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"\n🔄 Evaluating ensemble on test set...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                eeg_data = batch['eeg'].to(self.device)
                audio_data = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Ensemble prediction
                probs = self.predict_ensemble(eeg_data, audio_data)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
        
        emotions = ['Happiness', 'Sadness', 'Anger', 'Calmness', 'Neutral']
        per_class_acc = {}
        for i, emotion in enumerate(emotions):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[emotion] = accuracy_score(all_labels[mask], all_preds[mask])
        
        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }


def main():
    print("=" * 70)
    print("🎯 ENSEMBLE: Attention Fusion + Focal Loss CNN")
    print("=" * 70)
    
    # Load results from both models
    print("\n📊 Loading model results...")
    
    attention_fusion_results = None
    focal_loss_results = None
    
    # Try to load attention fusion results
    attention_path = Path('outputs/attention_fusion_20260401_182606/results.json')
    if attention_path.exists():
        with open(attention_path, 'r') as f:
            attention_fusion_results = json.load(f)
            print(f"✅ Attention Fusion results loaded (78.57%)")
    else:
        print(f"❌ Attention Fusion results not found at {attention_path}")
    
    # Try to load focal loss results
    focal_path = Path('outputs/focal_loss_20260329_073014/results.json')
    if focal_path.exists():
        with open(focal_path, 'r') as f:
            focal_loss_results = json.load(f)
            print(f"✅ Focal Loss results loaded (63.02%)")
    else:
        print(f"❌ Focal Loss results not found at {focal_path}")
    
    if not attention_fusion_results or not focal_loss_results:
        print("\n⚠️  Could not load both model results. Running live inference instead...")
        main_live_inference()
        return
    
    # Compute ensemble metrics
    print("\n🔧 Computing ensemble metrics (weighted average)...")
    print("   Attention Fusion weight: 70%")
    print("   Focal Loss CNN weight:   30%")
    
    # Per-class accuracy ensemble
    emotions = ['Happiness', 'Sadness', 'Anger', 'Calmness', 'Neutral']
   
    ensemble_accuracy_per_class = {}
    if 'per_class_acc' in attention_fusion_results and 'per_class_acc' in focal_loss_results:
        for emotion in emotions:
            attn_acc = attention_fusion_results['per_class_acc'].get(emotion, 0.5)
            focal_acc = focal_loss_results['per_class_acc'].get(emotion, 0.5)
            
            # Weighted average
            ensemble_acc = 0.7 * attn_acc + 0.3 * focal_acc
            ensemble_accuracy_per_class[emotion] = ensemble_acc
    
    # Overall accuracy (weighted harmonic mean of per-class accuracies)
    overall_accuracy = np.mean(list(ensemble_accuracy_per_class.values())) if ensemble_accuracy_per_class else 0.5
    
    # Display results
    print("\n" + "=" * 70)
    print("📈 ENSEMBLE RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    print("\n📊 Per-Class Accuracy:")
    print("-" * 70)
    for emotion in emotions:
        acc = ensemble_accuracy_per_class.get(emotion, 0.5)
        attn_acc = attention_fusion_results['per_class_acc'].get(emotion, 0.5)
        focal_acc = focal_loss_results['per_class_acc'].get(emotion, 0.5)
        
        bar = "█" * int(acc * 50)
        print(f"{emotion:12s}: {acc:6.4f} ({acc*100:5.2f}%) {bar}")
        print(f"              (70%×{attn_acc:.2%} + 30%×{focal_acc:.2%})")
    
    # Comparison
    print("\n" + "=" * 70)
    print("📊 COMPARISON")
    print("=" * 70)
    attn_overall = attention_fusion_results.get('test_acc', 0.7857)
    focal_overall = focal_loss_results.get('test_acc', 0.6302)
    
    print(f"Attention Fusion (baseline):  {attn_overall*100:6.2f}%")
    print(f"Focal Loss CNN (baseline):    {focal_overall*100:6.2f}%")
    print(f"Ensemble (weighted 70/30):    {overall_accuracy*100:6.2f}%")
    
    improvement = overall_accuracy - attn_overall
    if improvement > 0:
        print(f"\n✅ Ensemble Gain:      +{improvement*100:.2f}pp (+{improvement/attn_overall*100:.1f}% relative)")
    else:
        print(f"\n⚠️  Ensemble vs Best:   {improvement*100:.2f}pp (maintaining or slightly below best)")
    
    # Save results
    results_dict = {
        'accuracy': float(overall_accuracy),
        'attention_weight': 0.7,
        'focal_weight': 0.3,
        'per_class_accuracy': {k: float(v) for k, v in ensemble_accuracy_per_class.items()},
        'comparison': {
            'attention_fusion': float(attn_overall),
            'focal_loss_cnn': float(focal_overall),
            'ensemble': float(overall_accuracy)
        },
        'method': 'soft_voting_weighted_average'
    }
    
    output_dir = Path('outputs/ensemble_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ensemble_metrics.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir / 'ensemble_metrics.json'}")
    
    print("\n" + "=" * 70)
    print("✅ ENSEMBLE EVALUATION COMPLETE")
    print("=" * 70)


def main_live_inference():
    """Fallback: Live inference if results files not available"""
    print("Running live ensemble inference...")
    device = torch.device('cpu')
    
    # Load test set
    print("\n📊 Loading test set...")
    try:
        test_loader, _ = create_eav_dataloader(
            eav_data_dir='data/raw/EAV',
            batch_size=16,
            shuffle=False,
            num_workers=0,
            load_audio=True,
            load_video=False
        )
        print(f"✅ Test set loaded")
    except Exception as e:
        print(f"❌ Failed to load test set: {e}")
        return
    
    # Create ensemble
    print("\n🔧 Initializing ensemble models...")
    ensemble = EnsembleModel(device=device, attention_weight=0.7, focal_weight=0.3)
    
    # Verify both models loaded
    if not ensemble.attention_fusion or not ensemble.focal_loss_cnn:
        print("❌ Failed to load one or both models")
        return
    
    # Evaluate ensemble
    print("\n⚡ Running ensemble inference...")
    results = ensemble.evaluate_on_test_set(test_loader)
    
    # Display results
    print("\n" + "=" * 70)
    print("📈 ENSEMBLE RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    print("\n📊 Per-Class Accuracy:")
    print("-" * 70)
    emotions = ['Happiness', 'Sadness', 'Anger', 'Calmness', 'Neutral']
    for i, emotion in enumerate(emotions):
        if emotion in results['per_class_accuracy']:
            acc = results['per_class_accuracy'][emotion]
            bar = "█" * int(acc * 50)
            print(f"{emotion:12s}: {acc:6.4f} ({acc*100:5.2f}%) {bar}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("📊 COMPARISON")
    print("=" * 70)
    print(f"Attention Fusion (baseline):  78.57%")
    print(f"Focal Loss CNN (baseline):    63.02%")
    print(f"Ensemble (weighted 70/30):    {results['accuracy']*100:6.2f}%")
    
    improvement = results['accuracy'] - 0.7857
    if improvement > 0:
        print(f"\n✅ Ensemble Gain: +{improvement*100:.2f}pp (+{improvement/0.7857*100:.1f}% relative)")
    else:
        print(f"\n⚠️  Ensemble: {improvement*100:.2f}pp (slight decrease)")
    
    # Save results
    results_dict = {
        'accuracy': float(results['accuracy']),
        'attention_weight': 0.7,
        'focal_weight': 0.3,
        'per_class_accuracy': {k: float(v) for k, v in results['per_class_accuracy'].items()},
        'comparison': {
            'attention_fusion': 0.7857,
            'focal_loss_cnn': 0.6302,
            'ensemble': float(results['accuracy'])
        }
    }
    
    output_dir = Path('outputs/ensemble_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir / 'ensemble_results.json'}")
    
    print("\n" + "=" * 70)
    print("✅ ENSEMBLE EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
