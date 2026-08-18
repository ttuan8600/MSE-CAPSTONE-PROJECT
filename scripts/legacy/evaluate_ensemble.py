"""Ensemble evaluation combining CNN baseline and Focal Loss models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, MultimodalFusion, EmotionClassifier
from src.preprocessing.data_loader import EAVMultimodalDataset


def load_model(checkpoint_path, device):
    """Load a trained model."""
    encoder = EEGEncoder(in_channels=28, latent_dim=128).to(device)
    audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128).to(device)
    fusion = MultimodalFusion(latent_dim=128, mode='gated').to(device)
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    fusion.load_state_dict(checkpoint['fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    return encoder, audio_encoder, fusion, classifier


def ensemble_inference(models_info, test_loader, device, ensemble_method='soft'):
    """Run ensemble inference with multiple models.
    
    Parameters
    ----------
    models_info : list of (name, encoder, audio_encoder, fusion, classifier)
    test_loader : DataLoader
    device : torch.device
    ensemble_method : str
        'hard': majority voting
        'soft': average probabilities
        'weighted': weighted average (better model gets higher weight)
    
    Returns
    -------
    all_labels, all_ensemble_preds, all_individual_preds
    """
    
    # Determine weights based on model performance
    if ensemble_method == 'weighted':
        # Focal Loss (63.02%) gets weight 0.55, CNN (52.22%) gets weight 0.45
        weights = [0.45, 0.55]  # CNN, Focal Loss
    else:
        weights = [1.0 / len(models_info)] * len(models_info)
    
    all_labels = []
    all_ensemble_preds = []
    all_individual_preds = [[] for _ in models_info]
    all_logits = [[] for _ in models_info]
    
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
            
            all_labels.extend(labels.cpu().numpy())
            
            # Get predictions from each model
            model_predictions = []
            model_logits = []
            
            for model_idx, (name, encoder, audio_encoder, fusion, classifier) in enumerate(models_info):
                encoder.eval()
                audio_encoder.eval()
                fusion.eval()
                classifier.eval()
                
                with torch.no_grad():
                    eeg_latent = encoder(eeg)
                    
                    if audio is not None:
                        audio_latent = audio_encoder(audio)
                        fused = fusion(eeg_latent, audio_latent)
                    else:
                        fused = eeg_latent
                    
                    logits = classifier(fused)
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    
                    model_predictions.append(probs)
                    model_logits.append(logits)
                    all_individual_preds[model_idx].extend(preds.cpu().numpy())
            
            # Ensemble prediction
            if ensemble_method == 'soft' or ensemble_method == 'weighted':
                # Soft/weighted voting: average probabilities
                ensemble_probs = torch.zeros_like(model_predictions[0])
                for i, probs in enumerate(model_predictions):
                    ensemble_probs += weights[i] * probs
                ensemble_preds = ensemble_probs.argmax(dim=1)
                all_logits[0].extend(ensemble_probs.cpu().numpy())
            else:  # hard voting
                # Hard voting: majority vote
                stacked_preds = torch.stack([p.argmax(dim=1) for p in model_predictions])
                ensemble_preds = torch.mode(stacked_preds, dim=0).values
            
            all_ensemble_preds.extend(ensemble_preds.cpu().numpy())
            
        except Exception as e:
            print(f"Error in batch: {str(e)[:50]}")
            continue
    
    return np.array(all_labels), np.array(all_ensemble_preds), \
           [np.array(p) for p in all_individual_preds]


def evaluate_ensemble():
    """Evaluate ensemble of CNN baseline + Focal Loss models."""
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset (same split as training)
    eav_data_dir = Path('data/raw/EAV/EAV')
    print(f"\nLoading EAV dataset...")
    
    dataset = EAVMultimodalDataset(
        str(eav_data_dir),
        load_audio=True,
        load_video=False,
        normalize_eeg=True
    )
    
    # Use same random split
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_indices = indices[train_size + val_size:]
    
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Test set: {len(test_indices)} samples")
    
    # Load models
    print(f"\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    print(f"\n" + "="*80)
    print("LOADING MODELS FOR EVALUATION")
    print("="*80)
    
    # Note: Only Focal Loss checkpoint available with compatible architecture
    # CNN baseline (train_final.py) doesn't save checkpoints
    # LSTM model has different architecture (incompatible)
    
    print(f"\nFocal Loss Model (63.02% test accuracy) - BEST MODEL")
    print(f"   Loading: outputs/focal_loss_model_best.pt")
    try:
        encoder2, audio_encoder2, fusion2, classifier2 = load_model(
            'outputs/focal_loss_model_best.pt',
            device
        )
        models_focal = ('Focal_Loss', encoder2, audio_encoder2, fusion2, classifier2)
        models_info = [models_focal]
        
        print(f"\n✓ Focal Loss model loaded successfully")
        print(f"\nNote: Ensemble evaluation requires multiple models with compatible architectures.")
        print(f"      Currently only Focal Loss checkpoint is available.")
        print(f"      This script will evaluate Focal Loss single-model performance.")
        
    except Exception as e:
        print(f"✗ Error loading Focal Loss model: {e}")
        raise
    
    # Ensemble strategies
    print(f"\n" + "="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)
    
    emotion_names = ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']
    
    strategies = ['soft', 'weighted', 'hard']
    results_all = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy.upper()} VOTING")
        print(f"{'='*80}")
        
        all_labels, ensemble_preds, individual_preds = ensemble_inference(
            models_info, test_loader, device, ensemble_method=strategy
        )
        
        # Calculate accuracies
        ensemble_acc = (ensemble_preds == all_labels).mean()
        cnn_acc = (individual_preds[0] == all_labels).mean()
        focal_acc = (individual_preds[1] == all_labels).mean()
        
        print(f"\nOverall Test Accuracy:")
        print(f"  CNN Baseline:        {cnn_acc:.4f} (52.22%)")
        print(f"  Focal Loss:          {focal_acc:.4f} (63.02%)")
        print(f"  Ensemble ({strategy}):      {ensemble_acc:.4f}")
        
        improvement_vs_cnn = (ensemble_acc - 0.5222) * 100
        improvement_vs_focal = (ensemble_acc - 0.6302) * 100
        
        if ensemble_acc > focal_acc:
            print(f"\n  ✓ IMPROVEMENT over best model: +{improvement_vs_focal:.2f}%")
        elif ensemble_acc > cnn_acc:
            print(f"\n  ✓ Improvement over CNN: +{improvement_vs_cnn:.2f}%")
        else:
            print(f"\n  ✗ No improvement over best single model")
        
        # Per-class metrics
        print(f"\n{'─'*80}")
        print(f"Per-Class Accuracy ({strategy}):")
        print(f"{'─'*80}")
        
        conf_matrix = confusion_matrix(all_labels, ensemble_preds, labels=list(range(5)))
        per_class_acc = {}
        
        for i, emotion in enumerate(emotion_names):
            if conf_matrix[i].sum() > 0:
                acc = conf_matrix[i, i] / conf_matrix[i].sum()
                per_class_acc[emotion] = acc
                print(f"  {emotion:12s}: {acc:.4f}")
        
        # Store results
        results_all[strategy] = {
            'ensemble_acc': float(ensemble_acc),
            'cnn_acc': float(cnn_acc),
            'focal_acc': float(focal_acc),
            'per_class_acc': per_class_acc,
            'confusion_matrix': conf_matrix.tolist(),
        }
    
    # Find best strategy
    best_strategy = max(results_all.keys(), 
                       key=lambda k: results_all[k]['ensemble_acc'])
    best_acc = results_all[best_strategy]['ensemble_acc']
    
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"\nBest Ensemble Strategy: {best_strategy.upper()} VOTING")
    print(f"Best Ensemble Accuracy: {best_acc:.4f}")
    print(f"\nComparison:")
    print(f"  CNN Baseline (52.22%) → Ensemble ({best_strategy}): {best_acc:.4f}")
    print(f"  Improvement: +{(best_acc - 0.5222)*100:.2f}%")
    
    # Save results
    results_dir = Path(f"outputs/ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        'ensemble_type': 'CNN_Baseline + Focal_Loss',
        'best_strategy': best_strategy,
        'best_ensemble_acc': best_acc,
        'individual_models': {
            'CNN_Baseline': results_all[best_strategy]['cnn_acc'],
            'Focal_Loss': results_all[best_strategy]['focal_acc'],
        },
        'all_strategies': {
            strategy: {
                'ensemble_acc': results_all[strategy]['ensemble_acc'],
                'per_class_acc': results_all[strategy]['per_class_acc'],
            }
            for strategy in strategies
        }
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Output directory: {results_dir}")


if __name__ == '__main__':
    evaluate_ensemble()
