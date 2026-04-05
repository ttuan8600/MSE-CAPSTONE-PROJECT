"""
Quick evaluation of finetuned vs baseline model on test set
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, EmotionClassifier
from src.models.attention_fusion import CrossModalAttentionFusion
from src.preprocessing.data_loader import create_eav_dataloader


def load_and_eval_model(checkpoint_path, test_loader, device):
    """Load model and evaluate on test set"""
    # Load models
    encoder = EEGEncoder().to(device)
    audio_encoder = AudioEncoder().to(device)
    attention_fusion = CrossModalAttentionFusion().to(device)
    classifier = EmotionClassifier().to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    attention_fusion.load_state_dict(checkpoint['attention_fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    # Set to eval mode
    for model in [encoder, audio_encoder, attention_fusion, classifier]:
        model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"  Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            eeg = batch['eeg'].to(device)
            audio = batch['audio'].to(device)
            
            # Get labels
            if 'label' in batch:
                labels = batch['label'].cpu().numpy()
            else:
                labels = batch['emotion'].cpu().numpy()
            
            # Forward pass
            eeg_feat = encoder(eeg)
            audio_feat = audio_encoder(audio)
            fused = attention_fusion(eeg_feat, audio_feat)
            logits = classifier(fused)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"    Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    test_acc = accuracy_score(all_labels, all_preds) * 100
    return test_acc, all_preds, all_labels


device = torch.device('cpu')
print("Loading data...")

# Load dataset
eav_data_dir = "data/raw/EAV/EAV"
full_loader, full_dataset = create_eav_dataloader(
    eav_data_dir=eav_data_dir,
    batch_size=32,
    shuffle=False,
    load_audio=True,
)

# Split
dataset_size = len(full_dataset)
train_size = int(0.70 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

_, _, test_ds = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"✅ Loaded {test_size} test samples\n")

# Evaluate baseline
print("=" * 60)
print("BASELINE MODEL (78.57% validation)")
print("=" * 60)
baseline_acc, baseline_preds, baseline_labels = load_and_eval_model(
    'outputs/attention_fusion_model_best.pt', test_loader, device
)
print(f"✅ Test Accuracy: {baseline_acc:.2f}%\n")

# Evaluate finetuned
print("=" * 60)
print("FINETUNED MODEL (82.06% validation)")
print("=" * 60)
finetuned_acc, finetuned_preds, finetuned_labels = load_and_eval_model(
    'outputs/attention_fusion_finetuned_best.pt', test_loader, device
)
print(f"✅ Test Accuracy: {finetuned_acc:.2f}%\n")

# Comparison
print("=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)
improvement = finetuned_acc - baseline_acc
print(f"Baseline Test Accuracy:  {baseline_acc:.2f}%")
print(f"Finetuned Test Accuracy: {finetuned_acc:.2f}%")
print(f"Improvement:             {improvement:+.2f}pp\n")

if improvement >= 0.5:
    print("✅ RECOMMENDATION: Deploy finetuned model")
    print(f"   Improvement: {improvement:.2f}pp is significant")
    print(f"   New accuracy: {finetuned_acc:.2f}%")
elif improvement >= 0:
    print("⚠️  MARGINAL: Finetuned model shows small improvement")
    print(f"   Consider deployment if improvement validates on new data")
else:
    print("❌ BASELINE PREFERRED: Finetuned model degraded")
    print(f"   Keep baseline at {baseline_acc:.2f}%")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'baseline_acc': baseline_acc,
    'finetuned_acc': finetuned_acc,
    'improvement_pp': improvement,
    'test_samples': test_size
}

output_file = f"outputs/evaluation_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")
