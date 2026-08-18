"""Analyze class distribution in EAV dataset to detect imbalance."""

import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import EAVMultimodalDataset
import argparse


def analyze_class_balance(data_dir: str, subjects=None):
    """Analyze emotion class distribution in EAV dataset."""
    
    print(f"\n{'='*60}")
    print("CLASS IMBALANCE ANALYSIS - EAV DATASET")
    print(f"{'='*60}\n")
    
    # Load dataset (fast scan without loading audio)
    dataset = EAVMultimodalDataset(
        eav_data_dir=data_dir,
        subjects=subjects,
        load_audio=False  # Don't load audio to speed up scanning
    )
    
    print(f"Total samples: {len(dataset)}\n")
    
    # Count emotion labels
    emotion_counts = Counter()
    emotion_map_reverse = {
        0: 'Neutral',
        1: 'Anger',
        2: 'Calmness',
        3: 'Sadness',
        4: 'Happiness'
    }
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        emotion = sample['emotion']
        emotion_counts[emotion] += 1
    
    # Print statistics
    total_samples = len(dataset)
    print("Emotion Distribution:")
    print("-" * 60)
    print(f"{'Emotion':<15} {'Count':<10} {'Percentage':<15} {'Weight (inverse)':<15}")
    print("-" * 60)
    
    weights = {}
    for emotion_id in sorted(emotion_counts.keys()):
        emotion_name = emotion_map_reverse[emotion_id]
        count = emotion_counts[emotion_id]
        percentage = (count / total_samples) * 100
        # Inverse frequency weighting
        weight = total_samples / (len(emotion_counts) * count)
        weights[emotion_id] = weight
        print(f"{emotion_name:<15} {count:<10} {percentage:>6.2f}%{'':<8} {weight:.4f}")
    
    print("-" * 60)
    print(f"Total: {total_samples}\n")
    
    # Check for imbalance
    counts = list(emotion_counts.values())
    if not counts:
        print("No emotion data found!")
        return {}, {}
    
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print("WARNING: Significant class imbalance detected!")
        print("Recommendation: Use weighted CrossEntropyLoss with computed weights\n")
    else:
        print("INFO: Dataset is relatively balanced\n")
    
    # Print weights for use in training
    print("\nWeights for torch.nn.CrossEntropyLoss:")
    print("-" * 60)
    weight_list = [weights[i] for i in range(5)]
    print(f"class_weights = {weight_list}")
    print(f"or use: class_weights = {[f'{w:.4f}' for w in weight_list]}\n")
    
    # Compute normalized weights (sum to number of classes)
    num_classes = len(emotion_counts)
    total_weight = sum(weight_list)
    normalized_weights = [w * num_classes / total_weight for w in weight_list]
    print(f"Normalized weights (sum={num_classes}):")
    print(f"class_weights = {[f'{w:.4f}' for w in normalized_weights]}\n")
    
    return weights, emotion_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class balance in EAV dataset")
    parser.add_argument(
        "--data-dir",
        default="data/raw/EAV",
        help="Path to EAV dataset directory"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Specific subjects to analyze (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        analyze_class_balance(args.data_dir, subjects=args.subjects)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
