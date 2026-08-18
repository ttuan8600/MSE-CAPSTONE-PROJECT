"""Fast class balance analysis by scanning metadata only."""

from pathlib import Path
from collections import Counter
import argparse


def analyze_class_balance_fast(data_dir: str):
    """Quickly analyze class distribution by scanning audio files only."""
    
    print(f"\n{'='*60}")
    print("CLASS IMBALANCE ANALYSIS - EAV DATASET (Fast Scan)")
    print(f"{'='*60}\n")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Path not found: {data_path}")
        return
    
    # Emotion keywords
    emotion_keywords = ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']
    emotion_map = {name: i for i, name in enumerate(emotion_keywords)}
    emotion_counts = Counter()
    
    # Scan all audio files
    audio_files = list(data_path.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files\n")
    
    for audio_file in audio_files:
        filename = audio_file.name
        # Extract emotion from filename (e.g., "002_Trial_02_Speaking_Neutral_Aud.wav")
        for emotion in emotion_keywords:
            if emotion in filename:
                emotion_counts[emotion] += 1
                break
    
    # Print results
    total = sum(emotion_counts.values())
    print("Emotion Distribution:")
    print("-" * 70)
    print(f"{'Emotion':<15} {'Count':<10} {'Percentage':<15} {'Inverse Weight':<20}")
    print("-" * 70)
    
    weights = {}
    for emotion in emotion_keywords:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / total * 100) if total > 0 else 0
        # Inverse frequency weighting
        weight = total / (len(emotion_keywords) * count) if count > 0 else 0
        weights[emotion] = weight
        print(f"{emotion:<15} {count:<10} {percentage:>6.2f}%{'':<8} {weight:.4f}")
    
    print("-" * 70)
    print(f"{'Total':<15} {total:<10}\n")
    
    # Imbalance ratio
    if total > 0:
        counts = [emotion_counts.get(e, 0) for e in emotion_keywords]
        counts_nonzero = [c for c in counts if c > 0]
        if counts_nonzero:
            imbalance_ratio = max(counts_nonzero) / min(counts_nonzero)
            print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
            if imbalance_ratio > 1.5:
                print(">> Significant class imbalance detected!")
                print(">> Use weighted CrossEntropyLoss for training\n")
            else:
                print(">> Dataset is relatively balanced\n")
    
    # Print weights for training
    print("PyTorch Loss Function Configuration:")
    print("-" * 70)
    weight_list = [weights.get(emotion, 1.0) for emotion in emotion_keywords]
    print(f"# Inverse frequency weights:")
    print(f"class_weights = torch.tensor({weight_list})")
    print(f"criterion = nn.CrossEntropyLoss(weight=class_weights)\n")
    
    # Normalized weights
    total_weight = sum(weight_list)
    num_classes = len([w for w in weight_list if w > 0])
    normalized = [w * num_classes / total_weight for w in weight_list]
    print(f"# Normalized weights (sum={num_classes}):")
    print(f"class_weights = torch.tensor({[f'{w:.4f}' for w in normalized]})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast class balance analysis")
    parser.add_argument("--data-dir", default="data/raw/EAV/EAV", help="Path to EAV dataset")
    args = parser.parse_args()
    analyze_class_balance_fast(args.data_dir)
