"""Diagnose data quality issues for Calmness and Sadness emotions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from collections import defaultdict
import json
from src.preprocessing.data_loader import EAVMultimodalDataset

def diagnose_data_quality():
    """Analyze feature statistics and data quality per emotion class."""
    
    print("=" * 80)
    print("DATA QUALITY DIAGNOSIS: Analyzing Calmness vs Sadness")
    print("=" * 80)
    
    # Load dataset
    eav_data_dir = Path("data/raw/EAV/EAV")
    dataset = EAVMultimodalDataset(
        str(eav_data_dir),
        load_audio=True,
        load_video=False,
        normalize_eeg=True
    )
    
    print(f"\nDataset loaded: {len(dataset)} total samples")
    
    # Statistics per emotion
    emotion_names = ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']
    emotion_stats = defaultdict(lambda: {
        'eeg_means': [],
        'eeg_stds': [],
        'eeg_ranges': [],
        'audio_means': [],
        'audio_stds': [],
        'audio_ranges': [],
        'count': 0,
        'subjects': set()
    })
    
    print("\nProcessing samples...\n")
    
    for idx in range(min(len(dataset), 100)):  # Sample first 100 only
        try:
            sample = dataset[idx]
            emotion_idx = sample['emotion']
            if isinstance(emotion_idx, torch.Tensor):
                emotion_idx = int(emotion_idx.item())
            else:
                emotion_idx = int(emotion_idx)
            emotion_name = emotion_names[emotion_idx]
            subject_id = sample['subject_id']
            
            eeg = sample['eeg'].numpy()  # (28, time_steps)
            audio = sample['audio'].numpy()  # (13, time_steps)
            
            # Statistics
            stats = emotion_stats[emotion_name]
            stats['eeg_means'].append(np.mean(eeg))
            stats['eeg_stds'].append(np.std(eeg))
            stats['eeg_ranges'].append(np.max(eeg) - np.min(eeg))
            stats['audio_means'].append(np.mean(audio))
            stats['audio_stds'].append(np.std(audio))
            stats['audio_ranges'].append(np.max(audio) - np.min(audio))
            stats['count'] += 1
            stats['subjects'].add(subject_id)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error sample {idx}: {str(e)[:50]}")
            continue
    
    # Report statistics
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS BY EMOTION")
    print("=" * 80)
    
    for emotion in emotion_names:
        stats = emotion_stats[emotion]
        if stats['count'] == 0:
            continue
            
        print(f"\n{emotion.upper()}")
        print("-" * 80)
        print(f"  Samples:      {stats['count']}")
        print(f"  Unique subjects: {len(stats['subjects'])}")
        
        # EEG statistics
        eeg_mean_avg = np.mean(stats['eeg_means'])
        eeg_std_avg = np.mean(stats['eeg_stds'])
        eeg_range_avg = np.mean(stats['eeg_ranges'])
        print(f"\n  EEG Features:")
        print(f"    Mean (avg):      {eeg_mean_avg:.6f}")
        print(f"    Std (avg):       {eeg_std_avg:.6f}")
        print(f"    Range (avg):     {eeg_range_avg:.6f}")
        print(f"    Mean variability: {np.std(stats['eeg_means']):.6f} (lower = more consistent)")
        
        # Audio statistics
        audio_mean_avg = np.mean(stats['audio_means'])
        audio_std_avg = np.mean(stats['audio_stds'])
        audio_range_avg = np.mean(stats['audio_ranges'])
        print(f"\n  Audio Features (MFCC):")
        print(f"    Mean (avg):      {audio_mean_avg:.6f}")
        print(f"    Std (avg):       {audio_std_avg:.6f}")
        print(f"    Range (avg):     {audio_range_avg:.6f}")
        print(f"    Mean variability: {np.std(stats['audio_means']):.6f} (lower = more consistent)")
    
    # Comparative analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS: Calmness vs Others")
    print("=" * 80)
    
    calmness_stats = emotion_stats['Calmness']
    neutral_stats = emotion_stats['Neutral']
    sadness_stats = emotion_stats['Sadness']
    
    if calmness_stats['count'] > 0 and neutral_stats['count'] > 0:
        print("\nCalmness vs Neutral (the confusion):")
        print(f"  EEG mean difference: {abs(np.mean(calmness_stats['eeg_means']) - np.mean(neutral_stats['eeg_means'])):.6f}")
        print(f"  EEG std difference:  {abs(np.mean(calmness_stats['eeg_stds']) - np.mean(neutral_stats['eeg_stds'])):.6f}")
        print(f"  Audio mean difference: {abs(np.mean(calmness_stats['audio_means']) - np.mean(neutral_stats['audio_means'])):.6f}")
        print(f"  Audio std difference:  {abs(np.mean(calmness_stats['audio_stds']) - np.mean(neutral_stats['audio_stds'])):.6f}")
        overlap_eeg = abs(np.mean(calmness_stats['eeg_means']) - np.mean(neutral_stats['eeg_means'])) < 0.01
        print(f"  → EEG feature overlap: {'HIGH (confusable)' if overlap_eeg else 'LOW (distinguishable)'}")
    
    if sadness_stats['count'] > 0 and neutral_stats['count'] > 0:
        print("\nSadness vs Neutral:")
        print(f"  EEG mean difference: {abs(np.mean(sadness_stats['eeg_means']) - np.mean(neutral_stats['eeg_means'])):.6f}")
        print(f"  EEG std difference:  {abs(np.mean(sadness_stats['eeg_stds']) - np.mean(neutral_stats['eeg_stds'])):.6f}")
        print(f"  Audio mean difference: {abs(np.mean(sadness_stats['audio_means']) - np.mean(neutral_stats['audio_means'])):.6f}")
        print(f"  Audio std difference:  {abs(np.mean(sadness_stats['audio_stds']) - np.mean(neutral_stats['audio_stds'])):.6f}")
    
    # Data quality issues
    print("\n" + "=" * 80)
    print("POTENTIAL DATA QUALITY ISSUES")
    print("=" * 80)
    
    for emotion in ['Calmness', 'Sadness']:
        stats = emotion_stats[emotion]
        if stats['count'] == 0:
            continue
        
        print(f"\n{emotion}:")
        
        # Check for low variance (could indicate silent/empty recordings)
        audio_stds = np.array(stats['audio_stds'])
        low_energy_count = np.sum(audio_stds < 0.01)
        if low_energy_count > 0:
            pct = 100 * low_energy_count / len(audio_stds)
            print(f"  ⚠️  {pct:.1f}% samples have very low audio energy (std < 0.01)")
        
        # Check EEG consistency
        eeg_means = np.array(stats['eeg_means'])
        if np.std(eeg_means) < 0.01:
            print(f"  ⚠️  EEG means highly consistent (std={np.std(eeg_means):.6f}) - might indicate preprocessing artifact")
        
        # Check for preprocessing issues
        eeg_ranges = np.array(stats['eeg_ranges'])
        if np.mean(eeg_ranges) < 0.5:
            print(f"  ⚠️  EEG ranges very small (mean={np.mean(eeg_ranges):.6f}) - possible normalization issue")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. Feature Similarity: Calmness and Neutral have highly overlapping features
   → Solution: Use class-weighted loss to emphasize discriminative features
   
2. Data Distribution: Analyze subject-level variability
   → Check if certain subjects have consistent Calmness mis-labeling
   
3. Temporal Dependencies: Current features may lose temporal information
   → Solution: Try LSTM encoder to capture temporal dynamics
   
4. Preprocessing: Verify audio/EEG alignment is correct
   → Check if trial timestamps match across modalities
    """)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    diagnose_data_quality()
