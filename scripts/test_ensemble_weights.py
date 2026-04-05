#!/usr/bin/env python3
"""
Test different ensemble weighting schemes to find optimal combination
"""

import json
from pathlib import Path

def evaluate_ensemble_weights():
    """Test different attention_weight values"""
    
    # Load results
    with open('outputs/attention_fusion_20260401_182606/results.json') as f:
        attn_results = json.load(f)
    
    with open('outputs/focal_loss_20260329_073014/results.json') as f:
        focal_results = json.load(f)
    
    emotions = ['Happiness', 'Sadness', 'Anger', 'Calmness', 'Neutral']
    attn_per_class = attn_results['per_class_acc']
    focal_per_class = focal_results['per_class_acc']
    
    print("=" * 70)
    print("🧪 TESTING ENSEMBLE WEIGHT COMBINATIONS")
    print("=" * 70)
    
    # Test different weights
    weights_to_test = [
        (1.0, 0.0),   # 100% Attention (baseline)
        (0.95, 0.05), # 95% Attention, 5% Focal
        (0.90, 0.10), # 90% Attention, 10% Focal
        (0.85, 0.15), # 85% Attention, 15% Focal
        (0.80, 0.20), # 80% Attention, 20% Focal
        (0.75, 0.25), # 75% Attention, 25% Focal
        (0.70, 0.30), # 70% Attention, 30% Focal (current)
        (0.60, 0.40), # 60% Attention, 40% Focal
        (0.50, 0.50), # 50% Attention, 50% Focal
    ]
    
    print("\n📊 Testing Weight Combinations:\n")
    print(f"{'Weights':15s} {'Overall Acc':12s} {'vs Best':10s} {'Status':20s}")
    print("-" * 70)
    
    best_overall_acc = 0
    best_weights = (1.0, 0.0)
    
    for attn_w, focal_w in weights_to_test:
        # Normalize weights
        total = attn_w + focal_w
        attn_w /= total
        focal_w /= total
        
        # Compute ensemble accuracy
        acc_per_class = []
        for emotion in emotions:
            attn_acc = attn_per_class.get(emotion, 0.5)
            focal_acc = focal_per_class.get(emotion, 0.5)
            ensemble_acc = attn_w * attn_acc + focal_w * focal_acc
            acc_per_class.append(ensemble_acc)
        
        overall_acc = sum(acc_per_class) / len(acc_per_class)
        
        # Improvement vs Attention Fusion
        improvement = overall_acc - 0.7857
        status = "✅ IMPROVEMENT" if improvement > 0.001 else ("⭐ BEST" if attn_w == 1.0 else "")
        
        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            best_weights = (attn_w, focal_w)
        
        print(f"{attn_w*100:5.0f}% / {focal_w*100:5.0f}%   {overall_acc*100:6.2f}%   "
              f"{improvement:+6.2f}pp   {status}")
    
    print("\n" + "=" * 70)
    print("🎯 CONCLUSION")
    print("=" * 70)
    
    if best_weights == (1.0, 0.0):
        print("\n✅ BEST RESULT: Pure Attention Fusion (78.57%)")
        print("   → Ensemble with Focal Loss does NOT improve accuracy")
        print("   → Reason: Attention Fusion is already optimal")
        print("\n💡 RECOMMENDATION: Stick with Attention Fusion (78.57%)")
        print("   It's better than any weighted combination with Focal Loss!")
    else:
        print(f"\n✅ BEST RESULT: {best_weights[0]*100:.0f}% Attention + {best_weights[1]*100:.0f}% Focal")
        print(f"   Overall Accuracy: {best_overall_acc*100:.2f}%")
        print(f"   Improvement: +{(best_overall_acc - 0.7857)*100:.2f}pp vs Attention alone")

if __name__ == '__main__':
    evaluate_ensemble_weights()
