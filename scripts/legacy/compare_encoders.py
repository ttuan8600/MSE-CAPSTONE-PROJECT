"""Compare CNN vs LSTM encoder performance on EAV dataset.

Runs training with both encoder types and compares results.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_training(encoder_type, epochs=20):
    """Run training script with specified encoder type."""
    print(f"\n{'='*70}")
    print(f"Starting {encoder_type.upper()} Encoder Training")
    print(f"{'='*70}\n")
    
    cmd = [
        ".venv\\Scripts\\python.exe",
        "scripts\\train_lstm_variant.py",
        "--encoder", encoder_type,
        "--epochs", str(epochs),
        "--batch-size", "32",
        "--lr", "2e-4",
        "--fusion", "gated",
        "--use-audio",
        "--device", "cpu"
    ]
    
    result = subprocess.run(cmd, cwd="c:\\Users\\ttuan8600\\Documents\\MyProjects\\MSE-CAPSTONE-PROJECT")
    return result.returncode == 0


def find_latest_results(pattern):
    """Find the latest results file matching pattern."""
    output_dir = Path("outputs")
    matches = sorted(output_dir.glob(f"{pattern}*/results.json"), 
                    key=lambda p: p.parent.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def compare_results():
    """Compare results from CNN and LSTM training."""
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}\n")
    
    cnn_results_file = find_latest_results("training_CNN_*")
    lstm_results_file = find_latest_results("training_LSTM_*")
    
    if not cnn_results_file or not lstm_results_file:
        print("Error: Could not find results files")
        return
    
    with open(cnn_results_file) as f:
        cnn_results = json.load(f)
    
    with open(lstm_results_file) as f:
        lstm_results = json.load(f)
    
    # Print comparison table
    print(f"{'Metric':<30} {'CNN':<15} {'LSTM':<15} {'Difference':<15}")
    print("-" * 75)
    
    cnn_test_acc = cnn_results['test_acc']
    lstm_test_acc = lstm_results['test_acc']
    diff = lstm_test_acc - cnn_test_acc
    
    print(f"{'Test Accuracy':<30} {cnn_test_acc:>6.4f}{'':<8} {lstm_test_acc:>6.4f}{'':<8} {diff:>+6.4f}")
    
    cnn_best_epoch = cnn_results['best_epoch']
    lstm_best_epoch = lstm_results['best_epoch']
    print(f"{'Best Epoch':<30} {cnn_best_epoch:>6d}{'':<8} {lstm_best_epoch:>6d}")
    
    print("\nPer-Class Test Accuracy Comparison:")
    print("-" * 75)
    print(f"{'Emotion':<20} {'CNN':<15} {'LSTM':<15} {'Difference':<15}")
    print("-" * 75)
    
    for emotion in ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']:
        cnn_class_acc = cnn_results['per_class_acc'].get(emotion, 0)
        lstm_class_acc = lstm_results['per_class_acc'].get(emotion, 0)
        diff = lstm_class_acc - cnn_class_acc
        print(f"{emotion:<20} {cnn_class_acc:>6.4f}{'':<8} {lstm_class_acc:>6.4f}{'':<8} {diff:>+6.4f}")
    
    print("\nRecommendation:")
    if lstm_test_acc > cnn_test_acc:
        improvement = (lstm_test_acc - cnn_test_acc) / cnn_test_acc * 100
        print(f"[LSTM Recommended] +{improvement:.2f}% improvement over CNN")
    elif lstm_test_acc < cnn_test_acc:
        improvement = (cnn_test_acc - lstm_test_acc) / lstm_test_acc * 100
        print(f"[CNN Recommended] +{improvement:.2f}% improvement over LSTM")
    else:
        print("[Tie] Both encoders achieve equal performance")
    
    print(f"\nDetailed results:")
    print(f"  CNN: {cnn_results_file}")
    print(f"  LSTM: {lstm_results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare CNN vs LSTM encoder variants")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--compare-only", action="store_true", help="Only compare existing results")
    
    args = parser.parse_args()
    
    if not args.compare_only:
        # Run both trainings
        cnn_success = run_training("cnn", epochs=args.epochs)
        lstm_success = run_training("lstm", epochs=args.epochs)
        
        if not cnn_success or not lstm_success:
            print("\nWarning: One or more training runs failed")
    
    # Compare results
    compare_results()
