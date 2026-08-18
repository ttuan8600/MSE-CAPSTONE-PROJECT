"""Systematic comparison of EEG-only vs EEG+Audio on real EAV data.

This script trains both configurations on actual EAV data and saves results
for careful comparison and ablation analysis.

Usage:
    python scripts/compare_modalities.py [--num-epochs 10] [--batch-size 16] [--quick]

The `--quick` flag uses a small subset for fast iteration.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eeg_encoder import EEGEncoder, AudioEncoder, MultimodalFusion, EmotionClassifier
from src.preprocessing.data_loader import create_eav_dataloader


def train_configuration(
    dataloader: DataLoader,
    use_audio: bool,
    device: torch.device,
    num_epochs: int,
    config_name: str,
    fusion_mode: str = "concat",
) -> dict:
    """Train a single configuration and return metrics."""
    
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"Modalities: {'EEG + Audio' if use_audio else 'EEG only'}")
    print(f"Epochs: {num_epochs}, Device: {device}")
    try:
        dataset_size = len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else 'Unknown'
    except (TypeError, AttributeError):
        dataset_size = 'Unknown'
    print(f"Dataset size: {dataset_size}")
    print(f"{'='*70}\n")
    
    # Initialize models
    encoder = EEGEncoder(in_channels=28, latent_dim=128).to(device)
    classifier = EmotionClassifier(latent_dim=128, num_emotions=5).to(device)
    
    params = [*encoder.parameters(), *classifier.parameters()]
    
    if use_audio:
        audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128).to(device)
        fusion = MultimodalFusion(latent_dim=128, mode=fusion_mode).to(device)
        params.extend(audio_encoder.parameters())
        params.extend(fusion.parameters())
    else:
        audio_encoder = None
        fusion = None
    
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    metrics = {
        'config': config_name,
        'use_audio': use_audio,
        'epochs': [],
        'batch_losses': [],
        'total_samples': 0,
        'total_batches': 0,
        'successful_batches': 0,
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        epoch_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Extract data
                if isinstance(batch, (tuple, list)):
                    batch_dict, labels = batch
                else:
                    batch_dict = batch
                    if 'label' not in batch_dict:
                        continue
                    labels = batch_dict['label']
                
                eeg = batch_dict['eeg'].to(device)
                labels = labels.to(device) if isinstance(labels, torch.Tensor) else None
                
                if labels is None or eeg.shape[0] == 0:
                    continue
                
                # Forward pass
                eeg_latent = encoder(eeg)
                
                if use_audio and 'audio' in batch_dict and batch_dict['audio'] is not None and fusion is not None and audio_encoder is not None:
                    audio = batch_dict['audio'].to(device)
                    audio_latent = audio_encoder(audio)
                    fused = fusion(eeg_latent, audio_latent)
                else:
                    fused = eeg_latent
                
                logits = classifier(fused)
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                # Metrics
                loss_val = loss.item()
                epoch_loss += loss_val * eeg.shape[0]
                pred = logits.argmax(dim=1)
                epoch_correct += (pred == labels).sum().item()
                epoch_samples += eeg.shape[0]
                epoch_batches += 1
                
                metrics['batch_losses'].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss_val,
                    'batch_size': eeg.shape[0]
                })
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1} [{batch_idx}] Loss: {loss_val:.4f}")
                
                metrics['successful_batches'] += 1
            
            except Exception as e:
                print(f"  ⚠ Batch {batch_idx} error: {type(e).__name__}: {str(e)[:50]}")
                continue
        
        # Epoch summary
        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            avg_acc = epoch_correct / epoch_samples
            metrics['epochs'].append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': avg_acc,
                'samples': epoch_samples,
                'batches': epoch_batches
            })
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f} | "
                  f"Samples: {epoch_samples}\n")
        
        scheduler.step()
        metrics['total_samples'] += epoch_samples
        metrics['total_batches'] += epoch_batches
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compare EEG-only and EEG+Audio on real EAV data"
    )
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--quick", action="store_true", 
                        help="Quick test on small subset")
    parser.add_argument("--fusion-mode", type=str, default="concat",
                        choices=["concat", "cross_attention", "gated"],
                        help="Fusion strategy when audio is used")
    parser.add_argument("--eav-dir", type=str, default="data/raw/EAV/EAV",
                        help="Path to EAV dataset")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated subject IDs (e.g., '1,2,3')")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eav_path = Path(args.eav_dir)
    
    if not eav_path.exists():
        print(f"❌ EAV dataset not found at {args.eav_dir}")
        print("   Please ensure the EAV data is properly located.")
        return
    
    # Parse subjects if specified
    subjects = None
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(',')]
    
    # Load dataloaders
    print("Loading EAV data...")
    try:
        eeg_loader, eeg_dataset = create_eav_dataloader(
            eav_data_dir=args.eav_dir,
            batch_size=args.batch_size,
            shuffle=True,
            load_audio=False,
            subjects=subjects,
        )
        print(f"✓ EEG-only dataset: {len(eeg_dataset)} samples")
        
        audio_loader, audio_dataset = create_eav_dataloader(
            eav_data_dir=args.eav_dir,
            batch_size=args.batch_size,
            shuffle=True,
            load_audio=True,
            subjects=subjects,
        )
        print(f"✓ EEG+Audio dataset: {len(audio_dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load EAV data: {e}")
        return
    
    # Quick mode: use smaller subset
    num_epochs = 2 if args.quick else args.num_epochs
    
    # Train both configurations
    results = {}
    
    results['eeg_only'] = train_configuration(
        eeg_loader,
        use_audio=False,
        device=device,
        num_epochs=num_epochs,
        config_name="EEG-only Baseline",
        fusion_mode=args.fusion_mode,
    )
    
    results['eeg_audio'] = train_configuration(
        audio_loader,
        use_audio=True,
        device=device,
        num_epochs=num_epochs,
        config_name="EEG+Audio Fusion",
        fusion_mode=args.fusion_mode,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"outputs/comparison_{timestamp}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for config_name, config_results in results.items():
        if config_results['epochs']:
            final_epoch = config_results['epochs'][-1]
            print(f"\n{config_name.upper().replace('_', ' ')}:")
            print(f"  Final Loss:     {final_epoch['loss']:.4f}")
            print(f"  Final Accuracy: {final_epoch['accuracy']:.4f}")
            print(f"  Total samples:  {config_results['total_samples']}")
    
    # Compute delta
    if results['eeg_only']['epochs'] and results['eeg_audio']['epochs']:
        eeg_acc = results['eeg_only']['epochs'][-1]['accuracy']
        audio_acc = results['eeg_audio']['epochs'][-1]['accuracy']
        delta = audio_acc - eeg_acc
        pct_change = 100 * delta / eeg_acc if eeg_acc > 0 else 0
        
        print(f"\nAudio Impact:")
        print(f"  Accuracy delta: {delta:+.4f} ({pct_change:+.1f}%)")
        if delta > 0:
            print(f"  ✓ Audio IMPROVED accuracy")
        elif delta < 0:
            print(f"  ✗ Audio DECREASED accuracy")
        else:
            print(f"  ≈ No change")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
