"""Advanced training script with LSTM encoder variant and class balancing.

Features:
- Support for both CNN (EEGEncoder) and LSTM (EEGEncoderLSTM) encoders
- Optional weighted CrossEntropyLoss for class balancing
- Extended training with proper validation/test splits
- Per-class accuracy metrics
- Best model tracking and checkpoint saving
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eeg_encoder import (
    EEGEncoder, EEGEncoderLSTM, AudioEncoder, MultimodalFusion
)
from src.preprocessing.data_loader import EAVMultimodalDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse


class MultimodalEmotionModel(nn.Module):
    """Multimodal emotion recognition model with selectable EEG encoder."""
    
    def __init__(self, 
                 eeg_encoder_type='cnn',  # 'cnn' or 'lstm'
                 eeg_latent_dim=128,
                 use_audio=True,
                 fusion_mode='gated'):
        """
        Parameters
        ----------
        eeg_encoder_type : str
            'cnn' for CNN-based or 'lstm' for LSTM-based EEG encoder
        eeg_latent_dim : int
            Latent dimension for EEG encoder
        use_audio : bool
            Whether to include audio modality
        fusion_mode : str
            Fusion strategy: 'gated', 'concat', 'cross_attention'
        """
        super().__init__()
        
        self.eeg_encoder_type = eeg_encoder_type
        self.use_audio = use_audio
        
        # EEG encoder
        if eeg_encoder_type.lower() == 'lstm':
            self.eeg_encoder = EEGEncoderLSTM(
                in_channels=28,
                hidden_dim=64,
                latent_dim=eeg_latent_dim,
                num_layers=2
            )
        else:  # 'cnn' or default
            self.eeg_encoder = EEGEncoder(
                in_channels=28,
                latent_dim=eeg_latent_dim
            )
        
        # Audio encoder
        if use_audio:
            self.audio_encoder = AudioEncoder(n_mfcc=13, latent_dim=128)
            audio_dim = 128
        else:
            audio_dim = 0
        
        # Fusion module
        if fusion_mode == 'gated' and use_audio:
            self.fusion = MultimodalFusion(latent_dim=eeg_latent_dim, fusion_dim=eeg_latent_dim + audio_dim, mode='gated')
            fusion_dim = eeg_latent_dim
        elif fusion_mode == 'cross_attention' and use_audio:
            self.fusion = MultimodalFusion(latent_dim=eeg_latent_dim, fusion_dim=eeg_latent_dim + audio_dim, mode='cross_attention')
            fusion_dim = eeg_latent_dim
        elif use_audio:
            # Concat fusion
            self.fusion = MultimodalFusion(latent_dim=eeg_latent_dim, fusion_dim=eeg_latent_dim + audio_dim, mode='concat')
            fusion_dim = eeg_latent_dim
        else:
            # Only EEG
            self.fusion = None
            fusion_dim = eeg_latent_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)  # 5 emotions
        )
        
        self.fusion_mode = fusion_mode
    
    def forward(self, eeg, audio=None):
        """
        Parameters
        ----------
        eeg : torch.Tensor
            EEG signal (batch_size, 28, time_steps)
        audio : torch.Tensor, optional
            Audio features (batch_size, 13, time_steps)
        
        Returns
        -------
        torch.Tensor
            Logits (batch_size, 5)
        """
        eeg_feat = self.eeg_encoder(eeg)
        
        if self.use_audio and audio is not None and self.fusion is not None:
            audio_feat = self.audio_encoder(audio)
            feat = self.fusion(eeg_feat, audio_feat)
        else:
            feat = eeg_feat
        
        logits = self.classifier(feat)
        return logits


def compute_class_weights(dataset, num_classes=5):
    """Compute weights for weighted cross entropy loss."""
    from collections import Counter
    
    emotion_counts = Counter()
    for sample in dataset.samples:
        emotion = sample['audio_emotion']
        if emotion in dataset.EMOTION_MAP:
            emotion_counts[dataset.EMOTION_MAP[emotion]] += 1
    
    total = sum(emotion_counts.values())
    weights = []
    for cls in range(num_classes):
        count = emotion_counts.get(cls, 1)
        weight = total / (num_classes * count) if count > 0 else 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in train_loader:
        eeg = batch['eeg'].to(device)
        audio = batch['audio'].to(device)
        emotion = batch['emotion'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(eeg, audio)
        loss = criterion(logits, emotion)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == emotion).sum().item()
        total += emotion.size(0)
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            audio = batch['audio'].to(device)
            emotion = batch['emotion'].to(device)
            
            logits = model(eeg, audio)
            loss = criterion(logits, emotion)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            
            correct += (pred == emotion).sum().item()
            total += emotion.size(0)
            
            # Per-class metrics
            for p, t in zip(pred.cpu().numpy(), emotion.cpu().numpy()):
                all_preds.append(p)
                all_labels.append(t)
                per_class_correct[t] += (p == t)
                per_class_total[t] += 1
    
    acc = correct / total if total > 0 else 0
    per_class_acc = {}
    emotion_names = ['Neutral', 'Anger', 'Calmness', 'Sadness', 'Happiness']
    for cls in range(5):
        if per_class_total[cls] > 0:
            per_class_acc[emotion_names[cls]] = per_class_correct[cls] / per_class_total[cls]
    
    return total_loss / len(val_loader), acc, per_class_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train multimodal emotion model with LSTM variant")
    parser.add_argument("--data-dir", default="data/raw/EAV/EAV", help="Path to EAV dataset")
    parser.add_argument("--encoder", choices=['cnn', 'lstm'], default='lstm', 
                       help="EEG encoder type: CNN or LSTM")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--use-audio", action="store_true", default=True, help="Use audio modality")
    parser.add_argument("--fusion", choices=['gated', 'concat'], default='gated', help="Fusion mode")
    parser.add_argument("--use-class-weights", action="store_true", default=False, 
                       help="Use weighted loss (for imbalanced data)")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_name = args.encoder.upper()
    output_dir = Path(args.output_dir) / f"training_{encoder_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining Configuration:")
    print(f"  EEG Encoder: {args.encoder.upper()}")
    print(f"  Fusion Mode: {args.fusion}")
    print(f"  Use Audio: {args.use_audio}")
    print(f"  Use Class Weights: {args.use_class_weights}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Output Dir: {output_dir}\n")
    
    # Load dataset
    print("Loading dataset...")
    from torch.utils.data import DataLoader, random_split
    
    full_dataset = EAVMultimodalDataset(
        eav_data_dir=args.data_dir,
        load_audio=args.use_audio
    )
    
    # Random split
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_loader) * args.batch_size} samples")
    print(f"  Val: {len(val_loader) * args.batch_size} samples")
    print(f"  Test: {len(test_loader) * args.batch_size} samples\n")
    
    # Model
    model = MultimodalEmotionModel(
        eeg_encoder_type=args.encoder,
        eeg_latent_dim=128,
        use_audio=args.use_audio,
        fusion_mode=args.fusion
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss function
    if args.use_class_weights:
        class_weights = compute_class_weights(full_dataset).to(device)
        print(f"Class weights: {class_weights.tolist()}\n")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Training
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    training_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("Starting training...\n")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, per_class_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Per-class: {', '.join(f'{name}: {acc:.3f}' for name, acc in per_class_acc.items())}")
        if val_acc == best_val_acc:
            print(f"  >> NEW BEST (Epoch {best_epoch})")
        print()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\nFinal Test Evaluation:")
    print("-" * 60)
    test_loss, test_acc, test_per_class, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nPer-Class Accuracy:")
    for emotion, acc in test_per_class.items():
        print(f"  {emotion}: {acc:.4f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Save results
    results = {
        'config': {
            'encoder': args.encoder,
            'fusion_mode': args.fusion,
            'use_audio': args.use_audio,
            'use_class_weights': args.use_class_weights,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
        },
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'per_class_acc': test_per_class,
        'confusion_matrix': cm.tolist(),
        'training_history': {k: [float(v) for v in vs] for k, vs in training_history.items()}
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_file = output_dir / "best_model.pt"
    torch.save(best_model_state, model_file)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  results.json")
    print(f"  best_model.pt")


if __name__ == "__main__":
    main()
