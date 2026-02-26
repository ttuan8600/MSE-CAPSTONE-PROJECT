"""Utilities for checkpoint management and model evaluation."""

import torch
import os
from pathlib import Path
from typing import Dict, Tuple, Optional


class CheckpointManager:
    """Manage model checkpoints and best model selection."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        models: Dict[str, torch.nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        tag: str = "",
    ) -> str:
        """Save a checkpoint.
        
        Parameters
        ----------
        models : Dict[str, nn.Module]
            Dict of model name -> model instance
        optimizers : Dict[str, Optimizer]
            Dict of optimizer name -> optimizer instance
        epoch : int
            Current epoch
        metrics : Dict[str, float]
            Training metrics (loss, accuracy, etc.)
        tag : str
            Optional tag for checkpoint (e.g., "best")
        
        Returns
        -------
        str
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
        }
        
        # Add models
        for name, model in models.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()
        
        # Add optimizers
        for name, optimizer in optimizers.items():
            checkpoint[f'{name}_state_dict'] = optimizer.state_dict()
        
        # Build filename
        if tag:
            filename = f"{tag}_epoch_{epoch}.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        return str(path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        models: Dict[str, torch.nn.Module],
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict:
        """Load a checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        models : Dict[str, nn.Module]
            Dict of model name -> model instance to load into
        optimizers : Dict[str, Optimizer], optional
            Dict of optimizer name -> optimizer instance to load into
        device : torch.device
            Device to load checkpoint on
        
        Returns
        -------
        Dict
            Checkpoint metadata (epoch, metrics, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load models
        for name, model in models.items():
            if f'{name}_state_dict' in checkpoint:
                model.load_state_dict(checkpoint[f'{name}_state_dict'])
                print(f"Loaded {name}")
        
        # Load optimizers
        if optimizers:
            for name, optimizer in optimizers.items():
                if f'{name}_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint[f'{name}_state_dict'])
                    print(f"Loaded {name}")
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"Loaded checkpoint from epoch {epoch}")
        if metrics:
            print(f"Metrics: {metrics}")
        
        return {'epoch': epoch, 'metrics': metrics}
    
    def find_best_checkpoint(self, metric: str = "loss") -> Optional[str]:
        """Find the best checkpoint based on a metric.
        
        Parameters
        ----------
        metric : str
            Metric to optimize (e.g., "loss" or "accuracy")
        
        Returns
        -------
        str or None
            Path to best checkpoint, or None if not found
        """
        checkpoints = list(self.checkpoint_dir.glob("best_*.pt"))
        if checkpoints:
            return str(checkpoints[0])
        return None


def evaluate_emotion_model(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate emotion recognition model.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate (encoder + classifier)
    dataloader : DataLoader
        Test dataloader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to run on
    
    Returns
    -------
    Dict[str, float]
        Metrics: loss, accuracy, per-class accuracies
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    class_correct = [0] * 5  # 5 emotion classes
    class_total = [0] * 5
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                eeg, labels = batch
            else:
                eeg = batch['eeg']
                labels = batch.get('label', torch.zeros(eeg.size(0)))
            
            eeg = eeg.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(eeg)
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item() * eeg.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += eeg.size(0)
            
            # Per-class metrics
            for c in range(5):
                class_mask = (labels == c)
                class_total[c] += class_mask.sum().item()
                class_correct[c] += ((pred == c) & (labels == c)).sum().item()
    
    metrics = {
        'loss': total_loss / max(total_samples, 1),
        'accuracy': total_correct / max(total_samples, 1),
    }
    
    # Per-class accuracies
    for c in range(5):
        if class_total[c] > 0:
            metrics[f'accuracy_class_{c}'] = class_correct[c] / class_total[c]
    
    return metrics


def print_model_info(model: torch.nn.Module):
    """Print model architecture and parameter count."""
    print("\nModel Architecture:")
    print("-" * 60)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Parameters:")
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params
