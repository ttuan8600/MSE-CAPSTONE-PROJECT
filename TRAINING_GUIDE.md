# EmoAI Training Guide: Pre-train on FACED, Fine-tune on EAV

_Includes multimodal EEG+audio fusion support (use `--use-audio` flag)_

## Overview

This guide walks you through the two-stage training pipeline for emotion recognition using synchronized multimodal data:

1. **Pre-training Stage**: Train EEG encoder on FACED dataset (123 subjects, massive unlabeled EEG pool)
2. **Fine-tuning Stage**: Adapt encoder to EAV multimodal data (40 subjects with synchronized audio/EEG/video)

## Dataset Structure

### FACED Dataset

- **Location**: `data/raw/Processed_data/Processed_data/`
- **Format**: 123 pickle files (`sub000.pkl` to `sub122.pkl`)
- **Data per subject**: (28 channels, 32 emotion trials, 7500 time steps)
- **Total subjects**: 123
- **Purpose**: Large-scale pre-training to learn robust EEG representations

### EAV Dataset

- **Location**: `data/raw/EAV/EAV/subject*/`
- **Modalities**: Audio (`.wav`), EEG (`.mat`), Video (`.mp4`) - synchronized
- **Total subjects**: 42 (subject1 through subject42)
- **Total trials per subject**: ~200 trials (mixed Listening & Speaking)
- **Emotion classes**: Neutral, Anger, Calmness, Sadness, Happiness (5 emotions)
- **EEG format**: MATLAB `.mat` files with 28 channels + label files
- **Audio format**: WAV files with emotion label in filename
- **Video format**: MP4 files with temporal synchronization
- **Purpose**: Fine-tune pre-trained encoder on multimodal emotion data

## Model Architecture

### EEGEncoder (CNN-based)

- **Input**: (batch_size, 28 channels, variable time steps)
- **Architecture**: 4 temporal convolution blocks with batch norm
- **Output**: (batch_size, latent_dim=128)
- **Use case**: Fast inference, good for real-time applications

### EEGEncoderLSTM (CNN-LSTM hybrid)

- **Input**: (batch_size, 28 channels, variable time steps)
- **Architecture**: 2 CNN blocks + 2-layer bidirectional LSTM
- **Output**: (batch_size, latent_dim=128)
- **Use case**: Better temporal modeling, recommended for high accuracy

## Quick Start

### 1. Pre-training on FACED (50 epochs recommended)

```bash
python scripts/train.py --mode pretrain \
    --num-epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --window-size 512
```

**Expected output:**

- Checkpoints saved to: `outputs/pretraining_YYYYMMDD_HHMMSS/`
- Best model: `best_model.pt`
- TensorBoard logs for monitoring training

**Typical training time**: 2-4 hours on GPU (RTX 3090), scales with your hardware

### 2. Fine-tuning on EAV (30 epochs recommended)

The fine-tuning script now supports optional audio modality fusion. By default
only EEG features are used; add `--use-audio` to include MFCC-based audio
features in the training pipeline.

Once pre-training completes, run:

```bash
python scripts/train.py --mode finetune \
    --num-epochs 30 \
    --batch-size 16 \
    --finetune-lr 1e-4 \
    --pretrained-path outputs/pretraining_YYYYMMDD_HHMMSS/best_model.pt \
    [--use-audio]           # include audio modality (MFCC features)
```

**Expected output:**

- Fine-tuned models saved to: `outputs/finetuning_YYYYMMDD_HHMMSS/`
- EAV-specific emotion classifier checkpoint

## Training Details

### Pre-training Workflow

1. Load all 123 subjects from FACED
2. Segment each subject's EEG into overlapping 512-sample windows
3. Normalize each window per-channel (z-score)
4. Train EEG encoder + emotion classifier on pseudo emotion labels
5. Save best model based on validation loss

### Fine-tuning Workflow

1. Load pre-trained EEG encoder
2. Load EAV dataset (40 subjects with real emotion labels)
3. Train only the emotion classifier head while fine-tuning encoder features
4. Use lower learning rate (1e-4) to preserve pre-trained features
5. Save checkpoint every 2 epochs

## Monitoring Training

### Using TensorBoard

```bash
tensorboard --logdir outputs/pretraining_YYYYMMDD_HHMMSS
```

Then open http://localhost:6006 in your browser to monitor:

- Training loss and accuracy
- Learning rate schedule
- Gradient norms

## Architecture Details

### EEGEncoder Block Diagram

```
Input (28, T)
    ↓
Conv1d + BatchNorm (→ 64 channels, T/2)
    ↓
Conv1d + BatchNorm (→ 128 channels, T/4)
    ↓
Conv1d + BatchNorm (→ 256 channels, T/8)
    ↓
Conv1d + BatchNorm (→ 256 channels, T/16)
    ↓
Adaptive Avg Pool (→ 256)
    ↓
Linear (→ 128) [Latent representation]
```

### Emotion Classifier Head

```
Latent (128)
    ↓
FC + ReLU + Dropout (→ 256)
    ↓
FC + ReLU + Dropout (→ 128)
    ↓
FC (→ 5 emotions) [Logits]
```

## Multi-modal Integration (Future Work)

After encoder pre-training/fine-tuning, integrate:

- **Speech features**: MFCC spectrograms + GAN augmentation
- **Video features**: Extracted landmarks/face embeddings
- **Cross-Modal Attention**: Synchronize temporal alignment
- **Fusion**: Concat or cross-attention pooling

## Hyperparameter Reference

| Parameter     | Pre-training    | Fine-tuning                   | Notes                                     |
| ------------- | --------------- | ----------------------------- | ----------------------------------------- |
| Batch size    | 32              | 16                            | Smaller for fine-tuning (more stable)     |
| Learning rate | 1e-3            | 1e-4                          | 10x lower for fine-tuning                 |
| Window size   | 512             | 512                           | ~260ms at typical EEG sampling rate       |
| Num epochs    | 50              | 30                            | Longer pre-training needed for scale      |
| Optimizer     | Adam            | Adam                          | Both use default β₁=0.9, β₂=0.999         |
| LR schedule   | CosineAnnealing | StepLR (γ=0.5 every 5 epochs) | More aggressive decay for fine-tune       |
| Weight decay  | 0               | 0                             | Consider L2 regularization if overfitting |

## Troubleshooting

**Q: GPU memory errors?**

- Reduce batch size: `--batch-size 16` or `8`
- Reduce window size: `--window-size 256`

**Q: Pre-training is slow?**

- Use Windows GPU driver (CUDA 12+)
- Check GPU is actually being used: `nvidia-smi` during training
- Reduce num_workers if CPU bottleneck: `DataLoader(..., num_workers=0)`

**Q: Fine-tuning loss not decreasing?**

- Learning rate might be too high, try: `--finetune-lr 5e-5`
- Verify pre-trained model path is correct
- Check that EAV dataset is properly formatted

## Next Steps

1. ✅ Pre-train EEG encoder on FACED
2. ✅ Fine-tune on EAV EEG data
3. 🔄 Implement multi-modal fusion (audio + video)
4. 🔄 Add GAN-based data augmentation
5. 🔄 Cross-modal attention mechanism
6. 🔄 End-to-end training on all modalities

## Data Loading Implementation

### Multi-modal Training

The training pipeline can now exploit both EEG and audio modalities. When
`--use-audio` is specified, the loader returns MFCC features for each
sample and the model uses an `AudioEncoder` to convert them into a 128‑D
latent representation. A `MultimodalFusion` module concatenates EEG and
audio latents before classification. Video is still a placeholder.

### EAV Dataset Loader

The `EAVMultimodalDataset` class in `src/preprocessing/data_loader.py` provides:

1. **EEG Loading** from `.mat` files:
   - Uses `scipy.io.loadmat()` to read MATLAB format
   - Extracts 28-channel EEG data per subject
   - Matches data with associated label files
   - Applies per-channel z-score normalization

2. **Audio Feature Extraction** from `.wav` files:
   - Computes MFCC features using `torchaudio.transforms.MFCC`
   - Default: 13 coefficients (n_mfcc=13)
   - Parses emotion label from filename (e.g., `002_Trial_02_Speaking_**Neutral**_Aud.wav`)
   - Output shape: (n_mfcc, time_steps)

3. **Video Loading** (`.mp4` files):
   - Stub implementation (to be extended)
   - Placeholder for future face landmark/embedding extraction

### Usage Example

```python
from src.preprocessing.data_loader import create_eav_dataloader

# Create dataloader
train_loader, train_dataset = create_eav_dataloader(
    eav_data_dir="data/raw/EAV/EAV",
    batch_size=16,
    shuffle=True,
    load_audio=True,
    load_video=False
)

# Iterate samples
for batch in train_loader:
    eeg = batch['eeg']              # (batch_size, 28, time_steps)
    audio = batch['audio']          # (batch_size, 13, time_steps)
    emotion = batch['emotion']      # (batch_size,) in [0-4]
    subject_id = batch['subject_id']
```

## File Structure

```
scripts/train.py                    # Main training script
src/models/eeg_encoder.py          # EEG encoder models
src/preprocessing/
  data_loader.py                   # FACED & EAV dataloaders
  eeg.py                           # EEG I/O utilities (MATLAB/NPY)
  speech.py                        # Audio preprocessing stubs
outputs/
  pretraining_YYYYMMDD_HHMMSS/
    best_model.pt                   # Use this for fine-tuning
    checkpoint_epoch_*.pt
    events.out.tfevents             # TensorBoard logs
  finetuning_YYYYMMDD_HHMMSS/
    finetuned_epoch_*.pt            # Final models
    events.out.tfevents
```

## References

- **FACED Dataset**: Large-scale EEG emotions dataset (~123 subjects)
- **EAV Dataset**: Emotion Audio-Visual with synchronized modalities (40 subjects)
- **CNN-LSTM Architecture**: Combines temporal convolution with recurrent modeling
- **Transfer Learning**: Pre-train on large dataset, fine-tune on target task
