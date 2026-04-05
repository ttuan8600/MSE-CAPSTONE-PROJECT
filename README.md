# MSE-CAPSTONE-PROJECT

This is the capstone project for the FPT Master of Software Engineering.

## 🚀 Quick Start

1. **Create & activate** a virtual environment (already present `.venv`):
   ```powershell
   & ".\.venv\Scripts\Activate.ps1"
   ```
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Install package in editable mode** (enables imports from `src`):
   ```powershell
   pip install -e .
   ```
4. **Run tests**:
   ```powershell
   pytest
   ```
5. **Launch the pipeline stub**:
   ```powershell
   python scripts\run_pipeline.py
   ```

Feel free to explore `src/preprocessing`, `src/gan`, and `src/models` for existing placeholders.

## 📁 Data

This project supports two main datasets:

- **FACED (Finer-grained Affective Computing EEG Dataset)**: pre-processed into `.pkl` files and stored under `data/raw/Processed_data/Processed_data/` (e.g., `sub000.pkl`, `sub001.pkl`, ...). This is used for pre-training the EEG encoder.
- **EAV Multimodal Dataset**: raw multimodal data under `data/raw/EAV/EAV/` (EEG, audio, video). Used for fine-tuning.

## Processed files should go in `data/processed/` after running preprocessing scripts. Avoid committing large data files to version control; add them to `.gitignore` if necessary.

## Current Status: Multimodal Fusion Baseline

**Baseline experiments complete with enhanced multimodal fusion:**

### What's Implemented

- **EEG Encoder**: 4-layer 1D-CNN 128-D latent
- **Audio Encoder**: 2-layer CNN processing MFCC 128-D
- **Multimodal Fusion** with **3 modes**:
  - \concat\: Concatenation + projection (baseline)
  - \cross_attention\: Cross-modal attention pooling
  - \gated\: Element-wise gating (adaptive weighting)
- **Per-channel learnable weights**: Independent scaling per modality
- **Emotion Classifier**: 3-layer head 5 class logits

### Quick Start

\\\ash

# Run all tests (9/9 passing)

pytest tests/test_models.py -v

# Test fusion modes on synthetic data

python scripts/baseline_experiment.py --use-audio --fusion-mode concat
python scripts/baseline_experiment.py --use-audio --fusion-mode cross_attention

# Train on real data with fusion mode selection

python scripts/train.py --use-audio --fusion-mode gated --num-epochs 5
\\\

### Documentation

- [FUSION_IMPROVEMENTS_SUMMARY.md](FUSION_IMPROVEMENTS_SUMMARY.md) Detailed fusion modes and performance
- [README_BASELINE.md](README_BASELINE.md) Baseline experiments with CLI options
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) Complete training pipeline documentation
- [BASELINE_FLOWCHART.md](BASELINE_FLOWCHART.md) Architecture and dataflow diagrams

---

## 🎯 Latest Results: Production Model (April 5, 2026)

### Current Production Accuracy

| Metric                        | Value       | Status     |
| ----------------------------- | ----------- | ---------- |
| **Production Model Accuracy** | **82.06%**  | ✅ Live    |
| **Previous Accuracy**         | 78.57%      | Archived   |
| **Improvement**               | **+3.49pp** | ⬆️ Upgrade |

### Model Details

- **Architecture**: Cross-Modal Attention Fusion (4-head multi-head attention)
- **Parameters**: ~920K total
- **Deployment Date**: April 5, 2026
- **Checkpoint**: `outputs/attention_fusion_model_best.pt`
- **Backup**: `outputs/attention_fusion_model_baseline_backup_20260405.pt`

### Recent Improvements

- ✅ **Fine-tuning with Data Augmentation**: +3.49pp improvement
  - SpecAugment: Time/frequency masking on audio
  - EEG Jitter: Gaussian noise (σ=0.01)
  - Lower learning rate (1e-4) for careful parameter updates
  - Results in [FINETUNING_RESULTS_SUMMARY.md](FINETUNING_RESULTS_SUMMARY.md)

### Training Configuration

```python
# Fine-tuning Hyperparameters
optimizer: Adam
learning_rate: 1e-4
loss_function: Focal Loss (γ=2.0, α-weighted)
data_augmentation: SpecAugment + EEG Jitter
early_stopping: patience=5 epochs
convergence: Epoch 11/20 with ~82% accuracy
```

### Deployment & Usage

To use the current production model:

```python
import torch
from src.models.eeg_encoder import EEGEncoder, AudioEncoder, EmotionClassifier
from src.models.attention_fusion import CrossModalAttentionFusion

# Load model
checkpoint = torch.load('outputs/attention_fusion_model_best.pt', map_location='cpu')
encoder = EEGEncoder()
audio_encoder = AudioEncoder()
attention_fusion = CrossModalAttentionFusion()
classifier = EmotionClassifier()

# Load weights
encoder.load_state_dict(checkpoint['encoder'])
audio_encoder.load_state_dict(checkpoint['audio_encoder'])
attention_fusion.load_state_dict(checkpoint['attention_fusion'])
classifier.load_state_dict(checkpoint['classifier'])

# Inference (82.06% accuracy)
eeg_features = encoder(eeg_data)
audio_features = audio_encoder(audio_data)
fused = attention_fusion(eeg_features, audio_features)
predictions = classifier(fused)
```

### Documentation for Latest Release

- [DEPLOYMENT_CHANGELOG.md](DEPLOYMENT_CHANGELOG.md) - Detailed deployment changes & rollback instructions
- [MODEL_PERFORMANCE_COMPARISON.md](MODEL_PERFORMANCE_COMPARISON.md) - Baseline vs Finetuned comparison
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - How to use and deploy the model
- [FINETUNING_RESULTS_SUMMARY.md](FINETUNING_RESULTS_SUMMARY.md) - Fine-tuning training results

---
