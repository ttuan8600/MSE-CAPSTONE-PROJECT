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

Processed files should go in `data/processed/` after running preprocessing scripts.
Avoid committing large data files to version control; add them to `.gitignore` if necessary.

## Current Status: Multimodal Fusion Baseline

**Baseline experiments complete with enhanced multimodal fusion:**

### What's Implemented

- **EEG Encoder**: 4-layer 1D-CNN 128-D latent
- **Audio Encoder**: 2-layer CNN processing MFCC 128-D
- **Multimodal Fusion** with **3 modes**:
  - `concat`: Concatenation + projection (baseline)
  - `cross_attention`: Cross-modal attention pooling
  - `gated`: Element-wise gating (adaptive weighting)
- **Per-channel learnable weights**: Independent scaling per modality
- **Emotion Classifier**: 3-layer head 5 class logits

### Quick Start

```bash
# Run all tests (9/9 passing)
pytest tests/test_models.py -v

# Test fusion modes on synthetic data
python scripts/baseline_experiment.py --use-audio --fusion-mode concat
python scripts/baseline_experiment.py --use-audio --fusion-mode cross_attention

# Train on real data with fusion mode selection
python scripts/train.py --use-audio --fusion-mode gated --num-epochs 5
```

### Documentation

Detailed baseline, fusion, and training notes are kept locally in
`docs/archive/` and are not published to this repository:

- `FUSION_IMPROVEMENTS_SUMMARY.md` Detailed fusion modes and performance
- `README_BASELINE.md` Baseline experiments with CLI options
- `TRAINING_GUIDE.md` Complete training pipeline documentation
- `BASELINE_FLOWCHART.md` Architecture and dataflow diagrams

---

## 🎯 Results

**[CHANGELOG.md](docs/CHANGELOG.md) is the single source of truth** for model status and
accuracy. Every number there is traced to an artifact under `outputs/`.

### Model of record

| Metric                     | Value                                        |
| -------------------------- | -------------------------------------------- |
| **Architecture**           | Cross-Modal Attention Fusion (4-head), ~920K params |
| **Held-out test accuracy** | **78.57%**                                   |
| **Validation accuracy**    | 75.87% (best epoch 36/40)                    |
| **Checkpoint**             | `outputs/attention_fusion_model_best.pt`     |
| **Artifact**               | `outputs/attention_fusion_20260401_182606/results.json` |

Clean progression: gated fusion 52.22% → focal-loss CNN 63.02% → **attention fusion 78.57%**.

> **⚠️ Do not cite 82.06%, 84.44%, or "+3.49pp".** Those figures appear throughout
> the archived documentation but are not supported by any artifact. `82.06` is a
> hardcoded literal (`scripts/deploy_finetuned_model.py:69`); the 84.xx% figures come
> from an evaluation whose "test set" was 69% training data, caused by the training
> and evaluation scripts seeding different random number generators. The only real
> fine-tuning comparison recorded **-0.48pp**. See
> [CHANGELOG.md](docs/CHANGELOG.md#known-measurement-issue-traintest-contamination).

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

# Inference (78.57% held-out test accuracy)
eeg_features = encoder(eeg_data)
audio_features = audio_encoder(audio_data)
fused = attention_fusion(eeg_features, audio_features)
predictions = classifier(fused)
```

### Documentation

- [CHANGELOG.md](docs/CHANGELOG.md) - **Results of record, experiment history, and open gaps**
- [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) - How to use and deploy the model
- [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) - REST API reference

Superseded status reports are retained in [docs/archive/](docs/archive/) for
provenance. They contain the disputed figures noted above and should not be cited.

---
