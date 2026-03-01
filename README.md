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

Place raw datasets under `data/raw/` (e.g., IEMOCAP, DEAP archives). Processed files should go in `data/processed/` after running preprocessing scripts. Avoid committing large data files to version control; add them to `.gitignore` if necessary.
---

##  Current Status: Multimodal Fusion Baseline 

**Baseline experiments complete with enhanced multimodal fusion:**

### What's Implemented

- **EEG Encoder**: 4-layer 1D-CNN  128-D latent
- **Audio Encoder**: 2-layer CNN processing MFCC  128-D  
- **Multimodal Fusion** with **3 modes**:
  - \concat\: Concatenation + projection (baseline)
  - \cross_attention\: Cross-modal attention pooling
  - \gated\: Element-wise gating (adaptive weighting)
- **Per-channel learnable weights**: Independent scaling per modality
- **Emotion Classifier**: 3-layer head  5 class logits

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

- [FUSION_IMPROVEMENTS_SUMMARY.md](FUSION_IMPROVEMENTS_SUMMARY.md)  Detailed fusion modes and performance
- [README_BASELINE.md](README_BASELINE.md)  Baseline experiments with CLI options  
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md)  Complete training pipeline documentation
- [BASELINE_FLOWCHART.md](BASELINE_FLOWCHART.md)  Architecture and dataflow diagrams

---
