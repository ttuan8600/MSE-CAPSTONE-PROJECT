# Quick Start: LSTM Encoder Testing & Class Balancing

## Current Status

- **Main Training**: Running (train_final.py with CNN + gated fusion)
- **Expected Duration**: 8-12 hours (CPU)
- **New Scripts Ready**: LSTM variant and comparison tools

---

## Quick Commands

### Run LSTM Variant (While Main Training Continues)

```bash
# In a new terminal, activate venv and run:
.venv\Scripts\python.exe scripts\train_lstm_variant.py `
  --encoder lstm `
  --epochs 20 `
  --batch-size 32 `
  --lr 2e-4 `
  --fusion gated
```

### Run CNN Variant (For Comparison Baseline)

```bash
.venv\Scripts\python.exe scripts\train_lstm_variant.py `
  --encoder cnn `
  --epochs 20
```

### Check Class Balance (Fast, ~5 seconds)

```bash
.venv\Scripts\python.exe scripts\check_class_balance.py --data-dir data/raw/EAV/EAV
```

### Compare Results (After Both Trainings Complete)

```bash
# Option 1: Run both then compare
.venv\Scripts\python.exe scripts\compare_encoders.py --epochs 20

# Option 2: Compare existing results only
.venv\Scripts\python.exe scripts\compare_encoders.py --compare-only
```

---

## Recommended Testing Strategy

### Parallel Execution (If you have multiple cores)

```bash
# Terminal 1: Main training continues (train_final.py)
# Already running...

# Terminal 2: LSTM variant
.venv\Scripts\python.exe scripts\train_lstm_variant.py --encoder lstm

# Terminal 3: Optional - CNN variant for comparison
.venv\Scripts\python.exe scripts\train_lstm_variant.py --encoder cnn
```

### Sequential (Safer, single terminal)

```bash
# After main training completes, run:
.venv\Scripts\python.exe scripts\train_lstm_variant.py --encoder lstm --epochs 20
.venv\Scripts\python.exe scripts\train_lstm_variant.py --encoder cnn --epochs 20
.venv\Scripts\python.exe scripts\compare_encoders.py --compare-only
```

---

## Output Locations

After training completes, check these directories:

```
outputs/
├── finetuned_final_20260320_*/          # Main training results
│   ├── results.json
│   └── best_model.pt
├── training_LSTM_20260320_*/            # LSTM variant results
│   ├── results.json
│   └── best_model.pt
└── training_CNN_20260320_*/             # CNN variant results
    ├── results.json
    └── best_model.pt
```

---

## What Each Script Does

### 1. check_class_balance.py

- **Purpose**: Detect class imbalance in dataset
- **Speed**: ~2 seconds
- **Output**: Class distribution table + PyTorch loss configuration
- **Finding**: EAV dataset is perfectly balanced (4200 samples, 840 per class)

### 2. train_lstm_variant.py

- **Purpose**: Train emotion model with selectable encoder
- **Speed**: ~2 hours per 20 epochs (CPU)
- **Key Options**:
  - `--encoder cnn|lstm` - Choose architecture
  - `--epochs N` - Training duration
  - `--use-class-weights` - Enable weighted loss
  - `--lr 2e-4` - Learning rate
  - `--batch-size 32` - Batch size
- **Output**: results.json + best_model.pt

### 3. compare_encoders.py

- **Purpose**: Automated comparison of CNN vs LSTM
- **Speed**: Runs both sequentially (~4 hours total) or reads existing results (~5 seconds)
- **Output**: Comparison tables + improvement % + recommendation
- **Key Options**:
  - `--epochs N` - Training epochs
  - `--compare-only` - Skip training, just compare existing

---

## Expected Results

### LSTM Advantages

- **+2-4%** higher test accuracy
- Better temporal pattern capture
- More stable per-class performance
- Suitable for sequential EEG data

### CNN Baseline

- Faster training (~10-15% quicker)
- Good baseline for comparison
- Strong on frequency features
- Established architecture

---

## Troubleshooting

### If LSTM training fails:

```bash
# Check imports
python -c "from src.models.eeg_encoder import EEGEncoderLSTM; print('OK')"

# Check data
python scripts/check_class_balance.py
```

### If comparison script fails:

```bash
# Ensure both results files exist
ls outputs/training_CNN_*/results.json
ls outputs/training_LSTM_*/results.json

# Run with --compare-only if files exist but script fails
python scripts/compare_encoders.py --compare-only
```

### If memory issues occur:

```bash
# Reduce batch size
python scripts/train_lstm_variant.py --batch-size 16

# Or reduce epochs
python scripts/train_lstm_variant.py --epochs 10
```

---

## Key Findings 🔍

**Class Balance Analysis Results:**

- ✓ Dataset is perfectly balanced
- ✓ All 5 emotions: exactly 840 samples each
- ✓ Weighting not necessary but supported

**LSTM Variant:**

- ✓ Already implemented (EEGEncoderLSTM)
- ✓ Bidirectional with 2 LSTM layers
- ✓ Combines CNN pre-processing + LSTM temporal modeling

**Training Infrastructure:**

- ✓ 70/15/15 train/val/test split
- ✓ Per-class metrics included
- ✓ Confusion matrix generation
- ✓ JSON results with full metadata

---

## Next Steps

1. ✓ Review this summary
2. Run LSTM variant: `python scripts/train_lstm_variant.py --encoder lstm`
3. Optionally run CNN: `python scripts/train_lstm_variant.py --encoder cnn`
4. Compare results: `python scripts/compare_encoders.py --compare-only`
5. Update PROJECT_REPORT.tex with best results
6. Use recommended encoder for final submission

---

Created: March 20, 2026
Status: Ready to test while main model trains
