# Quick Start: Emotion Recognition Training

Your EmoAI training pipeline is **ready to go**!

## One-Command Start

### Pre-train EEG Encoder on FACED (123 subjects)

```bash
python scripts/train.py --mode pretrain --num-epochs 50 --batch-size 32
```

This will:

- Load 123 subjects from FACED dataset
- Segment EEG into 512-sample windows
- Train EEGEncoder + EmotionClassifier
- Save best model to `outputs/pretraining_YYYYMMDD_HHMMSS/best_model.pt`
- Log metrics to TensorBoard

**Expected time**: 2-4 hours on GPU (RTX 3090)

---

### Fine-tune on EAV (40 subjects with audio/EEG/video)

```bash
python scripts/train.py --mode finetune --num-epochs 30 --batch-size 16 \
    --pretrained-path outputs/pretraining_YYYYMMDD_HHMMSS/best_model.pt
```

This will:

- Load pre-trained EEG encoder from FACED
- Adapt to EAV dataset (synchronized audio/EEG/video)
- Fine-tune with lower learning rate (1e-4)
- Save checkpoints to `outputs/finetuning_YYYYMMDD_HHMMSS/`

**Expected time**: 1-2 hours on GPU

---

## Monitor Training

While training runs, monitor progress with TensorBoard:

```bash
tensorboard --logdir outputs/pretraining_YYYYMMDD_HHMMSS
```

Then open http://localhost:6006 in your browser to see:

- Training loss & accuracy curves
- Learning rate schedule
- Gradient norms

---

## What Just Got Implemented

### ✅ Models (src/models/eeg_encoder.py)

- **EEGEncoder**: Fast CNN-based encoder (~1.8M params)
- **EEGEncoderLSTM**: Enhanced CNN-LSTM variant (~2.1M params)
- **EmotionClassifier**: 5-emotion classification head

### ✅ Data Loading (src/preprocessing/data_loader.py)

- **FAEDDataset**: Loads and segments FACED (123 subjects)
- **EAVMultimodalDataset**: Loads synchronized EAV data
- Auto-windowing, normalization, batching

### ✅ Training Scripts (scripts/train.py)

- **PretrainingTrainer**: FACED pre-training with checkpointing
- **FineTuningTrainer**: EAV fine-tuning with pretrained weights

### ✅ Utilities (src/utils/training.py)

- **CheckpointManager**: Save/load models and optimizers
- **evaluate_emotion_model()**: Per-class emotion metrics
- **print_model_info()**: Architecture and parameter summary

### ✅ Documentation

- **TRAINING_GUIDE.md**: Detailed workflow, hyperparameters, troubleshooting
- **IMPLEMENTATION_SUMMARY.md**: What was built and why

---

## Architecture Overview

```
FACED Pre-training (123 subjects)
        ↓
   EEGEncoder
      +
   EmotionClassifier
        ↓
   Best weights saved
        ↓
EAV Fine-tuning (40 subjects)
        ↓
   Load pre-trained encoder
      +
   New emotion classifier
        ↓
   Fine-tune on multimodal data
        ↓
   Production model
```

---

## Dataset Info

| Dataset   | Size         | Format                               | Purpose      |
| --------- | ------------ | ------------------------------------ | ------------ |
| **FACED** | 123 subjects | Pickles (28ch, 32trials, 7500 steps) | Pre-training |
| **EAV**   | 40+ subjects | Audio/EEG/Video folders              | Fine-tuning  |

---

## Key Features

✅ **Transfer Learning**: Pre-train on massive FACED, fine-tune on EAV
✅ **Flexible Architecture**: Choose CNN or CNN-LSTM encoder
✅ **Multimodal Ready**: Data loaders support audio + EEG + video
✅ **Robust Training**: Gradient clipping, learning rate scheduling, checkpointing
✅ **TensorBoard Integration**: Real-time loss/accuracy monitoring

---

## Next Steps After Training

1. **Evaluate** the fine-tuned model on EAV test set
2. **Integrate** speech and video streams (when preprocessing complete)
3. **Implement** Cross-Modal Attention (CMA) for fusion
4. **Add** GAN-based augmentation for class balancing
5. **Deploy** for real-time emotion recognition (target <100ms latency)

---

## Tips & Troubleshooting

**Q: GPU out of memory?**

```bash
python scripts/train.py --mode pretrain --batch-size 16 --window-size 256
```

**Q: Training too slow?**

- Check GPU is being used: `nvidia-smi`
- Reduce num_workers if CPU bottleneck
- Use mixed precision training (coming soon)

**Q: Loss not decreasing?**

- Try lower learning rate: `--learning-rate 5e-4`
- Check that FACED data is loading
- Look at TensorBoard gradient plots

---

## Files Structure

```
scripts/
  train.py                    ← Main training script

src/
  models/
    eeg_encoder.py          ← EEG encoder architectures
  preprocessing/
    data_loader.py          ← FACED & EAV data loaders
  utils/
    training.py             ← Checkpoint & evaluation utils

outputs/
  pretraining_*/            ← Pre-trained model checkpoints
    best_model.pt           ← Use this for fine-tuning!
  finetuning_*/             ← Fine-tuned model checkpoints

TRAINING_GUIDE.md           ← Complete documentation
IMPLEMENTATION_SUMMARY.md   ← What was built
```

---

## Ready to Train?

```bash
# Start pre-training now!
python scripts/train.py --mode pretrain --num-epochs 50

# Or customize:
python scripts/train.py --mode pretrain \
    --num-epochs 100 \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --window-size 256
```

Good luck! 🚀
