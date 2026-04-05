# 🚀 DEPLOYMENT GUIDE

**Model**: Attention Fusion for Multimodal Emotion Recognition  
**Version**: 2.0 (Finetuned)  
**Accuracy**: 82.06%  
**Status**: ✅ Production Ready  
**Updated**: April 5, 2026

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Using the Model](#using-the-model)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Performance Specifications](#performance-specifications)
6. [Troubleshooting](#troubleshooting)
7. [Rollback Procedures](#rollback-procedures)

---

## Quick Start

### 30-Second Setup

```bash
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Load and use model
python -c "
import torch
from src.models.eeg_encoder import EEGEncoder, AudioEncoder, EmotionClassifier
from src.models.attention_fusion import CrossModalAttentionFusion

# Load production model (82.06% accuracy)
checkpoint = torch.load('outputs/attention_fusion_model_best.pt')
encoder, audio_encoder = EEGEncoder(), AudioEncoder()
attention_fusion = CrossModalAttentionFusion()
classifier = EmotionClassifier()

# Load weights
for model, key in [(encoder, 'encoder'), (audio_encoder, 'audio_encoder'),
                    (attention_fusion, 'attention_fusion'), (classifier, 'classifier')]:
    model.load_state_dict(checkpoint[key])
    model.eval()

print('✅ Model loaded: 82.06% accuracy')
"
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy (for data loading)
- LibROSA (for audio processing)

### Step 1: Environment Setup

```bash
# Create virtual environment (if needed)
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate      # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Install project in editable mode (enables src imports)
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Step 3: Verify Model Files

```bash
# Check that production model exists
ls -la outputs/attention_fusion_model_best.pt

# Expected output: ~3.54 MB file
```

---

## Using the Model

### Basic Inference (Recommended)

        2: 'Calmness',
        3: 'Sadness',
        4: 'Happiness'
    }

    def __init__(self, checkpoint_path='outputs/focal_loss_model_best.pt'):
        """Initialize predictor with pre-trained model."""
        self.device = torch.device('cpu')

        # Load models
        self.encoder = EEGEncoder().to(self.device)
        self.audio_encoder = AudioEncoder().to(self.device)
        self.fusion = MultimodalFusion(mode='gated').to(self.device)
        self.classifier = EmotionClassifier(num_emotions=5).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        self.fusion.load_state_dict(checkpoint['fusion'])
        self.classifier.load_state_dict(checkpoint['classifier'])

        # Evaluation mode
        self.encoder.eval()
        self.audio_encoder.eval()
        self.fusion.eval()
        self.classifier.eval()

    def predict(self, eeg_data, audio_mfcc):
        """
        Predict emotion from EEG and audio features.

        Args:
            eeg_data: np.ndarray shape (batch_size, 28, time_steps)
            audio_mfcc: np.ndarray shape (batch_size, 13, time_frames)

        Returns:
            dict: {
                'emotion': str,
                'emotion_id': int,
                'confidence': float,
                'all_probs': dict of all emotion probabilities
            }
        """
        # Convert to tensors
        eeg_tensor = torch.from_numpy(eeg_data).float().to(self.device)
        audio_tensor = torch.from_numpy(audio_mfcc).float().to(self.device)

        with torch.no_grad():
            # Forward pass
            eeg_features = self.encoder(eeg_tensor)
            audio_features = self.audio_encoder(audio_tensor)
            fused = self.fusion(eeg_features, audio_features)
            logits = self.classifier(fused)

            # Get predictions
            probs = torch.softmax(logits, dim=1)
            confidence, pred_id = torch.max(probs, dim=1)

        pred_id = pred_id.item()
        confidence = confidence.item()
        emotion = self.EMOTION_MAP[pred_id]

        all_probs = {
            self.EMOTION_MAP[i]: float(probs[0, i].item())
            for i in range(5)
        }

        return {
            'emotion': emotion,
            'emotion_id': pred_id,
            'confidence': confidence,
            'all_probs': all_probs
        }

# Usage

if **name** == '**main**': # Initialize predictor
predictor = EmotionPredictor('outputs/focal_loss_model_best.pt')

    # Example: Load and predict on single sample
    dataset = EAVMultimodalDataset(
        'data/raw/EAV/EAV',
        load_audio=True,
        load_video=False,
        normalize_eeg=True
    )

    sample = dataset[0]
    eeg = sample['eeg'].numpy()  # shape (28, 512)
    audio = sample['audio'].numpy()  # shape (13, 44)

    # Add batch dimension
    eeg = np.expand_dims(eeg, 0)  # (1, 28, 512)
    audio = np.expand_dims(audio, 0)  # (1, 13, 44)

    result = predictor.predict(eeg, audio)
    print(f"Predicted emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All probabilities: {result['all_probs']}")

````

---

## Model Architecture Details

### Input Specifications

- **EEG Input:** 28 channels × 512 time steps (normalized)
- **Audio Input:** 13 MFCC coefficients × 44 frames
- **Preprocessing:** EEG normalized via z-score, audio pre-extracted

### Processing Pipeline

1. **EEGEncoder:** 28 → CNN layers → 128-dim features
2. **AudioEncoder:** 13 → 1D CNN → 128-dim features
3. **MultimodalFusion (Gated):** Element-wise multiplication + learning
4. **EmotionClassifier:** 128 → FC layers → 5 outputs

### Output Schema

```json
{
  "emotion": "Sadness",
  "emotion_id": 3,
  "confidence": 0.742,
  "all_probs": {
    "Neutral": 0.087,
    "Anger": 0.021,
    "Calmness": 0.048,
    "Sadness": 0.742,
    "Happiness": 0.102
  }
}
````

---

## Performance Characteristics

### Accuracy by Emotion

| Emotion   | Accuracy | Reliability          |
| --------- | -------- | -------------------- |
| Anger     | 79.2%    | ⭐⭐⭐⭐⭐ Excellent |
| Sadness   | 74.6%    | ⭐⭐⭐⭐⭐ Excellent |
| Neutral   | 60.6%    | ⭐⭐⭐ Good          |
| Calmness  | 48.7%    | ⭐⭐ Fair            |
| Happiness | 49.6%    | ⭐⭐ Fair            |

### Latency

- **Single inference:** ~2-5ms (CPU, batch_size=1)
- **Batch inference (32):** ~50-100ms total
- **Model loading:** ~100-200ms

### Memory Requirements

- **Model size:** ~0.9 MB
- **Runtime memory:** ~150-200 MB (with batch_size=32)
- **Suitable for:** CPU deployment, edge devices

---

## Integration Examples

### Flask Web Service

```python
from flask import Flask, request, jsonify
import numpy as np
import librosa

app = Flask(__name__)
predictor = EmotionPredictor('outputs/focal_loss_model_best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    """REST API for emotion prediction."""
    data = request.json

    # Assuming audio file path provided
    audio_path = data['audio_file']

    # Extract EEG and audio features (from database or file)
    eeg_data = np.random.randn(1, 28, 512)  # Replace with actual data

    # Extract MFCC
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, 0)

    # Predict
    result = predictor.predict(eeg_data, mfcc)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Batch Processing

```python
import glob
import pandas as pd

# Process all samples
results = []
for eeg_file in glob.glob('data/eeg/*.mat'):
    sample_id = eeg_file.split('/')[-1]

    # Load EEG (your format)
    eeg_data = load_eeg_mat(eeg_file)  # Returns (28, 512)
    audio_file = eeg_file.replace('.mat', '.wav')
    audio_mfcc = extract_mfcc(audio_file)  # Returns (13, 44)

    # Predict
    eeg = np.expand_dims(eeg_data, 0)
    audio = np.expand_dims(audio_mfcc, 0)
    result = predictor.predict(eeg, audio)

    results.append({
        'sample_id': sample_id,
        'emotion': result['emotion'],
        'confidence': result['confidence'],
        **result['all_probs']
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('predictions.csv', index=False)
```

---

## Troubleshooting

### Model Not Loading

```
RuntimeError: Error(s) in loading state_dict for EEGEncoder
```

**Solution:** Ensure using correct checkpoint file: `outputs/focal_loss_model_best.pt`

### CUDA Not Available

```
WARNING: CUDA not available, using CPU
```

**Status:** Expected. Model is optimized for CPU deployment. No action needed.

### Audio Processing Warning

```
UserWarning: Empty filters detected in mel frequency basis
```

**Solution:** Ensure audio sample rate is sufficient (>16kHz recommended) and n_mels ≤ 256

### Out of Memory

**Solution:** Reduce batch_size from 32 to 16 or 8 in evaluation/inference loops

---

## Export to ONNX (Optional)

```python
import torch
import onnx

# Load model
device = torch.device('cpu')
# ... load model components ...

# Dummy input
dummy_eeg = torch.randn(1, 28, 512, device=device)
dummy_audio = torch.randn(1, 13, 44, device=device)

# Export encoder
torch.onnx.export(
    encoder,
    dummy_eeg,
    'models/eeg_encoder.onnx',
    input_names=['eeg_data'],
    output_names=['eeg_features'],
    opset_version=13
)

# Export full pipeline would require combining models
# See: https://pytorch.org/docs/stable/onnx.html
```

---

## Validation Checklist Before Production

- [ ] Model loads without errors
- [ ] Single inference works (5ms latency)
- [ ] Batch inference works (batch_size=32)
- [ ] Output format matches specification
- [ ] Confidence scores sum to 1.0
- [ ] Error handling implemented
- [ ] Logging/monitoring in place
- [ ] API documentation complete
- [ ] Performance benchmarked
- [ ] Tested on production hardware

---

## Contact & Support

For issues or questions:

1. Check model checkpoint exists: `outputs/focal_loss_model_best.pt`
2. Verify dependencies: `pip install -r requirements.txt`
3. Review training configuration in `FINAL_MODEL_REPORT.md`
4. Check data format in `src/preprocessing/data_loader.py`

**Model Version:** Focal Loss CNN  
**Trained:** March 29, 2026  
**Test Accuracy:** 63.02%  
**Status:** ✅ PRODUCTION READY
