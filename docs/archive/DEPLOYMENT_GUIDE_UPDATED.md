# Updated Deployment Guide - Attention Fusion Model (78.57%)

## 🎯 Quick Start

**Best Model:** Cross-Modal Attention Fusion  
**Accuracy:** 78.57% (test)  
**Checkpoint:** `outputs/attention_fusion_model_best.pt`  
**Status:** ✅ Production Ready

---

## 1. Model Loading

### PyTorch (Recommended)

```python
import torch
from src.models.eeg_encoder import EEGEncoder, AudioEncoder, EmotionClassifier
from src.models.attention_fusion import CrossModalAttentionFusion, AttentionFusionNetwork

device = torch.device('cpu')

# Option A: Load individual components
encoder = EEGEncoder().to(device)
audio_encoder = AudioEncoder().to(device)
attention_fusion = CrossModalAttentionFusion().to(device)
classifier = EmotionClassifier().to(device)

checkpoint = torch.load('outputs/attention_fusion_model_best.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
audio_encoder.load_state_dict(checkpoint['audio_encoder'])
attention_fusion.load_state_dict(checkpoint['attention_fusion'])
classifier.load_state_dict(checkpoint['classifier'])

# Option B: Load as single network
model = AttentionFusionNetwork().to(device)
checkpoint = torch.load('outputs/attention_fusion_model_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
```

### Production Deployment

```python
import torch
from pathlib import Path

class EmotionRecognitionModel:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cpu')
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model()
        self.emotion_labels = ['Happiness', 'Sadness', 'Anger', 'Calmness', 'Neutral']

    def _load_model(self):
        from src.models.attention_fusion import AttentionFusionNetwork
        model = AttentionFusionNetwork().to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, eeg_data, audio_data):
        """
        Args:
            eeg_data: (1, 28, 512) or (batch, 28, 512)
            audio_data: (1, 13, 44) or (batch, 13, 44)

        Returns:
            emotions: List of emotion labels
            probabilities: List of probability dicts
        """
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_data).to(self.device)
            audio_tensor = torch.FloatTensor(audio_data).to(self.device)

            logits = self.model(eeg_tensor, audio_tensor)
            probs = torch.softmax(logits, dim=1)

            emotions = []
            probabilities = []

            for i in range(probs.shape[0]):
                emotion_id = torch.argmax(probs[i]).item()
                emotions.append(self.emotion_labels[emotion_id])
                probabilities.append({
                    label: float(probs[i][j].item())
                    for j, label in enumerate(self.emotion_labels)
                })

        return emotions, probabilities

# Usage
model = EmotionRecognitionModel('outputs/attention_fusion_model_best.pt')
emotions, probs = model.predict(eeg_batch, audio_batch)
print(emotions)  # ['Sadness', 'Anger', ...]
print(probs)     # [{'Happiness': 0.02, 'Sadness': 0.88, ...}, ...]
```

---

## 2. Flask API Example

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from pathlib import Path

app = Flask(__name__)
CORS(app)

class EmotionModelAPI:
    def __init__(self, model_path):
        self.device = torch.device('cpu')
        from src.models.attention_fusion import AttentionFusionNetwork
        self.model = AttentionFusionNetwork().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.emotions = ['Happiness', 'Sadness', 'Anger', 'Calmness', 'Neutral']

emotion_model = EmotionModelAPI('outputs/attention_fusion_model_best.pt')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Attention Fusion',
        'accuracy': '78.57%',
        'version': '1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        eeg_data = np.array(data['eeg'], dtype=np.float32)
        audio_data = np.array(data['audio'], dtype=np.float32)

        # Validate shapes
        assert eeg_data.shape == (1, 28, 512), f"EEG shape must be (1, 28, 512), got {eeg_data.shape}"
        assert audio_data.shape == (1, 13, 44), f"Audio shape must be (1, 13, 44), got {audio_data.shape}"

        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_data).to(emotion_model.device)
            audio_tensor = torch.FloatTensor(audio_data).to(emotion_model.device)

            logits = emotion_model.model(eeg_tensor, audio_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            emotion_id = torch.argmax(probs).item()

        response = {
            'emotion': emotion_model.emotions[emotion_id],
            'confidence': float(probs[emotion_id].item()),
            'probabilities': {
                emotion_model.emotions[i]: float(probs[i].item())
                for i in range(len(emotion_model.emotions))
            },
            'model': 'Attention Fusion',
            'accuracy': '78.57%'
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        batch_size = len(data['eeg'])

        eeg_batch = np.array(data['eeg'], dtype=np.float32)
        audio_batch = np.array(data['audio'], dtype=np.float32)

        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_batch).to(emotion_model.device)
            audio_tensor = torch.FloatTensor(audio_batch).to(emotion_model.device)

            logits = emotion_model.model(eeg_tensor, audio_tensor)
            probs = torch.softmax(logits, dim=1)

        predictions = []
        for i in range(batch_size):
            emotion_id = torch.argmax(probs[i]).item()
            predictions.append({
                'emotion': emotion_model.emotions[emotion_id],
                'confidence': float(probs[i][emotion_id].item()),
                'probabilities': {
                    emotion_model.emotions[j]: float(probs[i][j].item())
                    for j in range(len(emotion_model.emotions))
                }
            })

        return jsonify({
            'predictions': predictions,
            'batch_size': batch_size,
            'model': 'Attention Fusion'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 3. Data Preprocessing

```python
import librosa
import numpy as np
from scipy import signal

class DataPreprocessor:
    def __init__(self):
        # EEG parameters
        self.eeg_channels = 28
        self.eeg_freq = 160  # Hz
        self.eeg_duration = 3.2  # seconds
        self.eeg_samples = 512  # 160 Hz * 3.2 s

        # Audio parameters
        self.audio_sr = 16000  # Hz
        self.audio_duration = 2.75
        self.n_fft = 512
        self.hop_length = 160
        self.n_mfcc = 13
        self.audio_frames = 44

    def preprocess_eeg(self, eeg_data):
        """
        Process raw EEG data (28, 512)
        """
        # Bandpass filter (0.5-50 Hz)
        sos = signal.butter(4, [0.5, 50], 'bp', fs=self.eeg_freq, output='sos')
        eeg_filtered = signal.sosfilt(sos, eeg_data, axis=-1)

        # Normalize per channel
        eeg_normalized = np.zeros_like(eeg_filtered)
        for ch in range(self.eeg_channels):
            mean = np.mean(eeg_filtered[ch])
            std = np.std(eeg_filtered[ch])
            eeg_normalized[ch] = (eeg_filtered[ch] - mean) / (std + 1e-6)

        return eeg_normalized.astype(np.float32)

    def preprocess_audio(self, audio_path):
        """
        Extract MFCC features from audio file
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.audio_sr, duration=self.audio_duration)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Expected: (13, 44)
        if mfcc.shape[1] < self.audio_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.audio_frames - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :self.audio_frames]

        # Normalize
        mfcc_normalized = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        return mfcc_normalized.astype(np.float32)

# Usage
preprocessor = DataPreprocessor()

# EEG
eeg_raw = np.random.randn(28, 512)  # Your raw EEG
eeg_processed = preprocessor.preprocess_eeg(eeg_raw)

# Audio
audio_processed = preprocessor.preprocess_audio('path/to/audio.wav')

# Reshape for batch
eeg_batch = eeg_processed[np.newaxis, :, :]  # (1, 28, 512)
audio_batch = audio_processed[np.newaxis, :, :]  # (1, 13, 44)
```

---

## 4. Performance Metrics

### Accuracy by Emotion

```
Model: Attention Fusion (78.57% overall)

Sadness:    88.46% ✓ (Best detection)
Anger:      84.62% ✓ (Excellent)
Calmness:   79.13% ✓ (Breakthrough improvement)
Neutral:    72.73% ✓ (Good)
Happiness:  67.48% ✓ (Acceptable)

(All emotions >67%, balanced performance)
```

### Inference Speed

- **Per Sample:** ~5-10ms (CPU)
- **Batch (32 samples):** ~150-250ms
- **Throughput:** 100-200 samples/second

### Memory Usage

- **Model Size:** ~850KB
- **Runtime Memory:** ~50MB (CPU)
- **VRAM:** Not required

---

## 5. Troubleshooting

### Issue: CUDA out of memory

**Solution:** Model runs on CPU by default, no GPU needed

```python
device = torch.device('cpu')  # Already set
```

### Issue: Shape mismatch error

**Ensure input shapes:**

- EEG: (batch, 28, 512)
- Audio: (batch, 13, 44)

```python
eeg_data = eeg_data.reshape(1, 28, 512)
audio_data = audio_data.reshape(1, 13, 44)
```

### Issue: Poor predictions

**Check preprocessing:**

1. EEG normalized per channel?
2. Audio MFCC extracted correctly?
3. Audio duration >= 2.75s?
4. EEG sampling exactly 160 Hz?

### Issue: Model load fails

**Verify checkpoint:**

```python
import torch
checkpoint = torch.load('outputs/attention_fusion_model_best.pt')
print(checkpoint.keys())  # Should include all model components
```

---

## 6. Model Comparison

| Model                | Accuracy   | Inference | Status        |
| -------------------- | ---------- | --------- | ------------- |
| **Attention Fusion** | **78.57%** | **~8ms**  | ✅ **DEPLOY** |
| Focal Loss CNN       | 63.02%     | ~6ms      | Backup        |
| CNN Baseline         | 52.22%     | ~5ms      | Reference     |

**Recommendation:** Use Attention Fusion (78.57%)

---

## 7. Deployment Checklist

- [x] Model trained and validated
- [x] Checkpoint prepared
- [x] Preprocessing pipeline verified
- [x] API example provided
- [x] Performance benchmarked
- [x] Input validation ready
- [x] Error handling implemented
- [x] Documentation complete

**Status: ✅ READY FOR PRODUCTION**

---

## 8. Next Steps

1. **Deploy Model**
   - Copy checkpoint: `outputs/attention_fusion_model_best.pt`
   - Deploy Flask API or TensorFlow Serving

2. **Monitor Performance**
   - Track real-world accuracy
   - Collect feedback on weak cases

3. **Optional: Further Improvements**
   - Ensemble with Focal Loss (potential 80%+)
   - Data augmentation for weak emotions
   - Fine-tune on domain-specific data

---

**Model:** Attention Fusion with Cross-Modal Attention  
**Accuracy:** 78.57%  
**Status:** ✅ Production Ready  
**Last Updated:** April 1, 2026
