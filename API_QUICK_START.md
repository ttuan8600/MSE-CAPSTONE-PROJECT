"""Quick Start Guide for Emotion Recognition API

Complete guide to get the API running in 5 minutes.
"""

# 🚀 Emotion Recognition API - Quick Start

## 5-Minute Setup

### Step 1: Install Flask (1 min)

```bash
pip install flask flask-cors
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Start the API (1 min)

```bash
python app.py --port 5000
```

You should see:

```
============================================================
Emotion Recognition REST API
============================================================
Starting server on http://0.0.0.0:5000
Model: outputs/attention_fusion_model_best.pt
Device: cpu

API Documentation:
  GET  /health           - Health check
  GET  /emotions         - List emotions
  GET  /model-info       - Model information
  POST /predict          - Single prediction
  POST /batch-predict    - Batch predictions
============================================================
```

### Step 3: Test with cURL (1 min)

In a new terminal:

```bash
# Check health
curl http://localhost:5000/health

# Get emotions
curl http://localhost:5000/emotions

# Get model info
curl http://localhost:5000/model-info
```

### Step 4: Make a Prediction (2 min)

```bash
python scripts/inference_examples.py --example 1
```

**Output:**

```
======================================================================
EXAMPLE 1: Basic Inference
======================================================================

Input shapes:
  EEG:   (28, 512)
  Audio: (13, 128)

Prediction Results:
  Emotion:    Happiness
  Emotion ID: 4
  Confidence: 0.76

Per-Class Probabilities:
  Happiness:  0.7639
  Sadness:    0.1204
  Anger:      0.0742
  Neutral:    0.0293
  Calmness:   0.0122
```

### Step 5: Test the API (1 min)

```bash
python scripts/test_api.py --url http://localhost:5000
```

---

## Common Tasks

### Run All Inference Examples

```bash
python scripts/inference_examples.py --all
```

This runs all 6 examples showing different use cases.

### Batch Process CSV Data

```bash
python scripts/batch_predict.py \
  --input data_samples.csv \
  --type csv \
  --output results.json
```

### Batch Process Directory of .mat Files

```bash
python scripts/batch_predict.py \
  --input data/raw/EAV/EAV/subject1/EEG \
  --type directory \
  --pattern "*.mat" \
  --output eeg_predictions.json
```

### Run API in Debug Mode

```bash
python app.py --port 5000 --debug
```

### Use Different Device

```bash
# Use GPU (if available)
python app.py --device cuda --port 5000

# Use CPU (default)
python app.py --device cpu --port 5000
```

---

## API Usage Examples

### Python with Requests

```python
import requests
import numpy as np

# Start API first: python app.py

# Create dummy EEG and audio data
eeg = np.random.randn(28, 512).tolist()
audio = np.random.randn(13, 128).tolist()

# Make prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'eeg': eeg, 'audio': audio}
)

result = response.json()
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### cURL Simple

```bash
# Single prediction (EEG only)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "eeg": [
      [0.1, 0.2, 0.3],
      [0.1, 0.2, 0.3],
      ...
    ]
  }'
```

### JavaScript

```javascript
const eeg = Array(28)
  .fill(0)
  .map(() => Array(512).fill(Math.random()));

fetch("http://localhost:5000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ eeg }),
})
  .then((r) => r.json())
  .then((data) =>
    console.log(`${data.emotion}: ${(data.confidence * 100).toFixed(1)}%`),
  );
```

---

## File Structure

Created files:

```
project_root/
├── app.py                           # Flask API server
├── src/
│   └── inference.py                 # Core inference class
├── scripts/
│   ├── inference_examples.py        # 6 usage examples
│   ├── batch_predict.py             # Batch processing utility
│   └── test_api.py                  # API test suite
├── API_DOCUMENTATION.md             # Full API docs
└── API_QUICK_START.md              # This file
```

---

## Endpoints Summary

| Method | Endpoint         | Purpose              |
| ------ | ---------------- | -------------------- |
| GET    | `/health`        | Check API status     |
| GET    | `/emotions`      | List emotion classes |
| GET    | `/model-info`    | Model details        |
| POST   | `/predict`       | Single prediction    |
| POST   | `/batch-predict` | Multiple predictions |

---

## Input Requirements

**EEG Data:**

- Shape: (28, time_steps)
- Type: float32
- Range: Any (auto-normalized)

**Audio Data (optional):**

- Shape: (13, time_steps)
- Type: float32
- Range: Any (auto-normalized)

---

## Troubleshooting

### Port already in use

```bash
# Find process on port 5000
lsof -i :5000

# Use different port
python app.py --port 5001
```

### Model not loading

```bash
# Check file exists
ls -lh outputs/attention_fusion_model_best.pt

# Verify torch works
python -c "import torch; print(torch.__version__)"
```

### Slow responses

```bash
# Use GPU if available
python app.py --device cuda

# Or reduce input sizes
# (Currently limited by model architecture)
```

### Connection refused

- Ensure API server is running: `python app.py`
- Check URL: `http://localhost:5000/health`
- Check firewall if connecting remotely

---

## Performance

- **Single prediction:** ~5-10ms (CPU), ~1-2ms (GPU)
- **Batch (10 samples):** ~100-200ms (CPU)
- **Memory:** ~200-300 MB

---

## Next Steps

1. ✅ API running
2. ✅ Test with curl/Python
3. 📊 Integrate into your application
4. 🚀 Deploy to production

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for full documentation.

---

**Get Help:**

```bash
# API help
python app.py --help

# Inference help
python scripts/inference_examples.py --help

# Batch predict help
python scripts/batch_predict.py --help

# API test help
python scripts/test_api.py --help
```

---

_Created: April 4, 2026_
_Model Accuracy: 78.57%_
