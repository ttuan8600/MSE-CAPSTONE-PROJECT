"""Comprehensive API documentation for emotion recognition service."""

# REST API Documentation - Emotion Recognition

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors torch numpy scipy
```

### 2. Start the API Server

```bash
python app.py --model outputs/attention_fusion_model_best.pt --port 5000
```

Expected output:

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

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Get emotions
curl http://localhost:5000/emotions
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the API is running and model is loaded.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "timestamp": "2024-04-04T10:30:45.123456",
  "model_loaded": true
}
```

---

### 2. List Emotions

**Endpoint:** `GET /emotions`

**Description:** Get list of supported emotion classes.

**Response (200 OK):**

```json
{
  "emotions": ["Neutral", "Anger", "Calmness", "Sadness", "Happiness"],
  "num_classes": 5,
  "emotion_ids": {
    "Neutral": 0,
    "Anger": 1,
    "Calmness": 2,
    "Sadness": 3,
    "Happiness": 4
  }
}
```

---

### 3. Model Information

**Endpoint:** `GET /model-info`

**Description:** Get model architecture and performance information.

**Response (200 OK):**

```json
{
  "model": "Multimodal Emotion Recognition",
  "modalities": ["EEG", "Audio"],
  "eeg_channels": 28,
  "audio_mfcc_channels": 13,
  "emotions": ["Neutral", "Anger", "Calmness", "Sadness", "Happiness"],
  "accuracy": "78.57%",
  "version": "1.0"
}
```

---

### 4. Single Prediction

**Endpoint:** `POST /predict`

**Description:** Predict emotion from EEG and optional audio data.

**Request Headers:**

```
Content-Type: application/json
```

**Request Body:**

```json
{
  "eeg": [
    [channel_1_values...],
    [channel_2_values...],
    ...
    [channel_28_values...]
  ],
  "audio": [
    [mfcc_1_values...],
    [mfcc_2_values...],
    ...
    [mfcc_13_values...]
  ]
}
```

**Request Body (EEG Only):**

```json
{
  "eeg": [
    [channel_1_values...],
    ...
    [channel_28_values...]
  ]
}
```

**Response (200 OK):**

```json
{
  "emotion": "Happiness",
  "emotion_id": 4,
  "confidence": 0.87,
  "probabilities": {
    "Neutral": 0.05,
    "Anger": 0.02,
    "Calmness": 0.03,
    "Sadness": 0.03,
    "Happiness": 0.87
  },
  "input_shapes": {
    "eeg": [28, 512],
    "audio": [13, 128]
  },
  "timestamp": "2024-04-04T10:30:45.123456"
}
```

**Response (400 Bad Request):**

```json
{
  "error": "EEG must have 28 channels, got 32",
  "error_type": "ValueError"
}
```

**Response (503 Service Unavailable):**

```json
{
  "error": "Model not loaded"
}
```

---

### 5. Batch Prediction

**Endpoint:** `POST /batch-predict`

**Description:** Predict emotions for multiple samples.

**Request Headers:**

```
Content-Type: application/json
```

**Request Body:**

```json
{
  "samples": [
    {
      "eeg": [[ch1...], [ch2...], ..., [ch28...]],
      "audio": [[mfcc1...], ..., [mfcc13...]],
      "id": "sample_001"
    },
    {
      "eeg": [[ch1...], [ch2...], ..., [ch28...]],
      "id": "sample_002"
    },
    ...
  ]
}
```

**Response (200 OK):**

```json
{
  "predictions": [
    {
      "emotion": "Happiness",
      "emotion_id": 4,
      "confidence": 0.87,
      "probabilities": {...},
      "input_shapes": {...},
      "id": "sample_001"
    },
    {
      "emotion": "Sadness",
      "emotion_id": 3,
      "confidence": 0.76,
      "probabilities": {...},
      "input_shapes": {...},
      "id": "sample_002"
    }
  ],
  "num_processed": 2,
  "num_successful": 2,
  "timestamp": "2024-04-04T10:30:45.123456"
}
```

---

## Client Examples

### Python with Requests

```python
import requests
import numpy as np

# Initialize
API_URL = "http://localhost:5000"

# Single prediction
eeg_data = np.random.randn(28, 512).tolist()
audio_data = np.random.randn(13, 128).tolist()

response = requests.post(
    f"{API_URL}/predict",
    json={"eeg": eeg_data, "audio": audio_data}
)

result = response.json()
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Python with urllib

```python
import urllib.request
import json
import numpy as np

data = {
    "eeg": np.random.randn(28, 512).tolist()
}

json_data = json.dumps(data).encode('utf-8')

request = urllib.request.Request(
    'http://localhost:5000/predict',
    data=json_data,
    headers={'Content-Type': 'application/json'}
)

with urllib.request.urlopen(request) as response:
    result = json.loads(response.read())
    print(result['emotion'])
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "eeg": [
      [0.1, 0.2, 0.3, ...],
      ...
    ]
  }'

# Health check
curl http://localhost:5000/health

# Get emotions
curl http://localhost:5000/emotions
```

### JavaScript/Node.js

```javascript
// Single prediction
const eegData = Array(28)
  .fill(0)
  .map(() => Array(512).fill(Math.random()));

const response = await fetch("http://localhost:5000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ eeg: eegData }),
});

const result = await response.json();
console.log(`Emotion: ${result.emotion}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

---

## Input Format Specifications

### EEG Data

- **Shape:** (28, time_steps)
- **Channels:** 28 electrode channels
- **Time Steps:** Variable (typically 512-2048)
- **Data Type:** Float32
- **Normalization:** Automatically applied (z-score per channel)

Example:

```python
eeg_data = np.random.randn(28, 512).astype(np.float32)
```

### Audio Data

- **Shape:** (13, time_steps)
- **Features:** 13 MFCC coefficients
- **Time Steps:** Variable (typically 128)
- **Data Type:** Float32
- **Normalization:** Automatically applied (z-score per channel)
- **Optional:** If not provided, dummy zeros are used

Example:

```python
audio_data = np.random.randn(13, 128).astype(np.float32)
```

---

## Error Handling

### Common Errors

**400 Bad Request** - Invalid input format

```json
{
  "error": "Expected 'eeg' in request",
  "error_type": "KeyError"
}
```

**503 Service Unavailable** - Model not loaded

```json
{
  "error": "Model not loaded"
}
```

**500 Internal Server Error** - Unexpected error

```json
{
  "error": "Internal server error"
}
```

### Handling Errors in Python

```python
import requests

try:
    response = requests.post(
        'http://localhost:5000/predict',
        json={"eeg": eeg_data}
    )
    response.raise_for_status()  # Raise for HTTP errors

    result = response.json()
    if 'emotion' in result:
        print(f"Prediction: {result['emotion']}")
    else:
        print(f"Error: {result.get('error')}")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

---

## Performance Considerations

### Latency

- **Per-Sample:** ~5-10ms (CPU)
- **Per-Sample:** ~1-2ms (GPU)
- **Batch (10 samples):** ~20-30ms total

### Throughput

- **Single-threaded:** ~100-200 samples/second (CPU)
- **Batch processing:** ~500-1000 samples/second (CPU)

### Memory

- **Model Size:** ~3.6 MB
- **Per-Prediction:** ~50-100 MB (temporary)
- **Inference Memory:** ~200-300 MB total

### Scale Requirements

For production deployment:

- **Minimum CPU:** 2 cores, 2 GB RAM
- **Recommended CPU:** 4 cores, 4 GB RAM
- **GPU:** NVIDIA GPU with 2+ GB VRAM (optional, ~10x speedup)

---

## Deployment Options

### Local Development

```bash
python app.py --host localhost --port 5000 --debug
```

### Production with Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
```

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py", "--host", "0.0.0.0"]
```

### Docker Compose

```yaml
version: "3"
services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DEVICE=cpu
    volumes:
      - ./outputs:/app/outputs
```

---

## Testing the API

### Using Testing Script

```bash
python scripts/test_api.py --url http://localhost:5000
```

### Manual Testing

```python
import json
import requests

# Test 1: Health check
print("Testing health check...")
resp = requests.get('http://localhost:5000/health')
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}\n")

# Test 2: List emotions
print("Testing emotions endpoint...")
resp = requests.get('http://localhost:5000/emotions')
print(f"Status: {resp.status_code}")
print(f"Emotions: {resp.json()['emotions']}\n")

# Test 3: Single prediction
print("Testing single prediction...")
import numpy as np
eeg = np.random.randn(28, 512).tolist()
resp = requests.post(
    'http://localhost:5000/predict',
    json={"eeg": eeg}
)
print(f"Status: {resp.status_code}")
result = resp.json()
print(f"Emotion: {result['emotion']} ({result['confidence']:.1%})\n")

# Test 4: Batch prediction
print("Testing batch prediction...")
samples = [
    {"eeg": np.random.randn(28, 512).tolist(), "id": "s1"},
    {"eeg": np.random.randn(28, 512).tolist(), "id": "s2"}
]
resp = requests.post(
    'http://localhost:5000/batch-predict',
    json={"samples": samples}
)
print(f"Status: {resp.status_code}")
result = resp.json()
print(f"Processed: {result['num_successful']}/{result['num_processed']}")
```

---

## Troubleshooting

### API won't start

- Check if port is already in use: `lsof -i :5000`
- Verify model file exists: `ls outputs/attention_fusion_model_best.pt`
- Check Python environment: `python --version`

### Model loading errors

- Ensure PyTorch is installed: `pip install torch`
- Check CUDA compatibility if using GPU: `python -m torch.utils.collect_env`
- Verify model checkpoint is valid: `python -c "import torch; torch.load('outputs/attention_fusion_model_best.pt')"`

### Slow predictions

- Consider using GPU: `--device cuda`
- Increase batch size for batch predictions
- Use dedicated hardware for production

### Invalid predictions

- Verify input shapes: 28 channels for EEG, 13 for audio
- Check data normalization
- Ensure data is in float32 format

---

## Advanced Usage

See also:

- `scripts/inference_examples.py` - Python examples
- `scripts/batch_predict.py` - Batch processing utility
- `src/inference.py` - Core inference class

---

_Last updated: April 4, 2026_
