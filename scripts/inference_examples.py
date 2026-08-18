"""Inference example scripts for emotion recognition model.

Shows how to:
1. Load and use the model directly
2. Make single predictions
3. Process batch data
4. Handle different input formats
"""

import numpy as np
import torch
from pathlib import Path
import json

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import EmotionPredictor


def example_1_basic_inference():
    """Example 1: Basic inference with dummy data."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Inference")
    print("="*70)
    
    # Load model
    model_path = 'outputs/model_of_record.pt'
    predictor = EmotionPredictor(model_path, device='cpu')
    
    # Create dummy EEG data (28 channels, 512 time steps)
    eeg_data = np.random.randn(30, 2500).astype(np.float32)
    
    # Create dummy audio data (13 MFCC channels, 128 time steps)
    audio_data = np.random.randn(13, 2101).astype(np.float32)
    
    # Make prediction
    result = predictor.predict(eeg_data, audio_data)
    
    # Display results
    print(f"\nInput shapes:")
    print(f"  EEG:   {eeg_data.shape}")
    print(f"  Audio: {audio_data.shape}")
    print(f"\nPrediction Results:")
    print(f"  Emotion:   {result['emotion']}")
    print(f"  Emotion ID: {result['emotion_id']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\nPer-Class Probabilities:")
    for emotion, prob in sorted(result['probabilities'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"  {emotion:12s}: {prob:.4f}")


def example_2_real_data():
    """Example 2: Load real data from files."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Inference with Real Data")
    print("="*70)
    
    predictor = EmotionPredictor('outputs/model_of_record.pt', device='cpu')
    
    # Try to load real data if available
    data_dir = Path('data/raw/EAV/EAV/subject1/EEG')
    
    if data_dir.exists():
        from scipy.io import loadmat
        
        eeg_files = list(data_dir.glob('*.mat'))
        if eeg_files:
            # Load first EEG file
            mat_data = loadmat(str(eeg_files[0]))
            
            # Extract EEG data (format may vary)
            eeg_data = None
            for key in ['seg', 'seg1', 'EEG', 'data']:
                if key in mat_data:
                    raw = mat_data[key]
                    if len(raw.shape) == 3:
                        eeg_data = raw[0, :, :].astype(np.float32)
                    else:
                        eeg_data = raw.astype(np.float32)
                    break
            
            if eeg_data is not None:
                print(f"\nLoaded EEG data: {eeg_data.shape}")
                
                result = predictor.predict(eeg_data)
                print(f"\nPrediction: {result['emotion']}")
                print(f"Confidence: {result['confidence']:.2%}")
            else:
                print("Could not extract EEG data from file")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Using dummy data instead...")
        
        eeg_data = np.random.randn(30, 2500).astype(np.float32)
        result = predictor.predict(eeg_data)
        
        print(f"Prediction: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")


def example_3_batch_processing():
    """Example 3: Process multiple samples in batch."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing")
    print("="*70)
    
    predictor = EmotionPredictor('outputs/model_of_record.pt', device='cpu')
    
    # Generate batch of dummy data
    batch_size = 5
    eeg_batch = [np.random.randn(30, 2500).astype(np.float32) for _ in range(batch_size)]
    
    print(f"\nProcessing {batch_size} samples...")
    results = predictor.batch_predict(eeg_batch)
    
    print(f"\nBatch Results:")
    print(f"{'Sample':<10} {'Emotion':<15} {'Confidence':<15}")
    print("-" * 40)
    
    for i, result in enumerate(results):
        emotion = result.get('emotion', 'ERROR')
        confidence = result.get('confidence', 0)
        print(f"{i+1:<10} {emotion:<15} {confidence:>6.2%}")
    
    # Summary statistics
    confidences = [r['confidence'] for r in results if 'confidence' in r]
    if confidences:
        print(f"\nStatistics:")
        print(f"  Mean confidence: {np.mean(confidences):.2%}")
        print(f"  Min confidence:  {np.min(confidences):.2%}")
        print(f"  Max confidence:  {np.max(confidences):.2%}")


def example_4_from_json():
    """Example 4: Load data from JSON file."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Load Data from JSON")
    print("="*70)
    
    # Create sample JSON file
    sample_data = {
        "eeg": np.random.randn(30, 2500).tolist(),
        "audio": np.random.randn(13, 2101).tolist(),
        "sample_id": "test_001"
    }
    
    json_file = Path('sample_input.json')
    with open(json_file, 'w') as f:
        json.dump(sample_data, f)
    
    print(f"\nSaved sample data to {json_file}")
    
    # Load and predict
    with open(json_file) as f:
        data = json.load(f)
    
    predictor = EmotionPredictor('outputs/model_of_record.pt', device='cpu')
    
    eeg = np.array(data['eeg'], dtype=np.float32)
    audio = np.array(data['audio'], dtype=np.float32)
    
    result = predictor.predict(eeg, audio)
    
    print(f"\nPrediction Result:")
    print(f"  Sample ID: {data.get('sample_id', 'N/A')}")
    print(f"  Emotion: {result['emotion']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    
    # Cleanup
    json_file.unlink()


def example_5_api_client():
    """Example 5: Use API client (requires running API server)."""
    print("\n" + "="*70)
    print("EXAMPLE 5: API Client")
    print("="*70)
    
    print("""
This example shows how to use the REST API.

1. Start the API server:
   python app.py --port 5000

2. In another terminal, run API calls:
   
   # Health check
   curl http://localhost:5000/health
   
   # Get emotions
   curl http://localhost:5000/emotions
   
   # Single prediction
   curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"eeg": [...], "audio": [...]}'
   
   # Batch prediction
   curl -X POST http://localhost:5000/batch-predict \\
     -H "Content-Type: application/json" \\
     -d '{"samples": [{"eeg": [...]}, ...]}'

3. Or use Python requests library:

   import requests
   import numpy as np
   
   # Single prediction
   data = {
       "eeg": np.random.randn(30, 2500).tolist(),
       "audio": np.random.randn(13, 2101).tolist()
   }
   
   response = requests.post('http://localhost:5000/predict', json=data)
   print(response.json())
    """)


def example_6_save_load_results():
    """Example 6: Save and load prediction results."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Save and Load Results")
    print("="*70)
    
    predictor = EmotionPredictor('outputs/model_of_record.pt', device='cpu')
    
    # Make predictions
    num_predictions = 3
    results = []
    
    print(f"\nMaking {num_predictions} predictions...\n")
    
    for i in range(num_predictions):
        eeg = np.random.randn(30, 2500).astype(np.float32)
        result = predictor.predict(eeg)
        result['sample_id'] = f"sample_{i:03d}"
        results.append(result)
        
        print(f"  Sample {i+1}: {result['emotion']} ({result['confidence']:.2%})")
    
    # Save to JSON
    output_file = Path('predictions_results.json')
    
    # Convert numpy types to Python types for JSON serialization
    serializable_results = []
    for r in results:
        sr = {
            'sample_id': r['sample_id'],
            'emotion': r['emotion'],
            'emotion_id': r['emotion_id'],
            'confidence': float(r['confidence']),
            'probabilities': {k: float(v) for k, v in r['probabilities'].items()}
        }
        serializable_results.append(sr)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Load and display
    with open(output_file) as f:
        loaded = json.load(f)
    
    print(f"\nLoaded results:")
    for r in loaded:
        print(f"  {r['sample_id']}: {r['emotion']} ({r['confidence']:.2%})")
    
    # Cleanup
    output_file.unlink()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference examples')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5, 6],
                       default=1, help='Which example to run (1-6)')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_inference,
        2: example_2_real_data,
        3: example_3_batch_processing,
        4: example_4_from_json,
        5: example_5_api_client,
        6: example_6_save_load_results
    }
    
    try:
        if args.all:
            for ex_func in examples.values():
                try:
                    ex_func()
                except Exception as e:
                    print(f"Example failed: {e}")
        else:
            examples[args.example]()
        
        print("\n" + "="*70)
        print("Examples completed!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
