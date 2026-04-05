"""Flask REST API for emotion recognition model.

Provides REST endpoints for real-time emotion prediction from EEG+Audio data.

Usage:
    python app.py --model outputs/attention_fusion_model_best.pt --port 5000

API Endpoints:
    POST /predict - Single prediction
    POST /batch-predict - Batch predictions  
    GET /health - Health check
    GET /emotions - List supported emotions
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import EmotionPredictor


def create_app(model_path: str = None, device: str = 'cpu'):
    """Create and configure Flask application."""
    app = Flask(__name__)
    CORS(app)
    
    # Load model
    if model_path is None:
        model_path = 'outputs/attention_fusion_model_best.pt'
    
    try:
        app.predictor = EmotionPredictor(model_path, device=device)
        logger.info(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        app.predictor = None
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy' if app.predictor else 'error',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': app.predictor is not None
        })
    
    # Get supported emotions
    @app.route('/emotions', methods=['GET'])
    def get_emotions():
        """Get list of supported emotion classes."""
        return jsonify({
            'emotions': EmotionPredictor.EMOTION_LABELS,
            'num_classes': len(EmotionPredictor.EMOTION_LABELS),
            'emotion_ids': {
                emotion: idx
                for idx, emotion in enumerate(EmotionPredictor.EMOTION_LABELS)
            }
        })
    
    # Single prediction endpoint
    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Predict emotion from EEG and optional audio data.
        
        Expected JSON:
        {
            "eeg": [[...], [...], ...],  # 28 x time_steps array
            "audio": [[...], [...], ...] # 13 x time_steps array (optional)
        }
        
        Returns:
        {
            "emotion": "Happiness",
            "emotion_id": 4,
            "confidence": 0.87,
            "probabilities": {
                "Neutral": 0.05,
                "Anger": 0.02,
                ...
            },
            "timestamp": "2024-04-04T10:30:45.123456"
        }
        """
        if app.predictor is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        try:
            data = request.get_json()
            
            # Validate inputs
            if 'eeg' not in data:
                return jsonify({'error': 'Missing required field: eeg'}), 400
            
            eeg_data = np.array(data['eeg'], dtype=np.float32)
            audio_data = np.array(data.get('audio'), dtype=np.float32) if 'audio' in data else None
            
            # Run prediction
            result = app.predictor.predict(eeg_data, audio_data)
            result['timestamp'] = datetime.now().isoformat()
            
            return jsonify(result), 200
        
        except Exception as e:
            logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
            return jsonify({
                'error': str(e),
                'error_type': type(e).__name__
            }), 400
    
    # Batch prediction endpoint
    @app.route('/batch-predict', methods=['POST'])
    def batch_predict():
        """
        Predict emotions for multiple samples.
        
        Expected JSON:
        {
            "samples": [
                {
                    "eeg": [...],
                    "audio": [...],  # optional
                    "id": "sample_1"  # optional
                },
                ...
            ]
        }
        
        Returns:
        {
            "predictions": [
                {
                    "emotion": "Happiness",
                    "confidence": 0.87,
                    "id": "sample_1"  # if provided in input
                },
                ...
            ],
            "num_processed": 5,
            "timestamp": "2024-04-04T10:30:45.123456"
        }
        """
        if app.predictor is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        try:
            data = request.get_json()
            
            if 'samples' not in data or not isinstance(data['samples'], list):
                return jsonify({'error': 'Expected "samples" list in JSON'}), 400
            
            samples = data['samples']
            predictions = []
            
            for i, sample in enumerate(samples):
                try:
                    eeg_data = np.array(sample['eeg'], dtype=np.float32)
                    audio_data = np.array(sample.get('audio'), dtype=np.float32) \
                                if 'audio' in sample else None
                    
                    result = app.predictor.predict(eeg_data, audio_data)
                    
                    # Add sample ID if provided
                    if 'id' in sample:
                        result['id'] = sample['id']
                    
                    predictions.append(result)
                
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    predictions.append({
                        'error': str(e),
                        'sample_index': i,
                        'id': sample.get('id')
                    })
            
            return jsonify({
                'predictions': predictions,
                'num_processed': len(predictions),
                'num_successful': sum(1 for p in predictions if 'emotion' in p),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}\n{traceback.format_exc()}")
            return jsonify({
                'error': str(e),
                'error_type': type(e).__name__
            }), 400
    
    # Model info endpoint
    @app.route('/model-info', methods=['GET'])
    def model_info():
        """Get model information."""
        return jsonify({
            'model': 'Multimodal Emotion Recognition',
            'modalities': ['EEG', 'Audio'],
            'eeg_channels': 28,
            'audio_mfcc_channels': 13,
            'emotions': EmotionPredictor.EMOTION_LABELS,
            'accuracy': '78.57%',
            'version': '1.0'
        })
    
    # Error handler
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Recognition API')
    parser.add_argument('--model', default='outputs/attention_fusion_model_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run API on')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--device', default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    app = create_app(model_path=args.model, device=args.device)
    
    print(f"\n{'='*60}")
    print("Emotion Recognition REST API")
    print(f"{'='*60}")
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"\nAPI Documentation:")
    print(f"  GET  /health           - Health check")
    print(f"  GET  /emotions         - List emotions")
    print(f"  GET  /model-info       - Model information")
    print(f"  POST /predict          - Single prediction")
    print(f"  POST /batch-predict    - Batch predictions")
    print(f"{'='*60}\n")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=args.debug
    )
