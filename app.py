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

from typing import Optional

import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import EmotionPredictor

#: Checkpoint served by default. Produced by scripts/train_attention_fusion.py.
DEFAULT_MODEL_PATH = 'outputs/model_of_record.pt'


def create_app(
    model_path: Optional[str] = None,
    device: str = 'cpu',
    allow_missing_model: bool = False,
):
    """Create and configure Flask application.

    Parameters
    ----------
    model_path : str, optional
        Checkpoint to serve. Defaults to :data:`DEFAULT_MODEL_PATH`.
    device : str
        'cpu' or 'cuda'.
    allow_missing_model : bool
        When False (the default) a checkpoint that cannot be loaded raises at
        startup instead of leaving the app serving 503s.
    """
    app = Flask(__name__)
    CORS(app)

    # Load model
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # A failure here used to be swallowed, leaving the app running with
    # predictor=None and returning 503 from every prediction endpoint -- which
    # read as "service starting up" rather than "the checkpoint does not fit the
    # architecture". Startup now fails loudly unless explicitly allowed.
    try:
        app.predictor = EmotionPredictor(model_path, device=device)
        logger.info("Loaded %s", model_path)
        logger.info("Model info: %s", app.predictor.info())
    except Exception as e:
        logger.error("Failed to load model from %s: %s", model_path, e)
        if not allow_missing_model:
            raise
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
        
        Expected JSON depends on the loaded checkpoint; query /model-info.

        The audio-only model of record takes:
        {
            "audio": [[...], [...], ...]  # 64 x frames log-mel, 16 kHz, 16 ms hop
        }

        A multimodal checkpoint additionally requires:
        {
            "eeg": [[...], [...], ...]    # 30 x time_steps at 125 Hz, or
                                          # 150 x windows band-power features
        }

        Every modality the loaded model needs is mandatory. A missing one used to
        be replaced with a zero tensor, producing a confident-looking prediction
        from half a model; it is now a 400.
        
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
            if not isinstance(data, dict):
                return jsonify({'error': 'Expected a JSON object'}), 400

            eeg_data = np.array(data['eeg'], dtype=np.float32) if 'eeg' in data else None
            audio_data = np.array(data['audio'], dtype=np.float32) if 'audio' in data else None

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
                    eeg_data = (
                        np.array(sample['eeg'], dtype=np.float32)
                        if 'eeg' in sample else None
                    )
                    audio_data = (
                        np.array(sample['audio'], dtype=np.float32)
                        if 'audio' in sample else None
                    )

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
        """Describe the checkpoint that is actually loaded.

        This used to return hardcoded values -- 28 EEG channels and '78.57%'
        accuracy -- regardless of what was loaded, and stayed truthful only for
        as long as those constants happened to match. Everything here is now read
        from the checkpoint.
        """
        if app.predictor is None:
            return jsonify({'error': 'Model not loaded'}), 503
        return jsonify(app.predictor.info())
    
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
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
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
