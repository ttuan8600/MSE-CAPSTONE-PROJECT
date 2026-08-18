"""Quick test script for the Flask API.

Tests all endpoints and validates responses.
"""

import requests
import json
import numpy as np
import argparse
from datetime import datetime


class APITester:
    """Test emotion recognition API."""
    
    def __init__(self, base_url: str = 'http://localhost:5000', verbose: bool = True):
        """Initialize API tester."""
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.results = []
    
    def log(self, message: str, level: str = 'INFO'):
        """Print log message."""
        if self.verbose:
            prefix = f"[{level}]"
            print(f"{prefix} {message}")
    
    def test_health(self) -> bool:
        """Test health check endpoint."""
        self.log("Testing /health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"  Status: {data['status']}", 'OK')
                self.log(f"  Model loaded: {data['model_loaded']}", 'OK')
                self.results.append(('health', True))
                return True
            else:
                self.log(f"  Unexpected status: {response.status_code}", 'ERROR')
                self.results.append(('health', False))
                return False
        
        except Exception as e:
            self.log(f"  Error: {e}", 'ERROR')
            self.results.append(('health', False))
            return False
    
    def test_emotions(self) -> bool:
        """Test emotions endpoint."""
        self.log("\nTesting /emotions endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/emotions", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                emotions = data.get('emotions', [])
                self.log(f"  Found {len(emotions)} emotions: {', '.join(emotions)}", 'OK')
                self.results.append(('emotions', True))
                return True
            else:
                self.log(f"  Unexpected status: {response.status_code}", 'ERROR')
                self.results.append(('emotions', False))
                return False
        
        except Exception as e:
            self.log(f"  Error: {e}", 'ERROR')
            self.results.append(('emotions', False))
            return False
    
    def test_model_info(self) -> bool:
        """Test model-info endpoint."""
        self.log("\nTesting /model-info endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/model-info", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"  Model: {data.get('model')}", 'OK')
                self.log(f"  Accuracy: {data.get('accuracy')}", 'OK')
                self.results.append(('model-info', True))
                return True
            else:
                self.log(f"  Unexpected status: {response.status_code}", 'ERROR')
                self.results.append(('model-info', False))
                return False
        
        except Exception as e:
            self.log(f"  Error: {e}", 'ERROR')
            self.results.append(('model-info', False))
            return False
    
    def test_single_prediction_eeg_only(self) -> bool:
        """Test single prediction with EEG only."""
        self.log("\nTesting /predict endpoint (EEG only)...")
        
        try:
            # Create dummy EEG data
            eeg = np.random.randn(30, 2500).tolist()
            
            response = requests.post(
                f"{self.base_url}/predict",
                json={"eeg": eeg},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                emotion = data.get('emotion', 'N/A')
                confidence = data.get('confidence', 0)
                self.log(f"  Predicted: {emotion} ({confidence:.2%})", 'OK')
                self.results.append(('predict-eeg', True))
                return True
            else:
                self.log(f"  Unexpected status: {response.status_code}", 'ERROR')
                self.log(f"  Response: {response.text}", 'ERROR')
                self.results.append(('predict-eeg', False))
                return False
        
        except Exception as e:
            self.log(f"  Error: {e}", 'ERROR')
            self.results.append(('predict-eeg', False))
            return False
    
    def test_single_prediction_multimodal(self) -> bool:
        """Test single prediction with EEG + Audio."""
        self.log("\nTesting /predict endpoint (EEG + Audio)...")
        
        try:
            # Create dummy data
            eeg = np.random.randn(30, 2500).tolist()
            audio = np.random.randn(13, 2101).tolist()
            
            response = requests.post(
                f"{self.base_url}/predict",
                json={"eeg": eeg, "audio": audio},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                emotion = data.get('emotion', 'N/A')
                confidence = data.get('confidence', 0)
                self.log(f"  Predicted: {emotion} ({confidence:.2%})", 'OK')
                self.results.append(('predict-multimodal', True))
                return True
            else:
                self.log(f"  Unexpected status: {response.status_code}", 'ERROR')
                self.results.append(('predict-multimodal', False))
                return False
        
        except Exception as e:
            self.log(f"  Error: {e}", 'ERROR')
            self.results.append(('predict-multimodal', False))
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint."""
        self.log("\nTesting /batch-predict endpoint...")
        
        try:
            # Create batch samples
            samples = []
            for i in range(3):
                samples.append({
                    "eeg": np.random.randn(30, 2500).tolist(),
                    "id": f"sample_{i:03d}"
                })
            
            response = requests.post(
                f"{self.base_url}/batch-predict",
                json={"samples": samples},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                num_processed = data.get('num_processed', 0)
                num_successful = data.get('num_successful', 0)
                self.log(f"  Processed: {num_successful}/{num_processed} successful", 'OK')
                
                if num_successful > 0:
                    pred = data['predictions'][0]
                    emotion = pred.get('emotion', 'N/A')
                    self.log(f"  First prediction: {emotion}", 'OK')
                
                self.results.append(('batch-predict', num_successful == num_processed))
                return num_successful == num_processed
            else:
                self.log(f"  Unexpected status: {response.status_code}", 'ERROR')
                self.results.append(('batch-predict', False))
                return False
        
        except Exception as e:
            self.log(f"  Error: {e}", 'ERROR')
            self.results.append(('batch-predict', False))
            return False
    
    def test_invalid_eeg_shape(self) -> bool:
        """Test error handling for invalid EEG shape."""
        self.log("\nTesting error handling (invalid EEG shape)...")
        
        try:
            # Wrong number of channels
            eeg = np.random.randn(32, 512).tolist()
            
            response = requests.post(
                f"{self.base_url}/predict",
                json={"eeg": eeg},
                timeout=10
            )
            
            # Should return error
            if response.status_code != 200:
                self.log(f"  Correctly rejected invalid input", 'OK')
                self.results.append(('error-handling', True))
                return True
            else:
                self.log(f"  Should have rejected invalid input", 'ERROR')
                self.results.append(('error-handling', False))
                return False
        
        except Exception as e:
            self.log(f"  Error: {e}", 'ERROR')
            self.results.append(('error-handling', False))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        print("="*70)
        print("Emotion Recognition API - Test Suite")
        print("="*70)
        print(f"Testing: {self.base_url}\n")
        
        # Run tests
        self.test_health()
        self.test_emotions()
        self.test_model_info()
        self.test_single_prediction_eeg_only()
        self.test_single_prediction_multimodal()
        self.test_batch_prediction()
        self.test_invalid_eeg_shape()
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)
        
        for test_name, result in self.results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {test_name}")
        
        print(f"\nTotal: {passed}/{total} passed")
        print("="*70 + "\n")
        
        return passed == total


def main():
    parser = argparse.ArgumentParser(description='Test emotion recognition API')
    parser.add_argument('--url', default='http://localhost:5000',
                       help='API URL (default: http://localhost:5000)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        tester = APITester(base_url=args.url, verbose=not args.quiet)
        success = tester.run_all_tests()
        
        exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
