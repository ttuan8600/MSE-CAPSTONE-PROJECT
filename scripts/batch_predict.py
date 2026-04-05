"""Batch prediction utility for emotion recognition.

Process multiple samples efficiently from various input formats.
Supports:
- CSV files with EEG data
- HDF5 datasets
- JSON arrays
- Directory scanning for .mat files
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import EmotionPredictor


class BatchProcessor:
    """Process batch of emotion samples."""
    
    def __init__(self, model_path: str, device: str = 'cpu', verbose: bool = True):
        """Initialize batch processor.
        
        Parameters
        ----------
        model_path : str
            Path to model checkpoint
        device : str
            Device to run on ('cpu' or 'cuda')
        verbose : bool
            Print progress information
        """
        self.predictor = EmotionPredictor(model_path, device=device)
        self.verbose = verbose
        self.results = []
    
    def process_csv(self, csv_file: str, eeg_columns: List[int] = None) -> Dict:
        """
        Process EEG data from CSV file.
        
        Parameters
        ----------
        csv_file : str
            Path to CSV file with EEG data
        eeg_columns : list
            Column indices for 28 EEG channels (if None, uses all columns)
        
        Returns
        -------
        dict
            Processing results summary
        """
        print(f"\nProcessing CSV: {csv_file}")
        
        # Load CSV
        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        except:
            data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        print(f"Loaded shape: {data.shape}")
        
        # Select EEG columns
        if eeg_columns is not None:
            eeg_data_selected = data[:, eeg_columns]
        else:
            eeg_data_selected = data
        
        # Reshape for model (28 channels)
        num_samples = eeg_data_selected.shape[0]
        
        if num_samples < 28:
            # Samples are rows, need (num_samples, 28, time_steps)
            eeg_data_selected = eeg_data_selected.T
        
        # Process
        self.results = []
        for i in range(eeg_data_selected.shape[0]):
            sample = eeg_data_selected[i]
            
            if sample.ndim == 1:
                # Reshape to (28, -1) if needed
                if sample.shape[0] == 28:
                    sample = sample.reshape(28, 1)
            
            result = self.predictor.predict(sample)
            result['sample_id'] = i
            self.results.append(result)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} samples...")
        
        return self._summarize()
    
    def process_json(self, json_file: str) -> Dict:
        """
        Process samples from JSON file.
        
        Expected format:
        {
            "samples": [
                {"eeg": [...], "audio": [...], "id": "s1"},
                ...
            ]
        }
        """
        print(f"\nProcessing JSON: {json_file}")
        
        with open(json_file) as f:
            data = json.load(f)
        
        samples = data.get('samples', data) if isinstance(data, dict) else data
        
        if not isinstance(samples, list):
            raise ValueError("Expected list of samples or {'samples': [...]}")
        
        print(f"Found {len(samples)} samples")
        
        self.results = []
        for i, sample in enumerate(samples):
            try:
                eeg = np.array(sample['eeg'], dtype=np.float32)
                audio = np.array(sample.get('audio'), dtype=np.float32) \
                       if 'audio' in sample else None
                
                result = self.predictor.predict(eeg, audio)
                result['sample_id'] = sample.get('id', i)
                self.results.append(result)
            
            except Exception as e:
                self.results.append({
                    'error': str(e),
                    'sample_id': sample.get('id', i)
                })
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} samples...")
        
        return self._summarize()
    
    def process_directory(self, directory: str, pattern: str = '*.mat') -> Dict:
        """
        Process all files matching pattern in directory.
        
        Parameters
        ----------
        directory : str
            Path to directory
        pattern : str
            File pattern (e.g., '*.mat', '*.npy')
        
        Returns
        -------
        dict
            Processing results summary
        """
        print(f"\nProcessing directory: {directory}")
        
        dir_path = Path(directory)
        files = sorted(dir_path.glob(pattern))
        
        print(f"Found {len(files)} files matching '{pattern}'")
        
        self.results = []
        
        for i, file_path in enumerate(files):
            try:
                if file_path.suffix == '.mat':
                    from scipy.io import loadmat
                    mat_data = loadmat(str(file_path))
                    
                    # Find EEG data
                    eeg_data = None
                    for key in ['seg', 'seg1', 'EEG', 'data', 'eeg']:
                        if key in mat_data:
                            raw = mat_data[key]
                            if len(raw.shape) == 3:
                                eeg_data = raw[0, :, :].astype(np.float32)
                            else:
                                eeg_data = raw.astype(np.float32)
                            break
                    
                    if eeg_data is not None:
                        result = self.predictor.predict(eeg_data)
                        result['sample_id'] = file_path.stem
                        result['file'] = file_path.name
                        self.results.append(result)
                
                elif file_path.suffix == '.npy':
                    data = np.load(file_path)
                    result = self.predictor.predict(data)
                    result['sample_id'] = file_path.stem
                    result['file'] = file_path.name
                    self.results.append(result)
            
            except Exception as e:
                self.results.append({
                    'error': str(e),
                    'sample_id': file_path.stem,
                    'file': file_path.name
                })
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} files...")
        
        return self._summarize()
    
    def _summarize(self) -> Dict:
        """Create summary of results."""
        total = len(self.results)
        successful = sum(1 for r in self.results if 'emotion' in r)
        failed = total - successful
        
        # Emotion distribution
        emotions = [r.get('emotion') for r in self.results if 'emotion' in r]
        emotion_counts = {}
        for emotion in EmotionPredictor.EMOTION_LABELS:
            emotion_counts[emotion] = emotions.count(emotion)
        
        # Confidence stats
        confidences = [r.get('confidence', 0) for r in self.results if 'confidence' in r]
        
        summary = {
            'total_processed': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'emotion_distribution': emotion_counts,
            'confidence_stats': {
                'mean': float(np.mean(confidences)) if confidences else 0,
                'min': float(np.min(confidences)) if confidences else 0,
                'max': float(np.max(confidences)) if confidences else 0,
                'std': float(np.std(confidences)) if confidences else 0
            }
        }
        
        return summary
    
    def save_results(self, output_file: str = None) -> str:
        """Save results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"batch_predictions_{timestamp}.json"
        
        # Prepare serializable results
        serializable = []
        for r in self.results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                elif isinstance(v, np.floating):
                    sr[k] = float(v)
                elif isinstance(v, np.integer):
                    sr[k] = int(v)
                elif isinstance(v, dict):
                    sr[k] = {kb: float(vb) if isinstance(vb, np.floating) else vb
                            for kb, vb in v.items()}
                else:
                    sr[k] = v
            serializable.append(sr)
        
        with open(output_file, 'w') as f:
            json.dump({
                'predictions': serializable,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        return output_file
    
    def print_summary(self, summary: Dict = None):
        """Print results summary."""
        if summary is None and self.results:
            summary = self._summarize()
        
        if summary is None:
            return
        
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"Total processed: {summary['total_processed']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        print(f"\nEmotion Distribution:")
        for emotion, count in summary['emotion_distribution'].items():
            print(f"  {emotion:12s}: {count:3d} ({count/summary['successful']*100 if summary['successful'] > 0 else 0:5.1f}%)")
        
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {summary['confidence_stats']['mean']:.4f}")
        print(f"  Min:  {summary['confidence_stats']['min']:.4f}")
        print(f"  Max:  {summary['confidence_stats']['max']:.4f}")
        print(f"  Std:  {summary['confidence_stats']['std']:.4f}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Batch process emotion predictions')
    parser.add_argument('--model', default='outputs/attention_fusion_model_best.pt',
                       help='Path to model')
    parser.add_argument('--input', required=True,
                       help='Input file or directory')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--type', choices=['csv', 'json', 'directory'],
                       default='csv', help='Input format')
    parser.add_argument('--pattern', default='*.mat',
                       help='File pattern for directory mode')
    parser.add_argument('--device', default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchProcessor(args.model, device=args.device)
    
    # Process based on type
    if args.type == 'csv':
        summary = processor.process_csv(args.input)
    elif args.type == 'json':
        summary = processor.process_json(args.input)
    else:  # directory
        summary = processor.process_directory(args.input, pattern=args.pattern)
    
    # Print summary
    processor.print_summary(summary)
    
    # Save results
    if not args.no_save:
        processor.save_results(args.output)


if __name__ == '__main__':
    main()
