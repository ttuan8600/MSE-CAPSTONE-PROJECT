"""Verification script: Check that multimodal baseline is ready.

Run this to confirm all components are in place and working.
"""

import subprocess
import sys
from pathlib import Path

def check_files():
    """Verify all required files exist."""
    print("\n📋 CHECKING FILES\n" + "="*60)
    
    required = {
        "src/models/eeg_encoder.py": "Model definitions",
        "src/preprocessing/data_loader.py": "Dataset loaders",
        "src/preprocessing/eeg.py": "EEG utilities",
        "scripts/train.py": "Training pipeline",
        "scripts/baseline_experiment.py": "Synthetic data test",
        "scripts/compare_modalities.py": "Real data comparison",
        "tests/test_models.py": "Unit tests",
        "notebook_baseline_comparison.ipynb": "Analysis notebook",
        "BASELINE_EXPERIMENT_REPORT.md": "Report documentation",
    }
    
    all_exist = True
    for file_path, description in required.items():
        p = Path(file_path)
        if p.exists():
            print(f"  ✓ {file_path:<40} ({description})")
        else:
            print(f"  ✗ {file_path:<40} MISSING!")
            all_exist = False
    
    return all_exist


def check_imports():
    """Verify all imports work."""
    print("\n🔧 CHECKING IMPORTS\n" + "="*60)
    
    imports = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("torchaudio", "torchaudio"),
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✓ {module:<20} ({name})")
        except ImportError:
            print(f"  ✗ {module:<20} NOT FOUND!")
            all_ok = False
    
    # Local imports
    sys.path.insert(0, str(Path.cwd()))
    local_imports = [
        ("src.models.eeg_encoder", "Model definitions"),
        ("src.preprocessing.data_loader", "Data loaders"),
    ]
    
    for module, desc in local_imports:
        try:
            __import__(module)
            print(f"  ✓ {module:<40} ({desc})")
        except ImportError as e:
            print(f"  ✗ {module:<40} ERROR: {e}")
            all_ok = False
    
    return all_ok


def check_models():
    """Verify model classes are defined."""
    print("\n🧠 CHECKING MODEL CLASSES\n" + "="*60)
    
    sys.path.insert(0, str(Path.cwd()))
    try:
        from src.models.eeg_encoder import (
            EEGEncoder, EEGEncoderLSTM, AudioEncoder, 
            MultimodalFusion, EmotionClassifier
        )
        print("  ✓ EEGEncoder")
        print("  ✓ EEGEncoderLSTM")
        print("  ✓ AudioEncoder")
        print("  ✓ MultimodalFusion")
        print("  ✓ EmotionClassifier")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def check_tests():
    """Run unit tests."""
    print("\n✅ RUNNING UNIT TESTS\n" + "="*60)
    
    try:
        result = subprocess.run(
            ["pytest", "tests/test_models.py", "-v"],
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("  ✓ All tests passed!")
            return True
        else:
            print("  ✗ Some tests failed:")
            print(result.stdout.decode('utf-8'))
            return False
    except Exception as e:
        print(f"  ✗ Error running tests: {e}")
        return False


def check_data():
    """Check if EAV data is available."""
    print("\n📊 CHECKING DATA\n" + "="*60)
    
    eav_path = Path("data/raw/EAV/EAV")
    if eav_path.exists():
        subject_dirs = list(eav_path.glob("subject*"))
        print(f"  ✓ EAV data found at {eav_path}")
        print(f"    Found {len(subject_dirs)} subject directories")
        
        # Check for example files
        if subject_dirs:
            ex_subj = subject_dirs[0]
            eeg_files = list((ex_subj / "EEG").glob("*.mat")) if (ex_subj / "EEG").exists() else []
            audio_files = list((ex_subj / "Audio").glob("*.wav")) if (ex_subj / "Audio").exists() else []
            print(f"    Example subject ({ex_subj.name}): {len(eeg_files)} EEG files, {len(audio_files)} audio files")
        
        return True
    else:
        print(f"  ⚠ EAV data NOT found at {eav_path}")
        print("    Baseline will run on synthetic data only")
        return False


def main():
    print("\n" + "="*60)
    print("MULTIMODAL FUSION BASELINE - VERIFICATION CHECKLIST")
    print("="*60)
    
    checks = [
        ("Files", check_files),
        ("Imports", check_imports),
        ("Models", check_models),
        ("Tests", check_tests),
        ("Data", check_data),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n⚠ Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20} {status}")
    
    all_pass = all(results.values())
    
    print("\n" + "="*60)
    if all_pass:
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
        print("\nNext steps:")
        print("  1. Run synthetic baseline:    python scripts/baseline_experiment.py")
        print("  2. Run real data comparison:  python scripts/compare_modalities.py --quick")
        print("  3. View results in notebook:  jupyter notebook notebook_baseline_comparison.ipynb")
    else:
        print("⚠ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        sys.exit(1)
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
