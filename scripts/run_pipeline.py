"""Entry point for running the EmoAI data processing and training pipeline.

This pipeline implements a two-stage training approach:
1. Pre-train EEG encoder on massive FACED dataset (123 subjects)
2. Fine-tune on EAV multimodal dataset (40 subjects with synchronized audio/EEG/video)

Usage:
    # Pre-train on FACED
    python scripts/train.py --mode pretrain --num-epochs 50 --batch-size 32
    
    # Fine-tune on EAV
    python scripts/train.py --mode finetune --num-epochs 30 --batch-size 16 \\
        --pretrained-path outputs/pretraining_YYYYMMDD_HHMMSS/best_model.pt
"""

from src.preprocessing import eeg, speech


def main():
    print("EmoAI Pipeline")
    print("=" * 60)
    print("\nTwo-stage training approach:")
    print("1. Pre-train on FACED (123 subjects, massive EEG dataset)")
    print("2. Fine-tune on EAV (40 subjects, multimodal synchronized data)")
    print("\nTo start pre-training:")
    print("  python scripts/train.py --mode pretrain --num-epochs 50")
    print("\nTo start fine-tuning:")
    print("  python scripts/train.py --mode finetune --num-epochs 30 \\")
    print("    --pretrained-path outputs/pretraining_*/best_model.pt")


if __name__ == "__main__":
    main()
