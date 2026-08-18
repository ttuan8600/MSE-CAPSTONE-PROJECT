"""Evaluate a trained checkpoint on a held-out split.

Replaces ``evaluate_best_model.py``, ``evaluate_finetuned_model.py`` and
``quick_eval.py``. Those three each re-derived the train/val/test split with
their own RNG, which is how the original evaluation ended up scoring the model on
69% of its own training data.

This script does not derive a split. It reads the split recorded in the training
run's ``results.json`` and uses exactly those indices, so an evaluation cannot
silently disagree with the training run that produced the checkpoint.

Run from the project root::

    python scripts/evaluate_model.py outputs/multimodal_subject_independent_*/
    python scripts/evaluate_model.py <run_dir> --partition val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splits import make_split
from src.inference import EmotionRecognitionModel
from src.preprocessing.eav_dataset import EAVMultimodalDataset, eav_collate
from src.preprocessing.eav_labels import EMOTION_NAMES


def load_run(run_dir: Path) -> dict:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Pass a directory produced by "
            f"scripts/train_attention_fusion.py."
        )
    return json.loads(results_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="training run directory containing results.json")
    parser.add_argument(
        "--partition", choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("--checkpoint", default=None, help="override checkpoint path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--json", default=None, help="write metrics here")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    record = load_run(run_dir)
    config = record["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EAVMultimodalDataset(
        cache_dir=config["cache_dir"],
        subjects=config.get("subjects"),
        load_audio=config["modality"] in ("multimodal", "audio"),
        eeg_features=config.get("eeg_features", "raw"),
        audio_features=config.get("audio_features", "mfcc"),
    )

    # Rebuild the identical split from the recorded strategy and seed, then
    # assert it matches the sizes the training run recorded.
    split_kwargs = {}
    if config["split_strategy"] == "subject_independent":
        split_kwargs = {
            "val_subjects": config["val_subjects"],
            "test_subjects": config["test_subjects"],
        }
    split = make_split(
        config["split_strategy"],
        dataset.subject_ids,
        seed=config["seed"],
        **split_kwargs,
    )
    recorded_sizes = record["split"]["sizes"]
    if split.sizes != recorded_sizes:
        raise RuntimeError(
            f"reconstructed split {split.sizes} does not match the split recorded "
            f"at training time {recorded_sizes}. The dataset or split code changed "
            f"since this run; the evaluation would not be comparable."
        )

    indices = getattr(split, args.partition)
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eav_collate,
    )

    checkpoint_path = Path(args.checkpoint or record["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EmotionRecognitionModel(
        modality=checkpoint.get("modality", config["modality"]),
        eeg_channels=checkpoint.get("eeg_channels", 30),
        n_mfcc=checkpoint.get("n_mfcc", 13),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device) if model.modality in ("multimodal", "eeg") else None
            audio = batch["audio"].to(device) if model.modality in ("multimodal", "audio") else None
            logits = model(eeg=eeg, audio=audio)
            y_true.extend(batch["emotion"].tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    # Reuse the training script's metric implementation so the numbers are
    # computed the same way in both places.
    from train_attention_fusion import metrics_report  # noqa: E402

    metrics = metrics_report(y_true, y_pred)

    print("=" * 78)
    print(f"run        : {run_dir.name}")
    print(f"checkpoint : {checkpoint_path.name} (epoch {checkpoint.get('epoch')})")
    print(f"modality   : {model.modality}")
    print(f"split      : {config['split_strategy']} / {args.partition} "
          f"({len(indices)} samples)")
    print("=" * 78)
    print(f"accuracy : {metrics['accuracy']:.4f}   (chance {1 / len(EMOTION_NAMES):.4f})")
    print(f"UAR      : {metrics['uar']:.4f}")
    print(f"macro F1 : {metrics['macro_f1']:.4f}")
    print("\nper-class recall:")
    for name, value in metrics["per_class_recall"].items():
        print(f"  {name:<10} {value:.4f}  (n={metrics['support'][name]})")

    print("\nconfusion matrix (rows = true, cols = predicted):")
    header = "".join(f"{n[:4]:>7}" for n in EMOTION_NAMES)
    print(f"{'':<11}{header}")
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"{EMOTION_NAMES[i]:<11}" + "".join(f"{v:>7}" for v in row))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "run": run_dir.name,
                    "checkpoint": str(checkpoint_path),
                    "partition": args.partition,
                    "modality": model.modality,
                    "n_samples": int(len(indices)),
                    **metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
