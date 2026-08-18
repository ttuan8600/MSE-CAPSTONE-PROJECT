"""Subject-wise k-fold cross-validation.

Exists because a single held-out partition could not rank two candidate models:
audio-only and fusion reversed order between the 6-subject validation partition
and the 8-subject test partition. With between-subject variance that large, one
partition is not enough to choose between them.

Here every subject takes a turn in a test fold, so aggregating predictions across
folds yields exactly one held-out prediction per sample --- 4,200 predictions
covering all 42 subjects. Two figures are reported:

* **Pooled accuracy**, computed over all held-out predictions at once. This is
  the headline estimate.
* **Per-fold mean and standard deviation**, which shows how much the single-
  partition figures were at the mercy of which subjects landed where.

Per-subject accuracies are also recorded, since inter-subject variability is the
quantity this whole line of investigation keeps running into.

Run from the project root::

    python scripts/cross_validate.py --modality audio --audio-features mel --specaugment
    python scripts/cross_validate.py --modality multimodal --eeg-features de \\
        --audio-features mel --specaugment
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.data.splits import DEFAULT_SEED, subject_kfold
from src.models.adversarial import (
    SubjectDiscriminator,
    SubjectIndexMapper,
    adversarial_schedule,
)
from src.preprocessing.eav_dataset import EAVMultimodalDataset, eav_collate
from src.preprocessing.eav_labels import EMOTION_NAMES
from src.preprocessing.features import SpecAugment
from train_attention_fusion import (  # noqa: E402  -- reuse, do not duplicate
    EmotionModel,
    FocalLoss,
    git_commit,
    metrics_report,
    run_epoch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/processed/eav")
    parser.add_argument("--modality", choices=["multimodal", "eeg", "audio"],
                        default="multimodal")
    parser.add_argument("--eeg-features", choices=["raw", "de"], default="raw")
    parser.add_argument("--audio-features", choices=["mfcc", "mel"], default="mfcc")
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument(
        "--fusion", choices=["pooled", "sequence"], default="pooled",
        help="pooled fuses two pooled vectors (attention over length-1 sequences, "
             "i.e. no attention); sequence fuses the encoders' temporal sequences "
             "before pooling",
    )
    parser.add_argument(
        "--adversarial", action="store_true",
        help="subject-adversarial training via gradient reversal",
    )
    parser.add_argument("--adv-lambda", type=float, default=0.3,
                        help="maximum adversarial weight, ramped from 0")
    parser.add_argument("--eeg-dropout", type=float, default=0.5)
    parser.add_argument("--audio-dropout", type=float, default=0.3)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--val-subjects", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Built once and shared across folds -- the memory maps are several GB.
    dataset = EAVMultimodalDataset(
        cache_dir=args.cache_dir,
        load_audio=args.modality in ("multimodal", "audio"),
        eeg_features=args.eeg_features,
        audio_features=args.audio_features,
    )
    print(dataset.describe())

    folds = subject_kfold(
        dataset.subject_ids,
        n_folds=args.folds,
        val_subjects=args.val_subjects,
        seed=args.seed,
    )

    tag = args.tag or f"cv_{args.modality}_{args.eeg_features}_{args.audio_features}"
    run_dir = Path(args.out_dir) / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eeg_channels, _ = dataset.eeg_shape
    n_mfcc = dataset.audio_shape[0] if dataset.audio_shape else 13

    print(f"\n{'=' * 78}")
    print(f"{args.folds}-fold subject-wise cross-validation")
    print(f"modality={args.modality} eeg={args.eeg_features} "
          f"audio={args.audio_features} specaugment={args.specaugment}")
    print(f"epochs={args.epochs}  output -> {run_dir}")
    print("=" * 78)

    # One held-out prediction per sample, filled in as each fold completes.
    pooled_true = np.full(len(dataset), -1, dtype=np.int64)
    pooled_pred = np.full(len(dataset), -1, dtype=np.int64)
    pooled_proba = np.full((len(dataset), len(EMOTION_NAMES)), np.nan, dtype=np.float32)

    fold_records = []
    started = time.time()

    for fold_index, split in enumerate(folds, 1):
        torch.manual_seed(args.seed + fold_index)
        np.random.seed(args.seed + fold_index)

        loaders = {
            name: DataLoader(
                Subset(dataset, idx.tolist()),
                batch_size=args.batch_size,
                shuffle=(name == "train"),
                collate_fn=eav_collate,
                drop_last=(name == "train"),
            )
            for name, idx in (("train", split.train), ("val", split.val),
                              ("test", split.test))
        }

        model = EmotionModel(
            modality=args.modality,
            eeg_channels=eeg_channels,
            n_mfcc=n_mfcc,
            eeg_features=args.eeg_features,
            audio_features=args.audio_features,
            eeg_dropout=args.eeg_dropout,
            audio_dropout=args.audio_dropout,
            fusion=args.fusion,
        ).to(device)

        # The discriminator is fitted per fold on that fold's training subjects
        # only, so it can never be shown a held-out subject.
        discriminator = subject_mapper = None
        if args.adversarial:
            subject_mapper = SubjectIndexMapper(dataset.subject_ids[split.train])
            discriminator = SubjectDiscriminator(
                latent_dim=128, n_subjects=subject_mapper.n_subjects
            ).to(device)

        counts = np.bincount(
            dataset.labels[split.train], minlength=len(EMOTION_NAMES)
        ).astype(np.float64)
        alpha = counts.sum() / (len(EMOTION_NAMES) * np.maximum(counts, 1))
        criterion = FocalLoss(alpha=alpha, gamma=args.gamma).to(device)
        trainable = list(model.parameters())
        if discriminator is not None:
            trainable += list(discriminator.parameters())
        optimizer = optim.Adam(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        augment = SpecAugment() if args.specaugment else None

        best_val, best_epoch = -1.0, -1
        best_state = None
        fold_started = time.time()

        for epoch in range(1, args.epochs + 1):
            adv_lambda = (
                adversarial_schedule(epoch - 1, args.epochs, args.adv_lambda)
                if args.adversarial else 0.0
            )
            train_loss, train_acc, _, _ = run_epoch(
                model, loaders["train"], criterion, optimizer, device,
                args.modality, True, augment=augment,
                discriminator=discriminator, subject_mapper=subject_mapper,
                adv_lambda=adv_lambda,
            )
            _, val_acc, _, _ = run_epoch(
                model, loaders["val"], criterion, optimizer, device,
                args.modality, False,
            )
            scheduler.step(val_acc)
            if val_acc > best_val:
                best_val, best_epoch = val_acc, epoch
                # Keep in memory: 7 folds x 3.5 MB of checkpoints is not worth
                # writing to disk when only the metrics are needed.
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Restore the best-validation weights before touching this fold's test
        # subjects. Selection never sees them.
        if best_state is not None:
            model.load_state_dict(best_state)
        fold_probabilities: list = []
        _, test_acc, y_true, y_pred = run_epoch(
            model, loaders["test"], criterion, optimizer, device,
            args.modality, False, probability_sink=fold_probabilities,
        )

        pooled_true[split.test] = y_true
        pooled_pred[split.test] = y_pred
        # Held-out probability distributions, needed to evaluate late-fusion
        # rules without retraining. The test loader is not shuffled, so batch
        # order matches split.test.
        pooled_proba[split.test] = np.concatenate(fold_probabilities)

        subject_ids = dataset.subject_ids[split.test]
        per_subject = {
            int(s): float((np.array(y_true)[subject_ids == s]
                           == np.array(y_pred)[subject_ids == s]).mean())
            for s in np.unique(subject_ids)
        }

        fold_metrics = metrics_report(y_true, y_pred)
        fold_records.append(
            {
                "fold": fold_index,
                "test_subjects": split.subjects["test"],
                "val_subjects": split.subjects["val"],
                "n_test": int(split.test.size),
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val,
                "test_accuracy": fold_metrics["accuracy"],
                "test_uar": fold_metrics["uar"],
                "test_macro_f1": fold_metrics["macro_f1"],
                "per_subject_accuracy": per_subject,
                "duration_seconds": round(time.time() - fold_started, 1),
            }
        )

        print(
            f"fold {fold_index}/{args.folds}  "
            f"test subjects {sorted(split.subjects['test'])}  "
            f"val {best_val:.4f}@{best_epoch}  test {test_acc:.4f}  "
            f"({(time.time() - fold_started) / 60:.1f} min)",
            flush=True,
        )
        # Persist after each fold so a long run is never lost.
        (run_dir / "folds_partial.json").write_text(
            json.dumps(fold_records, indent=2), encoding="utf-8"
        )

    if (pooled_true < 0).any():
        raise RuntimeError("some samples were never assigned to a test fold")

    pooled = metrics_report(pooled_true.tolist(), pooled_pred.tolist())
    fold_accuracies = np.array([f["test_accuracy"] for f in fold_records])
    fold_uars = np.array([f["test_uar"] for f in fold_records])

    all_subject_acc = {}
    for record in fold_records:
        all_subject_acc.update(record["per_subject_accuracy"])
    subject_values = np.array(list(all_subject_acc.values()))

    results = {
        "run": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": vars(args),
        "n_folds": args.folds,
        "n_samples": int(len(dataset)),
        "n_subjects": int(dataset.n_subjects),
        "pooled": pooled,
        "fold_accuracy_mean": float(fold_accuracies.mean()),
        "fold_accuracy_std": float(fold_accuracies.std(ddof=1)),
        "fold_uar_mean": float(fold_uars.mean()),
        "fold_uar_std": float(fold_uars.std(ddof=1)),
        "subject_accuracy_mean": float(subject_values.mean()),
        "subject_accuracy_std": float(subject_values.std(ddof=1)),
        "subject_accuracy_min": float(subject_values.min()),
        "subject_accuracy_max": float(subject_values.max()),
        "per_subject_accuracy": {str(k): v for k, v in sorted(all_subject_acc.items())},
        "folds": fold_records,
        "duration_seconds": round(time.time() - started, 1),
    }
    (run_dir / "cv_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if np.isnan(pooled_proba).any():
        raise RuntimeError(
            "pooled probabilities have gaps: some samples were never in a test "
            "fold, so the folds do not partition the dataset."
        )
    np.savez(
        run_dir / "pooled_predictions.npz",
        y_true=pooled_true,
        y_pred=pooled_pred,
        y_proba=pooled_proba,
        subject_ids=dataset.subject_ids,
    )
    (run_dir / "folds_partial.json").unlink(missing_ok=True)

    print(f"\n{'=' * 78}")
    print(f"POOLED over all {len(dataset)} held-out predictions ({dataset.n_subjects} subjects)")
    print(f"  accuracy : {pooled['accuracy']:.4f}   (chance {1 / len(EMOTION_NAMES):.4f})")
    print(f"  UAR      : {pooled['uar']:.4f}")
    print(f"  macro F1 : {pooled['macro_f1']:.4f}")
    print(f"\nper-fold accuracy : {fold_accuracies.mean():.4f} "
          f"+/- {fold_accuracies.std(ddof=1):.4f} (sd over {args.folds} folds)")
    print(f"per-subject range : {subject_values.min():.4f} to {subject_values.max():.4f} "
          f"(sd {subject_values.std(ddof=1):.4f} over {len(subject_values)} subjects)")
    print(f"\ntotal {(time.time() - started) / 60:.1f} min")
    print(f"results -> {run_dir / 'cv_results.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
