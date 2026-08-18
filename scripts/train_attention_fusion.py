"""Train the cross-modal attention fusion model on the corrected EAV pipeline.

This is the canonical training entry point. It differs from the version that
produced the superseded 78.57% figure in three ways that change what the result
means:

* it reads genuine per-trial EEG from the preprocessing cache, rather than a
  single mis-indexed time-point shared across 100 samples;
* it splits by **subject**, so the test score measures generalisation to unseen
  people rather than to unseen trials from people already in training;
* it takes its split from ``src.data.splits``, the one implementation shared by
  every script, closing the RNG mismatch that contaminated the earlier
  evaluation.

The ``--modality`` flag runs the ablations needed to substantiate a multimodal
claim: an EEG-only and an audio-only model trained under identical conditions.
Fusion is only worth reporting if it beats both.

Examples
--------
::

    python scripts/train_attention_fusion.py
    python scripts/train_attention_fusion.py --modality eeg --epochs 40
    python scripts/train_attention_fusion.py --split-strategy pooled_random
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splits import DEFAULT_SEED, make_split
from src.models.adversarial import (
    SubjectDiscriminator,
    SubjectIndexMapper,
    adversarial_schedule,
)
from src.models.attention_fusion import CrossModalAttentionFusion
from src.models.eeg_encoder import EmotionClassifier
from src.models.feature_encoders import build_audio_encoder, build_eeg_encoder
from src.models.sequence_fusion import SequenceCrossModalFusion
from src.preprocessing.eav_dataset import EAVMultimodalDataset, eav_collate
from src.preprocessing.eav_labels import EMOTION_NAMES
from src.preprocessing.features import SpecAugment


class FocalLoss(nn.Module):
    """Focal loss with optional per-class alpha weighting."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer(
            "alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        )
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class EmotionModel(nn.Module):
    """EEG / audio / fused emotion classifier.

    A single module covering all three modality settings keeps the ablations
    honest: the encoders, classifier head and training loop are identical, and
    only the fusion path differs.
    """

    def __init__(
        self,
        modality: str = "multimodal",
        eeg_channels: int = 30,
        n_mfcc: int = 13,
        latent_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 5,
        eeg_features: str = "raw",
        audio_features: str = "mfcc",
        eeg_dropout: float = 0.5,
        audio_dropout: float = 0.3,
        fusion: str = "pooled",
    ):
        super().__init__()
        self.modality = modality
        self.eeg_features = eeg_features
        self.audio_features = audio_features
        self.fusion_mode = fusion

        if modality in ("multimodal", "eeg"):
            self.eeg_encoder = build_eeg_encoder(
                eeg_features, eeg_channels, latent_dim, eeg_dropout
            )
        if modality in ("multimodal", "audio"):
            self.audio_encoder = build_audio_encoder(
                audio_features, n_mfcc, latent_dim, audio_dropout
            )
        if modality == "multimodal":
            if fusion == "pooled":
                self.fusion = CrossModalAttentionFusion(
                    latent_dim=latent_dim, num_heads=num_heads
                )
            elif fusion == "sequence":
                self.fusion = SequenceCrossModalFusion(
                    latent_dim=latent_dim, num_heads=num_heads
                )
            else:
                raise ValueError(f"unknown fusion mode {fusion!r}")

        self.classifier = EmotionClassifier(
            latent_dim=latent_dim, num_emotions=num_classes
        )

    def encode(self, eeg=None, audio=None):
        """Return the latent fed to the classifier.

        Exposed separately so subject-adversarial training can attach a
        discriminator to the representation without a second forward pass.
        """
        if self.modality == "eeg":
            return self.eeg_encoder(eeg)
        if self.modality == "audio":
            return self.audio_encoder(audio)
        if self.fusion_mode == "sequence":
            # Fuse before pooling: each encoder yields its temporal sequence.
            return self.fusion(
                self.eeg_encoder(eeg, return_sequence=True),
                self.audio_encoder(audio, return_sequence=True),
            )
        return self.fusion(self.eeg_encoder(eeg), self.audio_encoder(audio))

    def forward(self, eeg=None, audio=None):
        return self.classifier(self.encode(eeg=eeg, audio=audio))


def git_commit() -> str:
    """Record the code version alongside the result, for reproducibility."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def run_epoch(model, loader, criterion, optimizer, device, modality, train: bool,
              augment=None, probability_sink=None, discriminator=None,
              subject_mapper=None, adv_lambda: float = 0.0):
    """Run one pass. Returns ``(mean_loss, accuracy, y_true, y_pred)``.

    ``augment`` is applied to the audio tensor on training batches only, never
    during validation or test.

    ``probability_sink``, when a list is supplied, receives the per-batch softmax
    outputs as ``(batch, n_classes)`` arrays in loader order. Late-fusion
    experiments need the full distribution rather than the argmax; collecting it
    through an optional sink keeps the return signature -- and every existing
    caller -- unchanged.
    """
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            targets = batch["emotion"].to(device)
            eeg = batch["eeg"].to(device) if modality in ("multimodal", "eeg") else None
            audio = batch["audio"].to(device) if modality in ("multimodal", "audio") else None
            if train and augment is not None and audio is not None:
                audio = augment(audio)

            latent = model.encode(eeg=eeg, audio=audio)
            logits = model.classifier(latent)
            loss = criterion(logits, targets)

            # Subject-adversarial term: the discriminator tries to identify the
            # subject, the gradient reversal makes the encoder prevent it. Train
            # only -- at evaluation the discriminator is irrelevant, and the
            # held-out subjects have no index it could legitimately predict.
            if (train and discriminator is not None and subject_mapper is not None
                    and adv_lambda > 0.0):
                subject_targets = subject_mapper(batch["subject_id"].to(device))
                subject_logits = discriminator(latent, adv_lambda)
                loss = loss + F.cross_entropy(subject_logits, subject_targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            if probability_sink is not None:
                probability_sink.append(
                    torch.softmax(logits.detach(), dim=1).cpu().numpy()
                )

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    return total_loss / max(n_batches, 1), accuracy, y_true, y_pred


def metrics_report(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, UAR and per-class recall.

    UAR (unweighted average recall) is reported because it is the metric named
    in the project proposal and is insensitive to the class imbalance that
    inflates plain accuracy.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    labels = list(range(len(EMOTION_NAMES)))

    per_class = np.asarray(
        recall_score(
            y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
        )
    )
    return {
        "accuracy": float((y_true_arr == y_pred_arr).mean()),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)
        ),
        "uar": float(per_class.mean()),
        "per_class_recall": {
            EMOTION_NAMES[i]: float(per_class[i]) for i in labels
        },
        "confusion_matrix": confusion_matrix(
            y_true_arr, y_pred_arr, labels=labels
        ).tolist(),
        "support": {
            EMOTION_NAMES[i]: int((y_true_arr == i).sum()) for i in labels
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/processed/eav")
    parser.add_argument(
        "--modality",
        choices=["multimodal", "eeg", "audio"],
        default="multimodal",
        help="multimodal runs cross-modal attention fusion; the others are ablations",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["subject_independent", "pooled_random"],
        default="subject_independent",
        help="pooled_random reproduces the original, subject-dependent (inflated) split",
    )
    parser.add_argument(
        "--eeg-features",
        choices=["raw", "de"],
        default="raw",
        help="'de' uses Euclidean-aligned differential-entropy band power, which "
             "reduces the input 25-fold and targets the measured memorisation",
    )
    parser.add_argument(
        "--audio-features",
        choices=["mfcc", "mel"],
        default="mfcc",
        help="'mel' uses a 64-band log-mel spectrogram instead of 13 MFCCs",
    )
    parser.add_argument(
        "--specaugment",
        action="store_true",
        help="apply SpecAugment to audio on training batches only",
    )
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
    parser.add_argument("--val-subjects", type=int, default=6)
    parser.add_argument("--test-subjects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--tag", default=None, help="label for this run's output folder")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = EAVMultimodalDataset(
        cache_dir=args.cache_dir,
        subjects=args.subjects,
        load_audio=args.modality in ("multimodal", "audio"),
        eeg_features=args.eeg_features,
        audio_features=args.audio_features,
    )
    print(dataset.describe())

    split_kwargs = {}
    if args.split_strategy == "subject_independent":
        split_kwargs = {
            "val_subjects": args.val_subjects,
            "test_subjects": args.test_subjects,
        }
    split = make_split(
        args.split_strategy, dataset.subject_ids, seed=args.seed, **split_kwargs
    )
    print("\n" + split.describe())

    loaders = {
        name: DataLoader(
            Subset(dataset, idx.tolist()),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            collate_fn=eav_collate,
            drop_last=(name == "train"),
        )
        for name, idx in (("train", split.train), ("val", split.val), ("test", split.test))
    }

    eeg_channels, _ = dataset.eeg_shape
    n_mfcc = dataset.audio_shape[0] if dataset.audio_shape else 13
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
    n_params = sum(p.numel() for p in model.parameters())

    augment = SpecAugment() if args.specaugment else None

    # Inverse-frequency alpha, computed from the training split only.
    train_labels = dataset.labels[split.train]
    counts = np.bincount(train_labels, minlength=len(EMOTION_NAMES)).astype(np.float64)
    alpha = (counts.sum() / (len(EMOTION_NAMES) * np.maximum(counts, 1)))
    criterion = FocalLoss(alpha=alpha, gamma=args.gamma).to(device)

    # Subject-adversarial head, fitted on the training subjects only. It must
    # never be shown a validation or test subject -- SubjectIndexMapper raises
    # rather than silently assigning one an index.
    discriminator = subject_mapper = None
    if args.adversarial:
        subject_mapper = SubjectIndexMapper(dataset.subject_ids[split.train])
        discriminator = SubjectDiscriminator(
            latent_dim=128, n_subjects=subject_mapper.n_subjects
        ).to(device)

    trainable = list(model.parameters())
    if discriminator is not None:
        trainable += list(discriminator.parameters())
    optimizer = optim.Adam(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    tag = args.tag or f"{args.modality}_{args.split_strategy}"
    run_dir = Path(args.out_dir) / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model_best.pt"

    print(f"\n{'=' * 78}")
    print(f"modality={args.modality}  split={args.split_strategy}  params={n_params:,}")
    print(f"eeg={args.eeg_features}  audio={args.audio_features}  "
          f"specaugment={args.specaugment}")
    print(f"train={len(split.train)}  val={len(split.val)}  test={len(split.test)}")
    print(f"output -> {run_dir}")
    print("=" * 78)

    history = []
    best_val = -1.0
    best_epoch = -1
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        adv_lambda = (
            adversarial_schedule(epoch - 1, args.epochs, args.adv_lambda)
            if args.adversarial else 0.0
        )
        train_loss, train_acc, _, _ = run_epoch(
            model, loaders["train"], criterion, optimizer, device, args.modality, True,
            augment=augment, discriminator=discriminator,
            subject_mapper=subject_mapper, adv_lambda=adv_lambda,
        )
        val_loss, val_acc, val_true, val_pred = run_epoch(
            model, loaders["val"], criterion, optimizer, device, args.modality, False
        )
        val_uar = float(
            recall_score(
                val_true, val_pred, labels=list(range(len(EMOTION_NAMES))),
                average="macro", zero_division=0,
            )
        )
        scheduler.step(val_acc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_uar": val_uar,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        marker = ""
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "modality": args.modality,
                    "eeg_channels": eeg_channels,
                    "n_mfcc": n_mfcc,
                    "eeg_features": args.eeg_features,
                    "audio_features": args.audio_features,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            marker = "  <- best"

        print(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"train {train_loss:.4f}/{train_acc:.4f}  "
            f"val {val_loss:.4f}/{val_acc:.4f}  uar {val_uar:.4f}  "
            f"({time.time() - t0:.0f}s){marker}"
        )

    # Restore the best checkpoint before touching the test set.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
    _, _, test_true, test_pred = run_epoch(
        model, loaders["test"], criterion, optimizer, device, args.modality, False
    )
    test_metrics = metrics_report(test_true, test_pred)

    results = {
        "run": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": vars(args),
        "split": split.to_dict(),
        "model": {
            "modality": args.modality,
            "parameters": int(n_params),
            "eeg_channels": int(eeg_channels),
            "n_mfcc": int(n_mfcc),
            "eeg_features": args.eeg_features,
            "audio_features": args.audio_features,
            "specaugment": bool(args.specaugment),
        },
        "dataset": {
            "cache_dir": str(args.cache_dir),
            "n_samples": len(dataset),
            "n_subjects": dataset.n_subjects,
            "eeg_shape": list(dataset.eeg_shape),
            "audio_shape": list(dataset.audio_shape) if dataset.audio_shape else None,
            "class_counts": dataset.class_counts(),
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test": test_metrics,
        "history": history,
        "duration_seconds": round(time.time() - started, 1),
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    chance = 1.0 / len(EMOTION_NAMES)
    print(f"\n{'=' * 78}")
    print(f"best epoch      : {best_epoch} (val acc {best_val:.4f})")
    print(f"TEST accuracy   : {test_metrics['accuracy']:.4f}  (chance {chance:.4f})")
    print(f"TEST UAR        : {test_metrics['uar']:.4f}")
    print(f"TEST macro F1   : {test_metrics['macro_f1']:.4f}")
    print("per-class recall:")
    for name, value in test_metrics["per_class_recall"].items():
        print(f"  {name:<10} {value:.4f}  (n={test_metrics['support'][name]})")
    print(f"\nresults -> {run_dir / 'results.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
