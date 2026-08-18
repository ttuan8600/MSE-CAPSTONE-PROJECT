"""Can a per-trial gate recover the headroom the joint fusion model leaves?

``scripts/analyze_complementarity.py`` establishes that the EEG and audio streams
are complementary -- EEG classifies audio's errors at 47.50% against a 20% chance
rate -- and that an oracle choosing the better unimodal model per trial would
reach 80.79%. The trained cross-modal fusion model captures almost none of that,
because it applies one learned gate to every trial.

This script tests the cheapest possible remedy: leave both unimodal models
exactly as trained and combine their *output distributions* per trial, with rules
that can vary their weighting sample by sample.

Rules tested
------------
audio_only         baseline
eeg_only           reference
mean               unweighted average of the two probability vectors
weighted           fixed weight w on audio, (1-w) on EEG
entropy_gated      per-sample weight from predictive entropy: a confident model
                   is trusted more on that trial specifically
max_confidence     take whichever model is more confident on that trial
oracle             upper bound; requires the label, not attainable

Protocol
--------
**Every free parameter is chosen on the validation subjects and reported on the
test subjects**, which share no subject with validation or training. Choosing the
weight on the test set would make the comparison against the 67.13% audio
baseline meaningless, which is the trap this whole project exists to document.

Usage
-----
    python scripts/late_fusion_experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import subject_independent_split
from src.inference import EmotionRecognitionModel
from src.preprocessing.eav_dataset import EAVMultimodalDataset, eav_collate
from src.preprocessing.eav_labels import EMOTION_NAMES

EEG_CHECKPOINT = Path("outputs/eeg_de_subject_independent_20260809_004512/model_best.pt")
AUDIO_CHECKPOINT = Path(
    "outputs/audio_mel_subject_independent_20260809_004702/model_best.pt"
)

N_CLASSES = len(EMOTION_NAMES)
#: Entropy of the uniform distribution, used to normalise confidence to [0, 1].
MAX_ENTROPY = float(np.log(N_CLASSES))


def load_model(path: Path) -> EmotionRecognitionModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = EmotionRecognitionModel(
        modality=checkpoint["modality"],
        eeg_channels=checkpoint.get("eeg_channels", 30),
        n_mfcc=checkpoint.get("n_mfcc", 13),
        eeg_features=checkpoint.get("eeg_features", "raw"),
        audio_features=checkpoint.get("audio_features", "mfcc"),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def probabilities(
    model: EmotionRecognitionModel, dataset, indices: np.ndarray, batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Softmax outputs for ``indices``, in index order, with their labels."""
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eav_collate,
    )
    probs, labels = [], []
    for batch in loader:
        kwargs = {}
        if model.modality in ("multimodal", "eeg"):
            kwargs["eeg"] = batch["eeg"]
        if model.modality in ("multimodal", "audio"):
            kwargs["audio"] = batch["audio"]
        logits = model(**kwargs)
        probs.append(torch.softmax(logits, dim=1).numpy())
        labels.append(batch["emotion"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy per row, in nats."""
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(axis=1) == y).mean())


# -- combination rules --------------------------------------------------------


def rule_mean(pa: np.ndarray, pe: np.ndarray) -> np.ndarray:
    return 0.5 * pa + 0.5 * pe


def rule_weighted(pa: np.ndarray, pe: np.ndarray, w: float) -> np.ndarray:
    return w * pa + (1.0 - w) * pe


def rule_entropy_gated(pa: np.ndarray, pe: np.ndarray, temperature: float) -> np.ndarray:
    """Weight each model per sample by its confidence on that sample.

    Confidence is ``1 - H/H_max``. ``temperature`` sharpens the gate: 0 makes it
    a 50/50 average, large values make it a hard per-sample selection.
    """
    ca = 1.0 - entropy(pa) / MAX_ENTROPY
    ce = 1.0 - entropy(pe) / MAX_ENTROPY
    logits = np.stack([ca, ce], axis=1) * temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights[:, [0]] * pa + weights[:, [1]] * pe


def rule_max_confidence(pa: np.ndarray, pe: np.ndarray) -> np.ndarray:
    pick_audio = (entropy(pa) <= entropy(pe))[:, None]
    return np.where(pick_audio, pa, pe)


def oracle_accuracy(pa: np.ndarray, pe: np.ndarray, y: np.ndarray) -> float:
    return float(((pa.argmax(1) == y) | (pe.argmax(1) == y)).mean())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/late_fusion_experiment.json")
    )
    args = parser.parse_args(argv)

    dataset = EAVMultimodalDataset(eeg_features="de", audio_features="mel")
    split = subject_independent_split(
        dataset.subject_ids, val_subjects=6, test_subjects=8, seed=args.seed
    )
    print(f"validation subjects: {split.subjects['val']}")
    print(f"test subjects      : {split.subjects['test']}\n")

    audio_model = load_model(AUDIO_CHECKPOINT)
    eeg_model = load_model(EEG_CHECKPOINT)

    cache: Dict[str, dict] = {}
    for name, indices in (("val", split.val), ("test", split.test)):
        pa, y = probabilities(audio_model, dataset, indices)
        pe, y_check = probabilities(eeg_model, dataset, indices)
        assert np.array_equal(y, y_check), "label order diverged between models"
        cache[name] = {"audio": pa, "eeg": pe, "y": y}
        print(
            f"{name}: n={len(y)}  audio={accuracy(pa, y):.4f}  "
            f"eeg={accuracy(pe, y):.4f}  oracle={oracle_accuracy(pa, pe, y):.4f}"
        )

    val, test = cache["val"], cache["test"]

    # -- fit the two free parameters on validation only -----------------------
    weights = np.linspace(0.0, 1.0, 101)
    val_by_weight = [
        accuracy(rule_weighted(val["audio"], val["eeg"], w), val["y"]) for w in weights
    ]
    best_w = float(weights[int(np.argmax(val_by_weight))])

    temperatures = np.linspace(0.0, 40.0, 81)
    val_by_temperature = [
        accuracy(rule_entropy_gated(val["audio"], val["eeg"], t), val["y"])
        for t in temperatures
    ]
    best_t = float(temperatures[int(np.argmax(val_by_temperature))])

    print(
        f"\nselected on validation:  weight w={best_w:.2f}  "
        f"entropy-gate temperature={best_t:.1f}\n"
    )

    rules = {
        "audio_only": lambda d: d["audio"],
        "eeg_only": lambda d: d["eeg"],
        "mean": lambda d: rule_mean(d["audio"], d["eeg"]),
        f"weighted (w={best_w:.2f})": lambda d: rule_weighted(
            d["audio"], d["eeg"], best_w
        ),
        f"entropy_gated (T={best_t:.0f})": lambda d: rule_entropy_gated(
            d["audio"], d["eeg"], best_t
        ),
        "max_confidence": lambda d: rule_max_confidence(d["audio"], d["eeg"]),
    }

    baseline_test = accuracy(test["audio"], test["y"])
    report = {
        "val_subjects": split.subjects["val"],
        "test_subjects": split.subjects["test"],
        "selected_on_validation": {"weight": best_w, "temperature": best_t},
        "rules": {},
    }

    print(f"{'rule':<28}{'val':>9}{'test':>9}{'vs audio':>11}")
    print("-" * 57)
    for name, rule in rules.items():
        va = accuracy(rule(val), val["y"])
        ta = accuracy(rule(test), test["y"])
        delta = ta - baseline_test
        report["rules"][name] = {"val": va, "test": ta, "delta_vs_audio": delta}
        flag = "" if name == "audio_only" else f"{delta * 100:+7.2f}pp"
        print(f"{name:<28}{va:>9.2%}{ta:>9.2%}{flag:>11}")

    oracle_test = oracle_accuracy(test["audio"], test["eeg"], test["y"])
    report["oracle_test"] = oracle_test
    print("-" * 57)
    print(f"{'oracle (not attainable)':<28}{'':>9}{oracle_test:>9.2%}"
          f"{(oracle_test - baseline_test) * 100:+7.2f}pp")

    print(
        "\nNote: a single 8-subject partition carries several points of "
        "uncertainty\n(see docs/CHANGELOG.md). Treat any margin here as a "
        "screening result to be\nconfirmed by cross-validation, not as a "
        "measurement."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
