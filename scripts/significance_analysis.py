"""Test whether the differences between the ablation runs are real.

The modality ablations differ by a small margin, and a small margin on 800 test
samples is exactly the situation where an unqualified claim of improvement is
most likely to be wrong. This script therefore computes, for each pair of runs:

* **McNemar's exact test** on the paired per-sample predictions. Because all runs
  share one test partition, the comparison is paired -- an unpaired test would
  discard that structure and understate significance in one direction while an
  eyeball comparison overstates it in the other.
* **A bootstrap 95% confidence interval** on each run's accuracy and on the
  paired difference, resampling test samples with replacement.

Predictions are recomputed from the saved checkpoints rather than stored during
training, so the test partition is reconstructed from each run's recorded split
and verified to match.

Run from the project root::

    python scripts/significance_analysis.py --json outputs/significance.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splits import make_split
from src.inference import EmotionRecognitionModel
from src.preprocessing.eav_dataset import EAVMultimodalDataset, eav_collate

N_BOOTSTRAP = 10000
RNG_SEED = 12345


def predictions_for_run(run_dir: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(y_true, y_pred, config)`` on the run's own recorded test split."""
    record = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    config = record["config"]

    dataset = EAVMultimodalDataset(
        cache_dir=config["cache_dir"],
        subjects=config.get("subjects"),
        load_audio=config["modality"] in ("multimodal", "audio"),
        eeg_features=config.get("eeg_features", "raw"),
        audio_features=config.get("audio_features", "mfcc"),
    )

    split_kwargs = {}
    if config["split_strategy"] == "subject_independent":
        split_kwargs = {
            "val_subjects": config["val_subjects"],
            "test_subjects": config["test_subjects"],
        }
    split = make_split(
        config["split_strategy"], dataset.subject_ids, seed=config["seed"], **split_kwargs
    )
    if split.sizes != record["split"]["sizes"]:
        raise RuntimeError(f"{run_dir.name}: split reconstruction does not match record")

    checkpoint = torch.load(record["checkpoint"], map_location=device, weights_only=False)
    model = EmotionRecognitionModel(
        modality=checkpoint["modality"],
        eeg_channels=checkpoint["eeg_channels"],
        n_mfcc=checkpoint["n_mfcc"],
        eeg_features=checkpoint.get("eeg_features", "raw"),
        audio_features=checkpoint.get("audio_features", "mfcc"),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    loader = DataLoader(
        Subset(dataset, split.test.tolist()),
        batch_size=32,
        shuffle=False,
        collate_fn=eav_collate,
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device) if model.modality in ("multimodal", "eeg") else None
            audio = batch["audio"].to(device) if model.modality in ("multimodal", "audio") else None
            logits = model(eeg=eeg, audio=audio)
            y_true.extend(batch["emotion"].tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return np.array(y_true), np.array(y_pred), config


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on paired correctness vectors."""
    from scipy.stats import binomtest

    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "note": "models agree on every sample"}
    result = binomtest(b, n, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "discordant": n, "p_value": float(result.pvalue)}


def bootstrap_difference(
    correct_a: np.ndarray, correct_b: np.ndarray, rng: np.random.Generator
) -> dict:
    """Percentile bootstrap CI for each accuracy and for their paired difference."""
    n = correct_a.size
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    acc_a = correct_a[idx].mean(axis=1)
    acc_b = correct_b[idx].mean(axis=1)
    diff = acc_a - acc_b
    return {
        "accuracy_a": float(correct_a.mean()),
        "accuracy_b": float(correct_b.mean()),
        "difference": float(correct_a.mean() - correct_b.mean()),
        "ci95_a": [float(np.percentile(acc_a, 2.5)), float(np.percentile(acc_a, 97.5))],
        "ci95_b": [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))],
        "ci95_difference": [
            float(np.percentile(diff, 2.5)),
            float(np.percentile(diff, 97.5)),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="explicit run directories to compare; overrides the default glob. "
             "They must share a test partition, which the label check enforces.",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = Path(args.outputs)

    # By default, only subject-independent runs share a test partition, so only
    # they are pairwise-comparable. The pooled run uses a different partition.
    if args.runs:
        candidates = [Path(r) for r in args.runs]
    else:
        candidates = sorted(outputs.glob("*_subject_independent_*"))

    runs = {}
    for run_dir in candidates:
        if not (run_dir / "results.json").exists():
            continue
        y_true, y_pred, config = predictions_for_run(run_dir, device)
        # Key on the run tag so two runs of the same modality can be compared
        # (e.g. the same model under different training budgets).
        key = config.get("tag") or config["modality"]
        runs[key] = {
            "run_dir": run_dir.name,
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": y_true == y_pred,
        }
        print(f"loaded {key:<32} {run_dir.name}  acc={runs[key]['correct'].mean():.4f}")

    if len(runs) < 2:
        print("need at least two subject-independent runs", file=sys.stderr)
        return 1

    # All runs must be aligned on the same samples in the same order.
    reference = next(iter(runs.values()))["y_true"]
    for name, data in runs.items():
        if not np.array_equal(data["y_true"], reference):
            raise RuntimeError(f"{name}: test labels differ; runs are not paired")

    rng = np.random.default_rng(RNG_SEED)
    comparisons = []
    print(f"\n{'comparison':<28}{'acc A':>8}{'acc B':>8}{'diff':>9}{'95% CI of diff':>22}{'p':>10}")
    print("-" * 86)

    for a, b in combinations(sorted(runs), 2):
        ca, cb = runs[a]["correct"], runs[b]["correct"]
        boot = bootstrap_difference(ca, cb, rng)
        mc = mcnemar_exact(ca, cb)
        significant = mc["p_value"] < 0.05

        comparisons.append(
            {
                "model_a": a,
                "model_b": b,
                "run_a": runs[a]["run_dir"],
                "run_b": runs[b]["run_dir"],
                "mcnemar": mc,
                "bootstrap": boot,
                "significant_at_0.05": bool(significant),
            }
        )
        ci = boot["ci95_difference"]
        print(
            f"{a + ' vs ' + b:<28}{boot['accuracy_a']:>8.4f}{boot['accuracy_b']:>8.4f}"
            f"{boot['difference']:>+9.4f}"
            f"{f'[{ci[0]:+.4f}, {ci[1]:+.4f}]':>22}"
            f"{mc['p_value']:>10.4f}"
            + ("  *" if significant else "")
        )

    print("\n* significant at alpha = 0.05 (McNemar exact, two-sided)")
    print(f"bootstrap: {N_BOOTSTRAP} resamples, seed {RNG_SEED}, n = {reference.size}")

    per_run = {
        name: {
            "run_dir": data["run_dir"],
            "accuracy": float(data["correct"].mean()),
        }
        for name, data in runs.items()
    }

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n_test_samples": int(reference.size),
                    "n_bootstrap": N_BOOTSTRAP,
                    "seed": RNG_SEED,
                    "runs": per_run,
                    "comparisons": comparisons,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
