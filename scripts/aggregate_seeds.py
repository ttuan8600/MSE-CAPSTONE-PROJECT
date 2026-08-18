"""Aggregate cross-validation runs across seeds, and re-test every claim.

Every configuration in this project was originally trained once, at seed 42, so
each reported difference was confounded with initialisation variance. This script
repeats configurations across seeds and reports, for each, the mean and standard
deviation over seeds rather than a single number.

What varying the seed varies
----------------------------
``subject_kfold`` takes the same seed that drives weight initialisation, so a new
seed produces **both** a different fold assignment and a different
initialisation. The spread reported here is therefore the variability of the
whole procedure -- which is the quantity a reader needs in order to know whether
a margin would survive re-running the experiment -- and not initialisation
variance in isolation.

Comparisons *within* a seed remain exactly paired: audio, EEG and their fusion at
seed $s$ share the fold assignment of seed $s$, so the per-seed differences are
computed on identical partitions.

Usage
-----
    python scripts/aggregate_seeds.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Run-directory glob per configuration. A configuration may have several runs;
#: they are keyed by the seed recorded inside cv_results.json, never by the
#: directory name, so a mislabelled tag cannot silently merge two seeds.
CONFIGURATIONS = {
    "audio": ("outputs/cv_audio_mel_pinned_*", "outputs/cv_audio_s*"),
    "eeg_standard": ("outputs/cv_eeg_de_proba_*", "outputs/cv_eeg_std_s*"),
    "eeg_adversarial": ("outputs/cv_eeg_adversarial_*", "outputs/cv_eeg_adv_s*"),
}


def load_runs(patterns) -> Dict[int, dict]:
    """Map seed -> run payload, rejecting duplicate seeds within a configuration."""
    runs: Dict[int, dict] = {}
    for pattern in patterns:
        for directory in sorted(glob.glob(pattern)):
            results = Path(directory) / "cv_results.json"
            predictions = Path(directory) / "pooled_predictions.npz"
            if not results.exists() or not predictions.exists():
                continue
            payload = json.loads(results.read_text(encoding="utf-8"))
            seed = int(payload["config"]["seed"])
            if seed in runs:
                raise ValueError(
                    f"two runs claim seed {seed} for the same configuration: "
                    f"{runs[seed]['dir']} and {directory}. Refusing to merge; "
                    f"delete or retag one."
                )
            data = np.load(predictions)
            runs[seed] = {
                "dir": os.path.basename(directory),
                "accuracy": float(payload["pooled"]["accuracy"]),
                "y_true": data["y_true"],
                "y_proba": data["y_proba"] if "y_proba" in data.files else None,
                "y_pred": data["y_pred"],
                "subject_ids": data["subject_ids"],
            }
    return runs


def summarise(values: List[float]) -> str:
    array = np.asarray(values, dtype=float)
    if array.size == 1:
        return f"{array[0]:.2%}  (n=1, no spread)"
    return (
        f"{array.mean():.2%} +/- {array.std(ddof=1):.2%}  "
        f"(n={array.size}, min {array.min():.2%}, max {array.max():.2%})"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/multiseed_summary.json")
    )
    args = parser.parse_args(argv)

    loaded = {name: load_runs(pats) for name, pats in CONFIGURATIONS.items()}

    print("=" * 76)
    print("Per-configuration accuracy across seeds")
    print("=" * 76 + "\n")

    report: dict = {"configurations": {}, "paired": {}}
    for name, runs in loaded.items():
        if not runs:
            print(f"  {name:<18} no runs found")
            continue
        seeds = sorted(runs)
        accuracies = [runs[s]["accuracy"] for s in seeds]
        print(f"  {name:<18} {summarise(accuracies)}")
        print(f"  {'':<18} seeds {seeds} -> "
              f"{[round(a, 4) for a in accuracies]}\n")
        report["configurations"][name] = {
            "seeds": seeds,
            "accuracies": accuracies,
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None,
        }

    # -- paired, per seed --------------------------------------------------
    audio = loaded["audio"]
    print("=" * 76)
    print("Paired per-seed comparisons (identical folds within each seed)")
    print("=" * 76 + "\n")

    per_seed: Dict[str, List[float]] = defaultdict(list)
    rows = []
    for seed in sorted(audio):
        pa = audio[seed]["y_proba"]
        y = audio[seed]["y_true"]
        if pa is None:
            continue
        audio_acc = (pa.argmax(1) == y).mean()
        row = {"seed": seed, "audio": float(audio_acc)}
        for variant in ("eeg_standard", "eeg_adversarial"):
            run = loaded[variant].get(seed)
            if run is None or run["y_proba"] is None:
                continue
            if not np.array_equal(run["y_true"], y):
                raise ValueError(
                    f"seed {seed}: {variant} and audio disagree on y_true; the "
                    f"runs are not aligned and cannot be paired"
                )
            fused = 0.5 * pa + 0.5 * run["y_proba"]
            fused_acc = float((fused.argmax(1) == y).mean())
            row[variant] = float((run["y_proba"].argmax(1) == y).mean())
            row[f"fusion_{variant}"] = fused_acc
            row[f"delta_{variant}"] = fused_acc - float(audio_acc)
            per_seed[f"delta_{variant}"].append(fused_acc - float(audio_acc))
            per_seed[f"fusion_{variant}"].append(fused_acc)
        rows.append(row)

    header = (f"  {'seed':<6}{'audio':>9}{'eeg-std':>9}{'eeg-adv':>9}"
              f"{'fuse-std':>10}{'fuse-adv':>10}{'adv gain':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        adv_gain = (
            row.get("fusion_eeg_adversarial", float("nan"))
            - row.get("fusion_eeg_standard", float("nan"))
        )
        print(
            f"  {row['seed']:<6}{row['audio']:>8.2%}"
            f"{row.get('eeg_standard', float('nan')):>9.2%}"
            f"{row.get('eeg_adversarial', float('nan')):>9.2%}"
            f"{row.get('fusion_eeg_standard', float('nan')):>10.2%}"
            f"{row.get('fusion_eeg_adversarial', float('nan')):>10.2%}"
            f"{adv_gain * 100:>+9.2f}pp"
        )
    report["paired"]["rows"] = rows

    print()
    for key, values in sorted(per_seed.items()):
        if not key.startswith("delta_"):
            continue
        array = np.asarray(values)
        label = key.replace("delta_", "late fusion vs audio, ")
        if array.size > 1:
            print(f"  {label:<42} {array.mean() * 100:+.2f}pp "
                  f"+/- {array.std(ddof=1) * 100:.2f}pp  (n={array.size})")
            # A margin is only credible if it survives the worst seed.
            print(f"  {'':<42} worst seed {array.min() * 100:+.2f}pp, "
                  f"best {array.max() * 100:+.2f}pp")
        else:
            print(f"  {label:<42} {array[0] * 100:+.2f}pp  (n=1)")
        report["paired"][key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else None,
            "min": float(array.min()),
            "max": float(array.max()),
            "n": int(array.size),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
