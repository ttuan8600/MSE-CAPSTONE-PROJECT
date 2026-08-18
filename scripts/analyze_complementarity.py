"""Is the EEG stream redundant with audio, or complementary but unexploited?

A mean difference of $+0.71$pp between fusion and audio establishes that fusion
does not *improve* accuracy. It does not establish *why*, and two very different
explanations produce the same mean:

**Redundant.** EEG knows only what audio already knows. On the trials audio gets
wrong, EEG is no better than chance. Fusion has nothing to add and no
architectural change would help.

**Complementary but unexploited.** EEG is informative precisely where audio
fails, but the fusion architecture cannot route that information. The mean is
flat because gains and losses cancel. Here a better fusion design *would* help.

These are distinguished by conditioning on audio's errors. If EEG scores at
chance (20%) on the subset audio gets wrong, the streams are redundant; if it
scores well above chance, the information exists and the architecture is the
binding constraint.

This script also computes an oracle upper bound -- the accuracy obtainable by an
omniscient selector that picks the better of the two unimodal models per trial.
The oracle is not achievable, but it bounds what *any* fusion of these two
particular models could reach.

Usage
-----
    python scripts/analyze_complementarity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.eav_labels import EMOTION_NAMES

DEFAULT_RUNS = {
    "eeg": "outputs/cv_eeg_de_20260809_031551",
    "audio": "outputs/cv_audio_mel_20260809_032321",
    "fusion": "outputs/cv_fusion_improved_20260809_052155",
}

N_CLASSES = len(EMOTION_NAMES)
CHANCE = 1.0 / N_CLASSES


def load_predictions(run_dirs: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Load pooled held-out predictions, asserting the runs are aligned.

    The three cross-validation runs share a dataset order and a fold assignment,
    so their pooled prediction vectors are directly comparable element by
    element. That assumption is load-bearing for every paired statistic below,
    so it is checked rather than trusted.
    """
    loaded = {}
    reference_true = reference_subjects = None

    for name, directory in run_dirs.items():
        path = Path(directory) / "pooled_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/cross_validate.py ..."
            )
        data = np.load(path)
        if reference_true is None:
            reference_true = data["y_true"]
            reference_subjects = data["subject_ids"]
        else:
            if not np.array_equal(data["y_true"], reference_true):
                raise ValueError(
                    f"{name}: y_true differs from the reference run; the pooled "
                    f"predictions are not aligned and cannot be paired."
                )
            if not np.array_equal(data["subject_ids"], reference_subjects):
                raise ValueError(f"{name}: subject_ids differ from the reference run.")
        loaded[name] = data["y_pred"]

    loaded["y_true"] = reference_true
    loaded["subject_ids"] = reference_subjects
    return loaded


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval -- well behaved for proportions near 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def analyse(predictions: Dict[str, np.ndarray]) -> dict:
    y_true = predictions["y_true"]
    eeg_correct = predictions["eeg"] == y_true
    audio_correct = predictions["audio"] == y_true
    fusion_correct = predictions["fusion"] == y_true
    n = len(y_true)

    report: dict = {
        "n_samples": int(n),
        "n_subjects": int(np.unique(predictions["subject_ids"]).size),
        "chance": CHANCE,
        "accuracy": {
            "eeg": float(eeg_correct.mean()),
            "audio": float(audio_correct.mean()),
            "fusion": float(fusion_correct.mean()),
        },
    }

    # -- the decisive test: EEG's accuracy on the trials audio gets wrong ------
    audio_wrong = ~audio_correct
    n_audio_wrong = int(audio_wrong.sum())
    eeg_rescues = int((eeg_correct & audio_wrong).sum())
    low, high = wilson_interval(eeg_rescues, n_audio_wrong)

    report["eeg_on_audio_errors"] = {
        "n_audio_errors": n_audio_wrong,
        "eeg_correct_there": eeg_rescues,
        "rate": eeg_rescues / n_audio_wrong,
        "ci95": [low, high],
        "chance": CHANCE,
        "above_chance": bool(low > CHANCE),
    }

    # Symmetric check, to confirm the asymmetry is real and not an artefact of
    # the two models simply differing in overall accuracy.
    eeg_wrong = ~eeg_correct
    n_eeg_wrong = int(eeg_wrong.sum())
    audio_rescues = int((audio_correct & eeg_wrong).sum())
    report["audio_on_eeg_errors"] = {
        "n_eeg_errors": n_eeg_wrong,
        "audio_correct_there": audio_rescues,
        "rate": audio_rescues / n_eeg_wrong,
    }

    # -- oracle upper bound ---------------------------------------------------
    either_correct = eeg_correct | audio_correct
    both_correct = eeg_correct & audio_correct
    neither = ~either_correct
    report["oracle"] = {
        "either_correct": float(either_correct.mean()),
        "both_correct": float(both_correct.mean()),
        "neither_correct": float(neither.mean()),
        "headroom_over_audio": float(either_correct.mean() - audio_correct.mean()),
    }

    # -- what fusion actually does with audio's errors ------------------------
    report["fusion_behaviour"] = {
        "recovered": int((fusion_correct & audio_wrong).sum()),
        "recovered_rate_of_audio_errors": float(
            (fusion_correct & audio_wrong).sum() / n_audio_wrong
        ),
        "broke": int((~fusion_correct & audio_correct).sum()),
        "net": int((fusion_correct & audio_wrong).sum())
        - int((~fusion_correct & audio_correct).sum()),
        "oracle_capture_rate": float(
            (fusion_correct & audio_wrong & eeg_correct).sum()
            / max(1, (eeg_correct & audio_wrong).sum())
        ),
    }

    # -- per class: where, if anywhere, does EEG carry unique signal? ----------
    per_class = {}
    for class_index, name in enumerate(EMOTION_NAMES):
        mask = y_true == class_index
        mask_audio_wrong = mask & audio_wrong
        n_wrong = int(mask_audio_wrong.sum())
        rescued = int((eeg_correct & mask_audio_wrong).sum())
        per_class[name] = {
            "n": int(mask.sum()),
            "eeg_accuracy": float(eeg_correct[mask].mean()),
            "audio_accuracy": float(audio_correct[mask].mean()),
            "fusion_accuracy": float(fusion_correct[mask].mean()),
            "n_audio_errors": n_wrong,
            "eeg_rescue_rate": (rescued / n_wrong) if n_wrong else float("nan"),
        }
    report["per_class"] = per_class

    return report


def render(report: dict) -> None:
    accuracy = report["accuracy"]
    print("=" * 74)
    print("Are EEG and audio complementary, or is EEG redundant?")
    print("=" * 74)
    print(
        f"\n{report['n_samples']} pooled held-out predictions, "
        f"{report['n_subjects']} subjects, chance = {report['chance']:.1%}\n"
    )
    print(
        f"  EEG    {accuracy['eeg']:.2%}\n"
        f"  Audio  {accuracy['audio']:.2%}\n"
        f"  Fusion {accuracy['fusion']:.2%}\n"
    )

    section = report["eeg_on_audio_errors"]
    print("-" * 74)
    print("THE DECISIVE TEST -- how good is EEG where audio fails?\n")
    print(
        f"  Audio is wrong on {section['n_audio_errors']} trials.\n"
        f"  On those trials EEG is correct {section['eeg_correct_there']} times "
        f"= {section['rate']:.2%}\n"
        f"  95% CI [{section['ci95'][0]:.2%}, {section['ci95'][1]:.2%}]   "
        f"chance = {section['chance']:.1%}"
    )
    if section["above_chance"]:
        print(
            "\n  => ABOVE CHANCE. EEG carries information audio does not have.\n"
            "     The streams are complementary; the mean gain is flat because\n"
            "     the architecture fails to route it, not because it is absent."
        )
    else:
        print(
            "\n  => NOT above chance. On audio's errors EEG is uninformative.\n"
            "     The streams are redundant."
        )

    reverse = report["audio_on_eeg_errors"]
    print(
        f"\n  (reverse direction: audio is correct on "
        f"{reverse['rate']:.2%} of EEG's {reverse['n_eeg_errors']} errors)"
    )

    oracle = report["oracle"]
    print("\n" + "-" * 74)
    print("ORACLE UPPER BOUND -- perfect per-trial choice between the two\n")
    print(
        f"  at least one correct : {oracle['either_correct']:.2%}   <- ceiling\n"
        f"  both correct         : {oracle['both_correct']:.2%}\n"
        f"  neither correct      : {oracle['neither_correct']:.2%}\n"
        f"  headroom over audio  : {oracle['headroom_over_audio']:+.2%}"
    )

    fusion = report["fusion_behaviour"]
    print("\n" + "-" * 74)
    print("WHAT FUSION ACTUALLY DOES\n")
    print(
        f"  recovers {fusion['recovered']} of audio's errors "
        f"({fusion['recovered_rate_of_audio_errors']:.1%})\n"
        f"  breaks   {fusion['broke']} of audio's correct answers\n"
        f"  net      {fusion['net']:+d} trials\n"
        f"  captures {fusion['oracle_capture_rate']:.1%} of the trials where EEG "
        f"knew the answer and audio did not"
    )

    print("\n" + "-" * 74)
    print("PER CLASS\n")
    header = (
        f"  {'emotion':<11}{'EEG':>8}{'audio':>8}{'fusion':>8}"
        f"{'aud.err':>9}{'EEG rescue':>12}"
    )
    print(header)
    for name, row in report["per_class"].items():
        print(
            f"  {name:<11}{row['eeg_accuracy']:>7.1%}{row['audio_accuracy']:>8.1%}"
            f"{row['fusion_accuracy']:>8.1%}{row['n_audio_errors']:>9d}"
            f"{row['eeg_rescue_rate']:>11.1%}"
        )
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/complementarity_analysis.json"),
        help="where to write the JSON report",
    )
    args = parser.parse_args(argv)

    predictions = load_predictions(DEFAULT_RUNS)
    report = analyse(predictions)
    render(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
