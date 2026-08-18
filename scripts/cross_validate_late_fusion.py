"""Evaluate late-fusion rules under 7-fold subject-wise cross-validation.

``scripts/late_fusion_experiment.py`` found, on a single held-out partition,
that simply averaging the two unimodal probability vectors beat both the
audio-only model and the trained cross-modal attention fusion model. A single
eight-subject partition is not sufficient to establish that -- it was exactly
this project's earlier mistake -- so this script re-evaluates the rules on the
pooled cross-validated probabilities, where every one of the 42 subjects is held
out exactly once.

It consumes the ``y_proba`` arrays written by ``scripts/cross_validate.py`` and
combines them; it trains nothing. The unimodal models were each selected on
their own fold's validation subjects, never on the test fold.

Free parameters
---------------
``mean`` and ``max_confidence`` have none. The weighted rule has one, and it is
chosen by **leave-one-fold-out selection**: the weight applied to fold *k* is
fitted on the other six folds only, so no sample contributes to choosing the
weight that is then applied to it.

Usage
-----
    python scripts/cross_validate_late_fusion.py \\
        --eeg outputs/cv_eeg_de_proba_* --audio outputs/cv_audio_mel_proba_*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cv import bootstrap_by_subject, mcnemar_exact
from src.data.splits import subject_kfold
from src.preprocessing.eav_labels import EMOTION_NAMES

N_CLASSES = len(EMOTION_NAMES)
MAX_ENTROPY = float(np.log(N_CLASSES))
BOOTSTRAP_SEED = 20260809


def load(run_dir: Path, need_proba: bool = True) -> dict:
    """Load a CV run's pooled held-out predictions.

    ``need_proba`` is False for a reference model shown only for comparison: its
    argmax predictions suffice, so runs predating probability capture can still
    be included in the table.
    """
    path = run_dir / "pooled_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    data = np.load(path)
    if need_proba and "y_proba" not in data.files:
        raise KeyError(
            f"{path} has no 'y_proba'. Re-run scripts/cross_validate.py -- only "
            f"runs made after probability capture was added carry it."
        )
    return {k: data[k] for k in data.files}


def entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)


def rule_weighted(pa: np.ndarray, pe: np.ndarray, w: float) -> np.ndarray:
    return w * pa + (1.0 - w) * pe


def rule_max_confidence(pa: np.ndarray, pe: np.ndarray) -> np.ndarray:
    return np.where((entropy(pa) <= entropy(pe))[:, None], pa, pe)


def weighted_leave_one_fold_out(
    pa: np.ndarray, pe: np.ndarray, y: np.ndarray, folds: list, grid: np.ndarray
) -> tuple:
    """Apply to each fold the weight fitted on all the *other* folds."""
    combined = np.empty_like(pa)
    chosen = []
    for fold_index, split in enumerate(folds):
        held = split.test
        others = np.setdiff1d(np.arange(len(y)), held)
        scores = [
            (rule_weighted(pa[others], pe[others], w).argmax(1) == y[others]).mean()
            for w in grid
        ]
        w = float(grid[int(np.argmax(scores))])
        chosen.append(w)
        combined[held] = rule_weighted(pa[held], pe[held], w)
    return combined, chosen


def rule_per_class(pa: np.ndarray, pe: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Blend the two models with a separate weight per class score.

    Motivated by the measured asymmetry: EEG outperforms audio on Happiness and
    trails it on the other four classes, yet the trained gate is uniform.
    """
    return weights[None, :] * pa + (1.0 - weights[None, :]) * pe


def per_class_leave_one_fold_out(
    pa: np.ndarray, pe: np.ndarray, y: np.ndarray, folds: list, grid: np.ndarray
) -> tuple:
    """Fit one weight per class on the other folds, coordinate-wise.

    A full grid over five classes is ``len(grid)**5``; coordinate ascent over two
    sweeps is enough here because the objective is smooth in each weight and the
    classes interact only through the argmax.
    """
    combined = np.empty_like(pa)
    chosen = []
    for split in folds:
        held = split.test
        others = np.setdiff1d(np.arange(len(y)), held)
        weights = np.full(pa.shape[1], 0.5)
        for _ in range(2):
            for c in range(pa.shape[1]):
                best_score, best_w = -1.0, weights[c]
                for w in grid:
                    trial = weights.copy()
                    trial[c] = w
                    score = (
                        rule_per_class(pa[others], pe[others], trial).argmax(1)
                        == y[others]
                    ).mean()
                    if score > best_score:
                        best_score, best_w = score, w
                weights[c] = best_w
        chosen.append([round(float(w), 2) for w in weights])
        combined[held] = rule_per_class(pa[held], pe[held], weights)
    return combined, chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eeg", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, default=None,
                        help="trained cross-modal fusion CV run, for reference")
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/late_fusion_cv.json")
    )
    args = parser.parse_args(argv)

    eeg, audio = load(args.eeg), load(args.audio)
    if not np.array_equal(eeg["y_true"], audio["y_true"]):
        raise ValueError("y_true differs between runs; predictions are not aligned")
    if not np.array_equal(eeg["subject_ids"], audio["subject_ids"]):
        raise ValueError("subject_ids differ between runs")

    y = audio["y_true"]
    subject_ids = audio["subject_ids"]
    pa, pe = audio["y_proba"], eeg["y_proba"]

    folds = subject_kfold(subject_ids, n_folds=args.folds, seed=args.seed)

    grid = np.linspace(0.0, 1.0, 101)
    weighted, chosen_weights = weighted_leave_one_fold_out(pa, pe, y, folds, grid)
    coarse = np.linspace(0.0, 1.0, 21)
    per_class, chosen_per_class = per_class_leave_one_fold_out(
        pa, pe, y, folds, coarse
    )

    candidates: Dict[str, np.ndarray] = {
        "audio_only": pa,
        "eeg_only": pe,
        "mean": 0.5 * pa + 0.5 * pe,
        "weighted (LOFO)": weighted,
        "per_class (LOFO)": per_class,
        "max_confidence": rule_max_confidence(pa, pe),
    }

    correct = {name: (p.argmax(1) == y) for name, p in candidates.items()}

    if args.fusion is not None:
        fusion = load(args.fusion, need_proba=False)
        if not np.array_equal(fusion["y_true"], y):
            raise ValueError("fusion run is not aligned with the unimodal runs")
        correct["trained attention fusion"] = fusion["y_pred"] == y

    baseline = correct["audio_only"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    oracle = float((correct["audio_only"] | correct["eeg_only"]).mean())

    print(f"n = {len(y)} pooled held-out predictions, "
          f"{np.unique(subject_ids).size} subjects")
    print(f"leave-one-fold-out weights on audio: "
          f"{[round(w, 2) for w in chosen_weights]}\n")
    print(f"{'rule':<28}{'accuracy':>10}{'vs audio':>11}{'95% CI':>20}{'p':>10}")
    print("-" * 79)

    report = {"n": int(len(y)), "oracle": oracle,
              "lofo_weights": chosen_weights,
              "lofo_per_class_weights": chosen_per_class,
              "class_order": list(EMOTION_NAMES),
              "rules": {}}

    for name, is_correct in correct.items():
        accuracy = float(is_correct.mean())
        entry = {"accuracy": accuracy}
        if name == "audio_only":
            print(f"{name:<28}{accuracy:>10.2%}{'--':>11}{'':>20}{'':>10}")
        else:
            boot = bootstrap_by_subject(is_correct, baseline, subject_ids, rng)
            mcn = mcnemar_exact(is_correct, baseline)
            entry.update({
                "delta_vs_audio": boot["difference"],
                "ci95_difference": boot["ci95_difference"],
                "p_value": mcn["p_value"],
                "mcnemar": mcn,
                "significant": bool(mcn["p_value"] < 0.05),
            })
            ci = f"[{boot['ci95_difference'][0] * 100:+.2f}, {boot['ci95_difference'][1] * 100:+.2f}]"
            mark = " *" if mcn["p_value"] < 0.05 else ""
            print(f"{name:<28}{accuracy:>10.2%}"
                  f"{boot['difference'] * 100:>+10.2f}pp{ci:>20}"
                  f"{mcn['p_value']:>10.4f}{mark}")
        report["rules"][name] = entry

    print("-" * 79)
    print(f"{'oracle (not attainable)':<28}{oracle:>10.2%}"
          f"{(oracle - baseline.mean()) * 100:>+10.2f}pp")
    print("\n* significant at 0.05 (McNemar exact, paired)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
