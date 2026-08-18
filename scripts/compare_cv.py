"""Compare cross-validation runs on their pooled held-out predictions.

Each run from ``scripts/cross_validate.py`` produces exactly one held-out
prediction per sample, so two runs over the same folds are directly paired across
all 4,200 samples --- five times the sample size of a single test partition, and
covering every subject rather than eight of them.

Reports, for each pair:

* **McNemar's exact test** on paired per-sample correctness.
* **A percentile bootstrap** confidence interval on the paired difference,
  resampled **by subject** rather than by sample. Samples within a subject are
  not independent --- the per-subject accuracy range on this corpus is roughly
  50 percentage points --- so a by-sample bootstrap would report an interval far
  narrower than the data supports.
* **A per-subject win count**, which shows whether any aggregate difference is
  broad or driven by a handful of participants.

Run from the project root::

    python scripts/compare_cv.py outputs/cv_audio_mel_* outputs/cv_fusion_improved_*
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

N_BOOTSTRAP = 10000
RNG_SEED = 20260809


def load_run(run_dir: Path) -> dict:
    predictions = np.load(run_dir / "pooled_predictions.npz")
    results = json.loads((run_dir / "cv_results.json").read_text(encoding="utf-8"))
    return {
        "name": results["config"].get("tag") or run_dir.name,
        "run_dir": run_dir.name,
        "y_true": predictions["y_true"],
        "y_pred": predictions["y_pred"],
        "subject_ids": predictions["subject_ids"],
        "correct": predictions["y_true"] == predictions["y_pred"],
        "pooled_accuracy": results["pooled"]["accuracy"],
        "fold_std": results["fold_accuracy_std"],
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    from scipy.stats import binomtest

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": 0, "c": 0, "discordant": 0, "p_value": 1.0}
    return {
        "b": b,
        "c": c,
        "discordant": b + c,
        "p_value": float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue),
    }


def bootstrap_by_subject(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    subject_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Cluster bootstrap: resample whole subjects, not individual samples."""
    subjects = np.unique(subject_ids)
    index_by_subject = [np.flatnonzero(subject_ids == s) for s in subjects]

    diffs = np.empty(N_BOOTSTRAP)
    acc_a = np.empty(N_BOOTSTRAP)
    acc_b = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        picked = rng.integers(0, len(subjects), size=len(subjects))
        idx = np.concatenate([index_by_subject[p] for p in picked])
        acc_a[i] = correct_a[idx].mean()
        acc_b[i] = correct_b[idx].mean()
        diffs[i] = acc_a[i] - acc_b[i]

    return {
        "accuracy_a": float(correct_a.mean()),
        "accuracy_b": float(correct_b.mean()),
        "difference": float(correct_a.mean() - correct_b.mean()),
        "ci95_a": [float(np.percentile(acc_a, 2.5)), float(np.percentile(acc_a, 97.5))],
        "ci95_b": [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))],
        "ci95_difference": [
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
        ],
    }


def per_subject_wins(a: dict, b: dict) -> dict:
    subjects = np.unique(a["subject_ids"])
    a_wins = b_wins = ties = 0
    margins = []
    for s in subjects:
        mask = a["subject_ids"] == s
        acc_a = a["correct"][mask].mean()
        acc_b = b["correct"][mask].mean()
        margins.append(float(acc_a - acc_b))
        if acc_a > acc_b:
            a_wins += 1
        elif acc_b > acc_a:
            b_wins += 1
        else:
            ties += 1
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "n_subjects": int(subjects.size),
        "median_margin": float(np.median(margins)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="cross-validation run directories")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    runs = {}
    for path in args.runs:
        run_dir = Path(path)
        if not (run_dir / "pooled_predictions.npz").exists():
            print(f"skipping {run_dir} (no pooled predictions)", file=sys.stderr)
            continue
        data = load_run(run_dir)
        runs[data["name"]] = data
        print(f"loaded {data['name']:<22} pooled acc {data['pooled_accuracy']:.4f} "
              f"(fold sd {data['fold_std']:.4f})  n={data['correct'].size}")

    if len(runs) < 2:
        print("need at least two runs to compare", file=sys.stderr)
        return 1

    reference = next(iter(runs.values()))
    for name, data in runs.items():
        if not np.array_equal(data["y_true"], reference["y_true"]):
            raise RuntimeError(f"{name}: labels differ; runs are not paired")
        if not np.array_equal(data["subject_ids"], reference["subject_ids"]):
            raise RuntimeError(f"{name}: subject ids differ; runs are not paired")

    rng = np.random.default_rng(RNG_SEED)
    comparisons = []

    print(f"\n{'comparison':<34}{'A':>8}{'B':>8}{'diff':>9}{'95% CI (by subject)':>24}{'p':>9}")
    print("-" * 92)

    for a_name, b_name in combinations(sorted(runs), 2):
        a, b = runs[a_name], runs[b_name]
        boot = bootstrap_by_subject(a["correct"], b["correct"], a["subject_ids"], rng)
        mc = mcnemar_exact(a["correct"], b["correct"])
        wins = per_subject_wins(a, b)
        significant = mc["p_value"] < 0.05

        comparisons.append({
            "model_a": a_name,
            "model_b": b_name,
            "mcnemar": mc,
            "bootstrap_by_subject": boot,
            "per_subject_wins": wins,
            "significant_at_0.05": bool(significant),
        })

        ci = boot["ci95_difference"]
        print(
            f"{a_name + ' vs ' + b_name:<34}"
            f"{boot['accuracy_a']:>8.4f}{boot['accuracy_b']:>8.4f}"
            f"{boot['difference']:>+9.4f}"
            f"{f'[{ci[0]:+.4f}, {ci[1]:+.4f}]':>24}"
            f"{mc['p_value']:>9.4f}" + ("  *" if significant else "")
        )
        print(
            f"{'':>34}per-subject: {a_name} wins {wins['a_wins']}, "
            f"{b_name} wins {wins['b_wins']}, ties {wins['ties']} "
            f"(of {wins['n_subjects']})"
        )

    print("\n* significant at alpha = 0.05 (McNemar exact, two-sided)")
    print(f"bootstrap: {N_BOOTSTRAP} resamples over subjects, seed {RNG_SEED}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n_samples": int(reference["correct"].size),
                    "n_subjects": int(np.unique(reference["subject_ids"]).size),
                    "n_bootstrap": N_BOOTSTRAP,
                    "seed": RNG_SEED,
                    "runs": {
                        k: {"run_dir": v["run_dir"],
                            "pooled_accuracy": v["pooled_accuracy"],
                            "fold_std": v["fold_std"]}
                        for k, v in runs.items()
                    },
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
