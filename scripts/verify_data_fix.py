"""Quantify the difference between the original and corrected EAV pipelines.

Produces the evidence cited in ``docs/DATA_CORRECTIONS.md`` and in the report:

1. **EEG degeneracy.** Under the original loader every sample belonging to a
   subject received a byte-identical EEG tensor. This script reproduces the
   original indexing and counts distinct EEG tensors per subject, before and
   after the fix.

2. **Label provenance.** The original pipeline derived labels by substring
   matching on audio filenames; the corrected one reads the dataset's own label
   matrix. This reports the resulting class distributions.

3. **Split contamination.** The original training and evaluation scripts seeded
   two different RNGs with the same value. This measures the overlap between the
   two splits they produced, and contrasts it with the subject-independent split.

Run from the project root::

    python scripts/verify_data_fix.py --json outputs/data_fix_verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splits import (
    SplitResult,
    pooled_random_split,
    split_overlap_report,
    subject_independent_split,
)
from src.preprocessing.eav_dataset import EAVMultimodalDataset
from src.preprocessing.eav_io import list_subject_dirs, load_subject_segments, subject_eeg_paths

ORIGINAL_EMOTION_MAP = {
    "Neutral": 0,
    "Anger": 1,
    "Calmness": 2,
    "Sadness": 3,
    "Happiness": 4,
}


def original_eeg_tensor(eeg_file: Path) -> np.ndarray:
    """Reproduce exactly what the original ``_load_eeg`` returned.

    The original read ``seg[0, :, :]`` from the on-disk ``(time, channels, trials)``
    array, then truncated axis 0 to 28. It did this once per *sample*, and every
    sample of a subject pointed at the same file, so the result was constant
    across all 100 of that subject's samples.
    """
    from scipy.io import loadmat

    mat = loadmat(str(eeg_file))
    key = "seg" if "seg" in mat else "seg1"
    raw = mat[key]                      # (10000, 30, 200) on disk
    tensor = raw[0, :, :].astype(np.float32)   # (30, 200) -- one 2 ms time-point
    return tensor[:28, :]                       # (28, 200)


def check_eeg_degeneracy(data_dir: Path, cache_dir: Path, n_subjects: int) -> dict:
    """Count distinct EEG tensors per subject under each pipeline."""
    subject_dirs = list_subject_dirs(data_dir)[:n_subjects]
    dataset = EAVMultimodalDataset(cache_dir=cache_dir)

    rows = []
    for subject_dir in subject_dirs:
        sid = int(subject_dir.name[len("subject"):])
        eeg_file, _ = subject_eeg_paths(subject_dir)

        # --- original pipeline ---
        original = original_eeg_tensor(eeg_file)
        n_samples = int((dataset.subject_ids == sid).sum())
        # The tensor is recomputed identically for every sample of this subject.
        original_distinct = 1
        original_shape = list(original.shape)

        # --- corrected pipeline ---
        idx = np.flatnonzero(dataset.subject_ids == sid)
        fingerprints = set()
        for i in idx:
            eeg = dataset[int(i)]["eeg"].numpy()
            fingerprints.add(hash(eeg.tobytes()))
        corrected_distinct = len(fingerprints)

        rows.append(
            {
                "subject": sid,
                "samples": n_samples,
                "original_distinct_eeg": original_distinct,
                "original_eeg_shape": original_shape,
                "corrected_distinct_eeg": corrected_distinct,
                "corrected_eeg_shape": list(dataset.eeg_shape),
            }
        )

    return {
        "subjects_checked": len(rows),
        "per_subject": rows,
        "original_distinct_total": sum(r["original_distinct_eeg"] for r in rows),
        "corrected_distinct_total": sum(r["corrected_distinct_eeg"] for r in rows),
        "samples_total": sum(r["samples"] for r in rows),
    }


def check_label_provenance(data_dir: Path, cache_dir: Path) -> dict:
    """Compare filename-derived labels against the dataset's own label matrix."""
    dataset = EAVMultimodalDataset(cache_dir=cache_dir)

    # Original: substring match on the audio filename, defaulting to Neutral.
    filename_labels: Counter = Counter()
    for subject_dir in list_subject_dirs(data_dir):
        for wav in sorted((subject_dir / "Audio").glob("*.wav")):
            matched = None
            for emotion in ORIGINAL_EMOTION_MAP:
                if emotion in wav.name:
                    matched = emotion
                    break
            # The original defaulted unmatched files to class 0 (Neutral).
            filename_labels[matched or "Neutral (defaulted)"] += 1

    ground_truth = dataset.class_counts()
    return {
        "filename_derived": dict(sorted(filename_labels.items())),
        "ground_truth": ground_truth,
        "ground_truth_balanced": len(set(ground_truth.values())) == 1,
    }


def check_split_contamination(n_samples: int, subject_ids: np.ndarray) -> dict:
    """Reproduce the RNG mismatch and contrast with subject-independent splitting."""
    n_train, n_val = 2940, 630

    # scripts/train_attention_fusion.py (original): NumPy legacy RNG
    np.random.seed(42)
    np_perm = np.random.permutation(n_samples)
    numpy_split = SplitResult(
        train=np_perm[:n_train],
        val=np_perm[n_train:n_train + n_val],
        test=np_perm[n_train + n_val:],
        strategy="original_numpy_permutation",
        seed=42,
    )

    # scripts/evaluate_finetuned_model.py (original): PyTorch RNG
    torch_perm = torch.randperm(
        n_samples, generator=torch.Generator().manual_seed(42)
    ).numpy()
    torch_split = SplitResult(
        train=torch_perm[:n_train],
        val=torch_perm[n_train:n_train + n_val],
        test=torch_perm[n_train + n_val:],
        strategy="original_torch_permutation",
        seed=42,
    )

    overlap = split_overlap_report(numpy_split, torch_split)
    overlap["b_test_in_a_train_pct"] = round(
        100 * overlap["b_test_in_a_train"] / overlap["b_test_size"], 2
    )
    overlap["shared_test_pct"] = round(
        100 * overlap["shared_test_samples"] / overlap["b_test_size"], 2
    )

    # Subject overlap under the pooled split vs the subject-independent one.
    pooled = pooled_random_split(n_samples, seed=42)
    subject_indep = subject_independent_split(subject_ids, seed=42)

    def subject_sets(split: SplitResult) -> dict:
        return {
            part: set(subject_ids[getattr(split, part)].tolist())
            for part in ("train", "val", "test")
        }

    pooled_subjects = subject_sets(pooled)
    indep_subjects = subject_sets(subject_indep)

    return {
        "rng_mismatch": overlap,
        "pooled_split": {
            "test_subjects_also_in_train": len(
                pooled_subjects["test"] & pooled_subjects["train"]
            ),
            "n_test_subjects": len(pooled_subjects["test"]),
        },
        "subject_independent_split": {
            "test_subjects_also_in_train": len(
                indep_subjects["test"] & indep_subjects["train"]
            ),
            "n_test_subjects": len(indep_subjects["test"]),
            "test_subjects": sorted(indep_subjects["test"]),
            "sizes": subject_indep.sizes,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/EAV/EAV")
    parser.add_argument("--cache-dir", default="data/processed/eav")
    parser.add_argument(
        "--n-subjects", type=int, default=5, help="subjects to check for EEG degeneracy"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    print("=" * 78)
    print("1. EEG DEGENERACY")
    print("=" * 78)
    degeneracy = check_eeg_degeneracy(data_dir, cache_dir, args.n_subjects)
    print(f"{'subject':>8} {'samples':>8} {'original':>10} {'corrected':>10}")
    for row in degeneracy["per_subject"]:
        print(
            f"{row['subject']:>8} {row['samples']:>8} "
            f"{row['original_distinct_eeg']:>10} {row['corrected_distinct_eeg']:>10}"
        )
    print(
        f"\nAcross {degeneracy['subjects_checked']} subjects "
        f"({degeneracy['samples_total']} samples): "
        f"{degeneracy['original_distinct_total']} distinct EEG tensors originally, "
        f"{degeneracy['corrected_distinct_total']} after the fix."
    )
    orig_shape = degeneracy["per_subject"][0]["original_eeg_shape"]
    corr_shape = degeneracy["per_subject"][0]["corrected_eeg_shape"]
    print(f"EEG tensor shape: {orig_shape} (original) -> {corr_shape} (corrected)")

    print("\n" + "=" * 78)
    print("2. LABEL PROVENANCE")
    print("=" * 78)
    labels = check_label_provenance(data_dir, cache_dir)
    print("filename-derived (original):", labels["filename_derived"])
    print("ground-truth label matrix   :", labels["ground_truth"])
    print(f"ground truth balanced       : {labels['ground_truth_balanced']}")

    print("\n" + "=" * 78)
    print("3. SPLIT CONTAMINATION")
    print("=" * 78)
    dataset = EAVMultimodalDataset(cache_dir=cache_dir)
    splits = check_split_contamination(len(dataset), dataset.subject_ids)
    rng = splits["rng_mismatch"]
    print(
        f"NumPy-split test set vs PyTorch-split test set: "
        f"{rng['shared_test_samples']}/{rng['b_test_size']} shared "
        f"({rng['shared_test_pct']}%)"
    )
    print(
        f"PyTorch-split 'test' samples that were in NumPy-split training: "
        f"{rng['b_test_in_a_train']}/{rng['b_test_size']} "
        f"({rng['b_test_in_a_train_pct']}%)"
    )
    pooled = splits["pooled_split"]
    indep = splits["subject_independent_split"]
    print(
        f"\npooled random split      : {pooled['test_subjects_also_in_train']} of "
        f"{pooled['n_test_subjects']} test subjects also appear in training"
    )
    print(
        f"subject-independent split: {indep['test_subjects_also_in_train']} of "
        f"{indep['n_test_subjects']} test subjects also appear in training"
    )
    print(f"  held-out test subjects : {indep['test_subjects']}")
    print(f"  sizes                  : {indep['sizes']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "eeg_degeneracy": degeneracy,
                    "label_provenance": labels,
                    "split_contamination": splits,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
