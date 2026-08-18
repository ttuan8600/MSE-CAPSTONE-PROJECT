"""Dataset splitting for the EAV experiments.

This module is the **single** place a train/val/test split is defined. Two
defects motivated it:

1. *Two incompatible split implementations.* ``train_attention_fusion.py`` used
   ``np.random.seed(42)`` + ``np.random.permutation``, while
   ``evaluate_finetuned_model.py`` and ``finetune_attention_fusion.py`` used
   ``torch.randperm`` with a ``manual_seed(42)`` generator. The same seed value
   in two different RNGs produces unrelated permutations, so the "same split as
   training" comment in the evaluation script was false: 69% of its test set had
   been trained on.

2. *Subject-dependent splitting.* Both implementations permuted pooled samples,
   so every subject appeared in train, validation **and** test. For EEG the
   subject-specific component of the signal is large, and a model can reach a
   high pooled-split score by identifying the subject rather than the emotion.
   That is not a measurement of generalisation to a new person, which is the
   claim an affective-computing system needs to support.

``subject_independent_split`` is therefore the default, and every script imports
it from here. ``pooled_random_split`` is retained *only* so the inflated
subject-dependent number can be reproduced and reported side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_SEED = 42


@dataclass
class SplitResult:
    """Index arrays for one train/validation/test partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str
    seed: int
    subjects: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlaps = [
            ("train", "val", np.intersect1d(self.train, self.val)),
            ("train", "test", np.intersect1d(self.train, self.test)),
            ("val", "test", np.intersect1d(self.val, self.test)),
        ]
        for a, b, shared in overlaps:
            if shared.size:
                raise ValueError(
                    f"{self.strategy} split leaks {shared.size} sample(s) between "
                    f"{a} and {b}"
                )

    @property
    def sizes(self) -> Dict[str, int]:
        return {
            "train": int(self.train.size),
            "val": int(self.val.size),
            "test": int(self.test.size),
        }

    def describe(self) -> str:
        lines = [
            f"split strategy : {self.strategy}",
            f"seed           : {self.seed}",
            f"sizes          : train={self.train.size} "
            f"val={self.val.size} test={self.test.size}",
        ]
        if self.subjects:
            for part in ("train", "val", "test"):
                ids = self.subjects.get(part, [])
                lines.append(f"{part:<15}: {len(ids)} subjects {sorted(ids)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "sizes": self.sizes,
            "subjects": {k: sorted(v) for k, v in self.subjects.items()},
        }


def subject_independent_split(
    subject_ids: Sequence[int] | np.ndarray,
    val_subjects: int = 6,
    test_subjects: int = 8,
    seed: int = DEFAULT_SEED,
) -> SplitResult:
    """Partition samples so that no subject appears in more than one partition.

    Subjects -- not samples -- are shuffled and dealt into test, validation and
    training pools. Every sample inherits its subject's partition, so the test
    score answers "how well does this model work on a person it has never seen?"

    Parameters
    ----------
    subject_ids:
        Subject id for each sample, in dataset order.
    val_subjects, test_subjects:
        Number of whole subjects held out for validation and test.
    seed:
        Seed for the subject shuffle. A single ``np.random.default_rng`` is used
        so the split is reproducible and RNG-agnostic.
    """
    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1:
        raise ValueError(f"subject_ids must be 1-D, got shape {subject_ids.shape}")

    unique = np.unique(subject_ids)
    needed = val_subjects + test_subjects
    if needed >= unique.size:
        raise ValueError(
            f"cannot hold out {needed} subjects: only {unique.size} available"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)

    test_ids = shuffled[:test_subjects]
    val_ids = shuffled[test_subjects:test_subjects + val_subjects]
    train_ids = shuffled[test_subjects + val_subjects:]

    def indices_for(ids: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(subject_ids, ids))

    return SplitResult(
        train=indices_for(train_ids),
        val=indices_for(val_ids),
        test=indices_for(test_ids),
        strategy="subject_independent",
        seed=seed,
        subjects={
            "train": [int(i) for i in train_ids],
            "val": [int(i) for i in val_ids],
            "test": [int(i) for i in test_ids],
        },
    )


def pooled_random_split(
    n_samples: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> SplitResult:
    """Shuffle samples without regard to subject (the original, inflated split).

    Retained so the subject-dependent figure can be reproduced deliberately and
    quoted alongside the subject-independent one. **Do not use this to report
    generalisation performance** -- the same subject will appear in training and
    test, so the score is optimistic.
    """
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(round((1.0 - val_fraction - test_fraction) * n_samples))
    n_val = int(round(val_fraction * n_samples))

    return SplitResult(
        train=perm[:n_train],
        val=perm[n_train:n_train + n_val],
        test=perm[n_train + n_val:],
        strategy="pooled_random",
        seed=seed,
    )


def leave_one_subject_out(
    subject_ids: Sequence[int] | np.ndarray,
    val_subjects: int = 4,
    seed: int = DEFAULT_SEED,
) -> List[SplitResult]:
    """Yield one :class:`SplitResult` per subject, that subject being the test set.

    This is the strictest protocol reported in the affective-computing
    literature. It costs one training run per subject, so it is used for the
    final validation of the model of record rather than for routine iteration.
    """
    subject_ids = np.asarray(subject_ids)
    unique = np.unique(subject_ids)
    rng = np.random.default_rng(seed)

    folds: List[SplitResult] = []
    for held_out in unique:
        remaining = unique[unique != held_out]
        val_ids = rng.permutation(remaining)[:val_subjects]
        train_ids = np.setdiff1d(remaining, val_ids)
        folds.append(
            SplitResult(
                train=np.flatnonzero(np.isin(subject_ids, train_ids)),
                val=np.flatnonzero(np.isin(subject_ids, val_ids)),
                test=np.flatnonzero(subject_ids == held_out),
                strategy="leave_one_subject_out",
                seed=seed,
                subjects={
                    "train": [int(i) for i in train_ids],
                    "val": [int(i) for i in val_ids],
                    "test": [int(held_out)],
                },
            )
        )
    return folds


def subject_kfold(
    subject_ids: Sequence[int] | np.ndarray,
    n_folds: int = 7,
    val_subjects: int = 6,
    seed: int = DEFAULT_SEED,
) -> List[SplitResult]:
    """Partition subjects into ``n_folds`` disjoint test groups.

    Every subject appears in exactly one fold's test partition, so aggregating
    predictions across folds yields one held-out prediction for every sample in
    the dataset. That is the estimate a single 8-subject test partition cannot
    provide: with six validation and eight test subjects, this project measured
    the *ranking* of two models reversing between the two partitions, which means
    neither could rank them reliably.

    Cheaper than leave-one-subject-out (7 runs rather than 42) while still giving
    every subject a turn in the test set.

    Parameters
    ----------
    subject_ids:
        Subject id per sample, in dataset order.
    n_folds:
        Number of folds. Subjects are dealt round-robin after shuffling, so fold
        sizes differ by at most one when the count does not divide evenly.
    val_subjects:
        Subjects held out for validation within each fold, drawn from the
        non-test subjects. Model selection uses only these.
    seed:
        Seed for the subject shuffle.
    """
    subject_ids = np.asarray(subject_ids)
    unique = np.unique(subject_ids)

    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_folds > unique.size:
        raise ValueError(
            f"cannot build {n_folds} folds from {unique.size} subjects"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    fold_subjects = [shuffled[i::n_folds] for i in range(n_folds)]

    folds: List[SplitResult] = []
    for fold_index, test_ids in enumerate(fold_subjects):
        remaining = np.setdiff1d(unique, test_ids)
        if val_subjects >= remaining.size:
            raise ValueError(
                f"fold {fold_index}: cannot hold out {val_subjects} validation "
                f"subjects from {remaining.size} remaining"
            )
        # Deterministic per fold, but a different draw for each.
        fold_rng = np.random.default_rng(seed + 1000 + fold_index)
        val_ids = fold_rng.permutation(remaining)[:val_subjects]
        train_ids = np.setdiff1d(remaining, val_ids)

        folds.append(
            SplitResult(
                train=np.flatnonzero(np.isin(subject_ids, train_ids)),
                val=np.flatnonzero(np.isin(subject_ids, val_ids)),
                test=np.flatnonzero(np.isin(subject_ids, test_ids)),
                strategy=f"subject_kfold[{fold_index + 1}/{n_folds}]",
                seed=seed,
                subjects={
                    "train": [int(i) for i in train_ids],
                    "val": [int(i) for i in val_ids],
                    "test": [int(i) for i in test_ids],
                },
            )
        )
    return folds


def make_split(
    strategy: str,
    subject_ids: Sequence[int] | np.ndarray,
    seed: int = DEFAULT_SEED,
    **kwargs,
) -> SplitResult:
    """Dispatch to a split strategy by name. Used by the training CLIs."""
    if strategy == "subject_independent":
        return subject_independent_split(subject_ids, seed=seed, **kwargs)
    if strategy == "pooled_random":
        return pooled_random_split(len(subject_ids), seed=seed, **kwargs)
    raise ValueError(
        f"unknown split strategy {strategy!r}; expected 'subject_independent' "
        f"or 'pooled_random'"
    )


def split_overlap_report(a: SplitResult, b: SplitResult) -> Dict[str, int]:
    """Quantify how far two splits disagree.

    Used to document the original contamination: passing the NumPy-permutation
    split and the PyTorch-permutation split shows how much of one's test set the
    other had trained on.
    """
    return {
        "shared_test_samples": int(np.intersect1d(a.test, b.test).size),
        "b_test_in_a_train": int(np.intersect1d(b.test, a.train).size),
        "a_test_in_b_train": int(np.intersect1d(a.test, b.train).size),
        "a_test_size": int(a.test.size),
        "b_test_size": int(b.test.size),
    }
