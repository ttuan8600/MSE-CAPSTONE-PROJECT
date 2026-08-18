"""Tests for dataset splitting.

These guard the two defects that invalidated the original results: splits that
leak samples between partitions, and splits that put the same subject on both
sides of the train/test boundary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.data.splits import (
    SplitResult,
    leave_one_subject_out,
    make_split,
    pooled_random_split,
    split_overlap_report,
    subject_independent_split,
    subject_kfold,
)


@pytest.fixture
def subject_ids():
    """42 subjects x 100 samples, matching the real EAV multimodal set."""
    return np.repeat(np.arange(1, 43), 100)


def test_subject_independent_partitions_are_disjoint(subject_ids):
    split = subject_independent_split(subject_ids)
    assert np.intersect1d(split.train, split.val).size == 0
    assert np.intersect1d(split.train, split.test).size == 0
    assert np.intersect1d(split.val, split.test).size == 0


def test_subject_independent_covers_every_sample(subject_ids):
    split = subject_independent_split(subject_ids)
    total = split.train.size + split.val.size + split.test.size
    assert total == len(subject_ids)


def test_no_subject_appears_in_two_partitions(subject_ids):
    """The property the whole module exists to guarantee."""
    split = subject_independent_split(subject_ids)
    train_s = set(subject_ids[split.train])
    val_s = set(subject_ids[split.val])
    test_s = set(subject_ids[split.test])

    assert train_s & val_s == set()
    assert train_s & test_s == set()
    assert val_s & test_s == set()


def test_subject_counts_are_respected(subject_ids):
    split = subject_independent_split(subject_ids, val_subjects=6, test_subjects=8)
    assert len(set(subject_ids[split.val])) == 6
    assert len(set(subject_ids[split.test])) == 8
    assert len(set(subject_ids[split.train])) == 42 - 6 - 8


def test_split_is_reproducible(subject_ids):
    a = subject_independent_split(subject_ids, seed=7)
    b = subject_independent_split(subject_ids, seed=7)
    np.testing.assert_array_equal(a.test, b.test)


def test_different_seeds_give_different_splits(subject_ids):
    a = subject_independent_split(subject_ids, seed=1)
    b = subject_independent_split(subject_ids, seed=2)
    assert set(a.subjects["test"]) != set(b.subjects["test"])


def test_rejects_holding_out_every_subject():
    ids = np.repeat(np.arange(5), 10)
    with pytest.raises(ValueError, match="cannot hold out"):
        subject_independent_split(ids, val_subjects=2, test_subjects=3)


def test_pooled_split_does_leak_subjects(subject_ids):
    """Documents why pooled_random must not be used to report generalisation."""
    split = pooled_random_split(len(subject_ids))
    train_s = set(subject_ids[split.train])
    test_s = set(subject_ids[split.test])
    # Every test subject was also trained on.
    assert test_s <= train_s
    assert len(test_s) == 42


def test_split_result_rejects_overlapping_indices():
    with pytest.raises(ValueError, match="leaks"):
        SplitResult(
            train=np.array([0, 1, 2]),
            val=np.array([2, 3]),          # 2 appears twice
            test=np.array([4]),
            strategy="broken",
            seed=0,
        )


def test_reproduces_original_rng_contamination():
    """The historical bug: same seed, two RNGs, unrelated permutations.

    ``train_attention_fusion.py`` used ``np.random.seed(42)`` +
    ``np.random.permutation``; ``evaluate_finetuned_model.py`` used
    ``torch.randperm`` with ``manual_seed(42)``. This reproduces the exact
    overlap figures quoted in docs/DATA_CORRECTIONS.md.
    """
    n, n_train, n_val = 4200, 2940, 630

    np.random.seed(42)
    np_perm = np.random.permutation(n)
    numpy_split = SplitResult(
        train=np_perm[:n_train],
        val=np_perm[n_train:n_train + n_val],
        test=np_perm[n_train + n_val:],
        strategy="numpy",
        seed=42,
    )

    torch_perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    torch_split = SplitResult(
        train=torch_perm[:n_train],
        val=torch_perm[n_train:n_train + n_val],
        test=torch_perm[n_train + n_val:],
        strategy="torch",
        seed=42,
    )

    report = split_overlap_report(numpy_split, torch_split)
    assert report["shared_test_samples"] == 108
    assert report["b_test_in_a_train"] == 435
    assert report["b_test_size"] == 630


def test_make_split_dispatch(subject_ids):
    a = make_split("subject_independent", subject_ids, seed=3)
    assert a.strategy == "subject_independent"
    b = make_split("pooled_random", subject_ids, seed=3)
    assert b.strategy == "pooled_random"
    with pytest.raises(ValueError, match="unknown split strategy"):
        make_split("nonsense", subject_ids)


def test_leave_one_subject_out_yields_one_fold_per_subject():
    ids = np.repeat(np.arange(1, 11), 10)
    folds = leave_one_subject_out(ids, val_subjects=2)
    assert len(folds) == 10
    for fold in folds:
        assert len(fold.subjects["test"]) == 1
        held = fold.subjects["test"][0]
        assert held not in fold.subjects["train"]
        assert held not in fold.subjects["val"]
        assert set(ids[fold.test]) == {held}


def test_kfold_tests_every_subject_exactly_once(subject_ids):
    """The property that makes pooled cross-validated accuracy meaningful."""
    folds = subject_kfold(subject_ids, n_folds=7)
    tested = [s for fold in folds for s in fold.subjects["test"]]
    assert sorted(tested) == sorted(set(subject_ids.tolist()))
    assert len(tested) == len(set(tested))


def test_kfold_covers_every_sample_exactly_once(subject_ids):
    folds = subject_kfold(subject_ids, n_folds=7)
    covered = np.concatenate([fold.test for fold in folds])
    np.testing.assert_array_equal(np.sort(covered), np.arange(len(subject_ids)))


def test_kfold_partitions_are_subject_disjoint(subject_ids):
    for fold in subject_kfold(subject_ids, n_folds=7):
        train_s = set(subject_ids[fold.train])
        val_s = set(subject_ids[fold.val])
        test_s = set(subject_ids[fold.test])
        assert train_s & test_s == set()
        assert val_s & test_s == set()
        assert train_s & val_s == set()


def test_kfold_holds_out_the_requested_validation_size(subject_ids):
    for fold in subject_kfold(subject_ids, n_folds=7, val_subjects=6):
        assert len(fold.subjects["val"]) == 6


def test_kfold_is_reproducible(subject_ids):
    a = subject_kfold(subject_ids, n_folds=7, seed=11)
    b = subject_kfold(subject_ids, n_folds=7, seed=11)
    for fa, fb in zip(a, b):
        assert fa.subjects["test"] == fb.subjects["test"]
        assert fa.subjects["val"] == fb.subjects["val"]


def test_kfold_folds_differ_from_each_other(subject_ids):
    folds = subject_kfold(subject_ids, n_folds=7)
    val_sets = [tuple(sorted(f.subjects["val"])) for f in folds]
    assert len(set(val_sets)) > 1


def test_kfold_handles_uneven_division():
    """40 subjects into 7 folds: sizes differ by at most one."""
    ids = np.repeat(np.arange(1, 41), 10)
    folds = subject_kfold(ids, n_folds=7, val_subjects=4)
    sizes = [len(f.subjects["test"]) for f in folds]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 40


def test_kfold_rejects_too_many_folds():
    ids = np.repeat(np.arange(1, 6), 10)
    with pytest.raises(ValueError, match="cannot build"):
        subject_kfold(ids, n_folds=9)


def test_kfold_rejects_degenerate_fold_count(subject_ids):
    with pytest.raises(ValueError, match="at least 2"):
        subject_kfold(subject_ids, n_folds=1)


def test_kfold_rejects_oversized_validation_holdout():
    ids = np.repeat(np.arange(1, 11), 10)
    with pytest.raises(ValueError, match="cannot hold out"):
        subject_kfold(ids, n_folds=2, val_subjects=9)


def test_split_serialises_for_the_results_artifact(subject_ids):
    split = subject_independent_split(subject_ids)
    payload = split.to_dict()
    assert payload["strategy"] == "subject_independent"
    assert payload["sizes"]["test"] == 800
    assert len(payload["subjects"]["test"]) == 8
