"""Tests for EAV ground-truth label decoding.

The original pipeline never read the ``label`` matrix; it inferred emotions by
substring-matching audio filenames and defaulted anything unmatched to Neutral.
These tests pin the decoding of the real labels and ensure an ambiguous matrix is
an error rather than a silent default.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.preprocessing.eav_labels import (
    EMOTION_NAMES,
    EMOTION_TO_CLASS_INDEX,
    LABEL_ROW_TO_CONDITION,
    LABEL_ROW_TO_EMOTION,
    LabelDecodeError,
    N_LABEL_ROWS,
    decode_label_matrix,
    emotion_class_indices,
)


def one_hot(rows, n_rows=N_LABEL_ROWS):
    """Build a ``(n_rows, n_trials)`` one-hot matrix from a list of row indices."""
    matrix = np.zeros((n_rows, len(rows)), dtype=np.uint8)
    for col, row in enumerate(rows):
        matrix[row, col] = 1
    return matrix


def test_row_semantics_are_emotion_major_condition_minor():
    """row // 2 selects the emotion, row % 2 selects Listening/Speaking."""
    for row in range(N_LABEL_ROWS):
        expected_condition = "Listening" if row % 2 == 0 else "Speaking"
        assert LABEL_ROW_TO_CONDITION[row] == expected_condition
    # Each emotion occupies an adjacent Listening/Speaking pair.
    for row in range(0, N_LABEL_ROWS, 2):
        assert LABEL_ROW_TO_EMOTION[row] == LABEL_ROW_TO_EMOTION[row + 1]


def test_decode_returns_expected_names():
    label = one_hot([0, 1, 4, 5, 9])
    emotions, conditions = decode_label_matrix(label)
    assert list(emotions) == ["Neutral", "Neutral", "Anger", "Anger", "Calmness"]
    assert list(conditions) == [
        "Listening", "Speaking", "Listening", "Speaking", "Speaking",
    ]


def test_emotion_class_indices_uses_project_class_order():
    label = one_hot([0, 4, 8, 2, 6])   # Neutral, Anger, Calmness, Sadness, Happiness
    assert list(emotion_class_indices(label)) == [0, 1, 2, 3, 4]


def test_class_index_map_is_a_bijection_over_five_emotions():
    assert sorted(EMOTION_TO_CLASS_INDEX.values()) == [0, 1, 2, 3, 4]
    assert len(EMOTION_NAMES) == 5
    for name, idx in EMOTION_TO_CLASS_INDEX.items():
        assert EMOTION_NAMES[idx] == name


def test_rejects_wrong_row_count():
    with pytest.raises(LabelDecodeError, match="expected a"):
        decode_label_matrix(np.zeros((5, 20), dtype=np.uint8))


def test_rejects_all_zero_column():
    """A trial with no active row must raise, not default to Neutral."""
    label = one_hot([0, 1, 2])
    label[:, 1] = 0
    with pytest.raises(LabelDecodeError, match="not one-hot"):
        decode_label_matrix(label)


def test_rejects_multi_hot_column():
    label = one_hot([0, 1, 2])
    label[3, 1] = 1     # column 1 now has two active rows
    with pytest.raises(LabelDecodeError, match="not one-hot"):
        decode_label_matrix(label)
