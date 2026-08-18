"""Ground-truth label semantics for the EAV dataset.

Each EAV subject ships ``subject<N>_eeg_label.mat`` containing a ``(10, n_trials)``
one-hot matrix. The ten rows encode the cross product of five emotions and two
task conditions:

===  =========  =========
row  condition  emotion
===  =========  =========
0    Listening  Neutral
1    Speaking   Neutral
2    Listening  Sadness
3    Speaking   Sadness
4    Listening  Anger
5    Speaking   Anger
6    Listening  Happiness
7    Speaking   Happiness
8    Listening  Calmness
9    Speaking   Calmness
===  =========  =========

That is ``emotion = row // 2`` and ``condition = row % 2``.

This mapping was derived empirically by cross-referencing the label matrix
against the emotion and condition encoded in the Audio/Video filenames, and
verified to agree on all 8,400 media files across all 42 subjects
(``scripts/audit_eav_alignment.py``).

Note that the row order here (Neutral, Sadness, Anger, Happiness, Calmness) is
*not* the class-index order used by the models. ``EMOTION_TO_CLASS_INDEX`` maps
onto the project's established class indices so that previously reported
per-class results remain comparable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

N_LABEL_ROWS = 10

#: Emotion carried by each row of the ``label`` matrix, in row order.
LABEL_ROW_TO_EMOTION = [
    "Neutral",
    "Neutral",
    "Sadness",
    "Sadness",
    "Anger",
    "Anger",
    "Happiness",
    "Happiness",
    "Calmness",
    "Calmness",
]

#: Task condition carried by each row of the ``label`` matrix, in row order.
LABEL_ROW_TO_CONDITION = [
    "Listening",
    "Speaking",
    "Listening",
    "Speaking",
    "Listening",
    "Speaking",
    "Listening",
    "Speaking",
    "Listening",
    "Speaking",
]

#: Model class indices. Preserved from the original project code so that
#: per-class numbers stay comparable across the pre- and post-fix experiments.
EMOTION_TO_CLASS_INDEX = {
    "Neutral": 0,
    "Anger": 1,
    "Calmness": 2,
    "Sadness": 3,
    "Happiness": 4,
}

CLASS_INDEX_TO_EMOTION = {v: k for k, v in EMOTION_TO_CLASS_INDEX.items()}

#: Emotion names ordered by class index -- suitable for confusion-matrix axes.
EMOTION_NAMES = [CLASS_INDEX_TO_EMOTION[i] for i in range(len(CLASS_INDEX_TO_EMOTION))]


class LabelDecodeError(ValueError):
    """Raised when a label matrix does not have the expected one-hot structure."""


def decode_label_matrix(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a ``(10, n_trials)`` one-hot matrix into emotion and condition names.

    Parameters
    ----------
    label:
        The ``label`` array loaded from ``subject<N>_eeg_label.mat``.

    Returns
    -------
    (emotions, conditions)
        Two string arrays of length ``n_trials``.

    Raises
    ------
    LabelDecodeError
        If the matrix has the wrong number of rows, or if any trial does not
        have exactly one active row. Silently defaulting such trials to a class
        would fabricate ground truth, so this is always an error.
    """
    if label.ndim != 2 or label.shape[0] != N_LABEL_ROWS:
        raise LabelDecodeError(
            f"expected a ({N_LABEL_ROWS}, n_trials) label matrix, got {label.shape}"
        )

    active = label.astype(bool)
    counts = active.sum(axis=0)
    bad = np.flatnonzero(counts != 1)
    if bad.size:
        raise LabelDecodeError(
            f"{bad.size} of {label.shape[1]} trials are not one-hot "
            f"(first offending trial index: {int(bad[0])})"
        )

    rows = active.argmax(axis=0)
    emotions = np.array([LABEL_ROW_TO_EMOTION[r] for r in rows])
    conditions = np.array([LABEL_ROW_TO_CONDITION[r] for r in rows])
    return emotions, conditions


def emotion_class_indices(label: np.ndarray) -> np.ndarray:
    """Return the model class index (0-4) for every trial in a label matrix."""
    emotions, _ = decode_label_matrix(label)
    return np.array([EMOTION_TO_CLASS_INDEX[e] for e in emotions], dtype=np.int64)
