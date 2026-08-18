"""Audit the EAV dataset for structural consistency and EEG/audio trial alignment.

This script exists because the original data loader mis-read the EAV EEG arrays.
It records, per subject, the facts that the corrected loader depends on:

* the ``seg`` array is ``(time, channels, trials)`` -- NOT ``(segments, channels, time)``
* the ``label`` array is ``(10, trials)`` one-hot over ``emotion x condition``
* the 3-digit prefix of an Audio/Video filename is the 1-based trial index into
  those arrays
* the emotion encoded in the filename agrees with the emotion in ``label``

Run from the project root::

    python scripts/audit_eav_alignment.py --json outputs/eav_alignment_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.eav_io import (
    EAVDataError,
    list_subject_dirs,
    load_subject_labels,
    load_subject_segments,
    parse_media_filename,
    subject_eeg_paths,
)
from src.preprocessing.eav_labels import (
    LABEL_ROW_TO_CONDITION,
    LABEL_ROW_TO_EMOTION,
    N_LABEL_ROWS,
    decode_label_matrix,
)


def audit_subject(subject_dir: Path) -> dict:
    """Collect structural facts and alignment mismatches for one subject."""
    name = subject_dir.name
    result: dict = {"subject": name, "errors": [], "warnings": []}

    eeg_file, label_file = subject_eeg_paths(subject_dir)

    try:
        # Returned trial-major: (trials, channels, time)
        segments = load_subject_segments(eeg_file)
        label = load_subject_labels(label_file)
    except EAVDataError as exc:
        result["errors"].append(str(exc))
        return result

    n_trials, n_channels, n_time = segments.shape
    result["seg_shape"] = [n_trials, n_channels, n_time]
    result["label_shape"] = list(label.shape)
    result["dtype"] = str(segments.dtype)
    result["n_time"] = int(n_time)
    result["n_channels"] = int(n_channels)
    result["n_trials"] = int(n_trials)

    if label.shape != (N_LABEL_ROWS, n_trials):
        result["errors"].append(
            f"label shape {label.shape} does not match ({N_LABEL_ROWS}, {n_trials})"
        )
        return result

    # Every trial must carry exactly one active label row.
    active = label.astype(bool).sum(axis=0)
    bad = np.flatnonzero(active != 1)
    if bad.size:
        result["errors"].append(
            f"{bad.size} trials are not one-hot (first: trial {int(bad[0]) + 1})"
        )

    try:
        emotions, conditions = decode_label_matrix(label)
    except ValueError as exc:
        result["errors"].append(str(exc))
        return result

    result["emotion_counts"] = {
        k: int(v) for k, v in sorted(Counter(emotions.tolist()).items())
    }
    result["condition_counts"] = {
        k: int(v) for k, v in sorted(Counter(conditions.tolist()).items())
    }

    # Cross-check filenames against the label matrix.
    checked = mismatched = 0
    for media_dir, suffix in ((subject_dir / "Audio", "*.wav"), (subject_dir / "Video", "*.mp4")):
        if not media_dir.exists():
            result["warnings"].append(f"missing {media_dir.name}/ directory")
            continue
        files = sorted(media_dir.glob(suffix))
        result[f"n_{media_dir.name.lower()}"] = len(files)
        for media in files:
            try:
                idx, file_cond, file_emotion = parse_media_filename(media.name)
            except EAVDataError:
                result["warnings"].append(f"unparseable filename {media.name}")
                continue
            if not 1 <= idx <= n_trials:
                result["errors"].append(f"{media.name}: index {idx} out of range")
                continue
            checked += 1
            if emotions[idx - 1] != file_emotion or conditions[idx - 1] != file_cond:
                mismatched += 1
                if mismatched <= 3:
                    result["errors"].append(
                        f"{media.name}: filename says {file_cond}/{file_emotion}, "
                        f"label says {conditions[idx - 1]}/{emotions[idx - 1]}"
                    )

    result["filenames_checked"] = checked
    result["filename_label_mismatches"] = mismatched

    # Audio should exist for exactly the Speaking trials.
    speaking = int((conditions == "Speaking").sum())
    if result.get("n_audio") is not None and result["n_audio"] != speaking:
        result["warnings"].append(
            f"{result['n_audio']} audio files but {speaking} Speaking trials"
        )

    # Flat (zero-variance) trials indicate dead channels or truncated recordings.
    variances = segments.var(axis=2)  # (trials, channels)
    flat = int((variances < 1e-12).sum())
    result["flat_channel_trials"] = flat
    if flat:
        result["warnings"].append(f"{flat} channel/trial pairs have zero variance")

    # Non-finite values are already fatal in load_subject_segments().
    result["nan_values"] = 0

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/EAV/EAV")
    parser.add_argument("--json", default=None, help="write the full report here")
    args = parser.parse_args()

    root = Path(args.data_dir)
    try:
        subject_dirs = list_subject_dirs(root)
    except EAVDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Auditing {len(subject_dirs)} subjects under {root}\n")
    print(f"Label row semantics: {N_LABEL_ROWS} rows = 5 emotions x 2 conditions")
    for row in range(N_LABEL_ROWS):
        print(f"  row {row}: {LABEL_ROW_TO_CONDITION[row]:<9} {LABEL_ROW_TO_EMOTION[row]}")
    print()

    reports = [audit_subject(d) for d in subject_dirs]

    shapes = Counter(tuple(r["seg_shape"]) for r in reports if "seg_shape" in r)
    print("seg shapes observed:")
    for shape, count in shapes.most_common():
        print(f"  {shape}: {count} subject(s)")

    total_checked = sum(r.get("filenames_checked", 0) for r in reports)
    total_mismatch = sum(r.get("filename_label_mismatches", 0) for r in reports)
    print(f"\nfilename/label cross-check: {total_checked} files, {total_mismatch} mismatches")

    with_errors = [r for r in reports if r["errors"]]
    with_warnings = [r for r in reports if r["warnings"]]
    print(f"subjects with errors:   {len(with_errors)}")
    print(f"subjects with warnings: {len(with_warnings)}")

    for r in with_errors:
        print(f"\n  [{r['subject']}] errors:")
        for e in r["errors"]:
            print(f"    - {e}")
    for r in with_warnings:
        print(f"\n  [{r['subject']}] warnings:")
        for w in r["warnings"]:
            print(f"    - {w}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_dir": str(root),
            "n_subjects": len(subject_dirs),
            "label_row_to_emotion": LABEL_ROW_TO_EMOTION,
            "label_row_to_condition": LABEL_ROW_TO_CONDITION,
            "seg_shapes": {str(list(k)): v for k, v in shapes.items()},
            "filenames_checked": total_checked,
            "filename_label_mismatches": total_mismatch,
            "subjects": reports,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    return 1 if with_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
