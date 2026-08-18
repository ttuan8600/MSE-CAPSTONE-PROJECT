"""Build the EAV preprocessing cache used by ``EAVMultimodalDataset``.

Reads the raw EAV MATLAB/WAV files once, and writes per-subject memory-mappable
arrays plus a manifest. Every downstream training and evaluation script reads
this cache, so the expensive decode happens exactly once.

What is produced, per subject::

    data/processed/eav/subject<N>_eeg.npy    (200, 30, 2500) float32  - all trials
    data/processed/eav/subject<N>_mfcc.npy   (100, 13, 2101) float32  - Speaking only
    data/processed/eav/manifest.json                                  - sample index

EEG is band-pass filtered to 0.5-45 Hz and decimated from 500 Hz to 125 Hz, so a
20 s trial becomes 2,500 samples. Audio is resampled to 16 kHz and converted to
13 MFCCs (25 ms window, 10 ms hop).

Labels come from ``subject<N>_eeg_label.mat`` -- the dataset's own ground truth --
not from parsing media filenames. ``scripts/audit_eav_alignment.py`` verifies the
two agree on all 12,600 media files.

Run from the project root::

    python scripts/preprocess_eav.py
    python scripts/preprocess_eav.py --subjects 1 2 3 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.eav_io import (
    EAV_SAMPLE_RATE,
    EAVDataError,
    list_subject_dirs,
    load_subject_labels,
    load_subject_segments,
    parse_media_filename,
    subject_eeg_paths,
)
from src.preprocessing.eav_labels import EMOTION_TO_CLASS_INDEX, decode_label_matrix
from src.preprocessing.features import (
    EEG_BANDS,
    differential_entropy,
    euclidean_alignment,
    log_mel_spectrogram,
)

# --- Preprocessing parameters (recorded into the manifest for provenance) ---
EEG_BANDPASS_LOW = 0.5      # Hz - removes electrode drift
EEG_BANDPASS_HIGH = 45.0    # Hz - below the 62.5 Hz Nyquist after decimation
EEG_DECIMATION = 4          # 500 Hz -> 125 Hz
EEG_TARGET_RATE = EAV_SAMPLE_RATE // EEG_DECIMATION

#: Differential-entropy features: 1 s non-overlapping windows over 5 bands.
DE_WINDOW_SECONDS = 1.0

AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MFCC = 13
AUDIO_N_FFT = 400           # 25 ms
AUDIO_HOP = 160             # 10 ms

#: Log-mel spectrogram parameters, used in place of MFCCs.
MEL_N_MELS = 64
MEL_N_FFT = 1024            # 64 ms
MEL_HOP = 256               # 16 ms
MEL_FMIN = 20.0

CACHE_VERSION = 3


def preprocess_eeg(segments: np.ndarray) -> np.ndarray:
    """Band-pass filter and decimate ``(trials, channels, time)`` EEG."""
    from scipy.signal import butter, decimate, sosfiltfilt

    sos = butter(
        4,
        [EEG_BANDPASS_LOW, EEG_BANDPASS_HIGH],
        btype="bandpass",
        fs=EAV_SAMPLE_RATE,
        output="sos",
    )
    # Zero-phase filtering along the time axis preserves temporal alignment.
    filtered = sosfiltfilt(sos, segments, axis=-1)
    reduced = decimate(filtered, EEG_DECIMATION, axis=-1, ftype="fir", zero_phase=True)
    return np.ascontiguousarray(reduced, dtype=np.float32)


def preprocess_audio(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a WAV file and return ``(mfcc, log_mel)`` feature matrices."""
    import librosa

    waveform, sr = librosa.load(str(wav_path), sr=AUDIO_SAMPLE_RATE)
    if waveform.size == 0:
        raise EAVDataError(f"{wav_path}: empty waveform")

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=AUDIO_N_MFCC,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP,
    )
    mel = log_mel_spectrogram(
        waveform,
        sample_rate=sr,
        n_mels=MEL_N_MELS,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP,
        fmin=MEL_FMIN,
    )
    if not np.isfinite(mfcc).all():
        raise EAVDataError(f"{wav_path}: non-finite MFCC values")
    if not np.isfinite(mel).all():
        raise EAVDataError(f"{wav_path}: non-finite log-mel values")

    return np.ascontiguousarray(mfcc, dtype=np.float32), mel


def preprocess_de(eeg: np.ndarray) -> np.ndarray:
    """Euclidean-align a subject's trials, then extract band-power features.

    Alignment is applied before feature extraction because it operates on the
    spatial covariance of the time-domain signal. It uses only this subject's own
    trials and no labels, so it is valid for a held-out test subject.
    """
    aligned = euclidean_alignment(eeg)
    features = [
        differential_entropy(
            trial,
            sample_rate=EEG_TARGET_RATE,
            bands=EEG_BANDS,
            window_seconds=DE_WINDOW_SECONDS,
        )
        for trial in aligned
    ]
    stacked = np.stack(features).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise EAVDataError("non-finite differential-entropy values")
    return stacked


def process_subject(subject_dir: Path, out_dir: Path, force: bool) -> dict:
    """Preprocess one subject and return its manifest entry."""
    name = subject_dir.name
    subject_id = int(name[len("subject"):])
    eeg_out = out_dir / f"{name}_eeg.npy"
    de_out = out_dir / f"{name}_de.npy"
    mfcc_out = out_dir / f"{name}_mfcc.npy"
    mel_out = out_dir / f"{name}_mel.npy"

    eeg_file, label_file = subject_eeg_paths(subject_dir)
    segments = load_subject_segments(eeg_file)          # (200, 30, 10000)
    label = load_subject_labels(label_file)
    emotions, conditions = decode_label_matrix(label)

    n_trials = segments.shape[0]
    if len(emotions) != n_trials:
        raise EAVDataError(
            f"{name}: {n_trials} EEG trials but {len(emotions)} labels"
        )

    if force or not eeg_out.exists():
        np.save(eeg_out, preprocess_eeg(segments))

    if force or not de_out.exists():
        # Read back the filtered/decimated signal so DE is computed on exactly
        # what the raw-signal model sees.
        np.save(de_out, preprocess_de(np.load(eeg_out)))

    # Index the audio files by their trial number, taken from the filename prefix.
    audio_dir = subject_dir / "Audio"
    if not audio_dir.is_dir():
        raise EAVDataError(f"{name}: missing Audio/ directory")

    audio_by_trial: dict[int, Path] = {}
    for wav in sorted(audio_dir.glob("*.wav")):
        idx, cond, emo = parse_media_filename(wav.name)
        if not 1 <= idx <= n_trials:
            raise EAVDataError(f"{name}: {wav.name} trial index {idx} out of range")
        # The audit script proves these agree; assert it here so a future data
        # refresh cannot silently reintroduce a misalignment.
        if conditions[idx - 1] != cond or emotions[idx - 1] != emo:
            raise EAVDataError(
                f"{name}: {wav.name} says {cond}/{emo} but ground-truth label for "
                f"trial {idx} is {conditions[idx - 1]}/{emotions[idx - 1]}"
            )
        if idx in audio_by_trial:
            raise EAVDataError(f"{name}: duplicate audio for trial {idx}")
        audio_by_trial[idx] = wav

    speaking_trials = sorted(audio_by_trial)
    if not speaking_trials:
        raise EAVDataError(f"{name}: no usable audio files")

    if force or not mfcc_out.exists() or not mel_out.exists():
        pairs = [preprocess_audio(audio_by_trial[t]) for t in speaking_trials]
        mfccs = [p[0] for p in pairs]
        mels = [p[1] for p in pairs]
        # All EAV clips are 21.0 s; crop to the shortest if that ever changes.
        for features, path in ((mfccs, mfcc_out), (mels, mel_out)):
            widths = {f.shape[1] for f in features}
            if len(widths) > 1:
                target = min(widths)
                features = [f[:, :target] for f in features]
            np.save(path, np.stack(features).astype(np.float32))

    samples = [
        {
            "subject_id": subject_id,
            "trial_index": int(trial),            # 1-based index into the EEG array
            "eeg_row": int(trial - 1),            # row in <name>_eeg.npy
            "mfcc_row": row,                      # row in <name>_mfcc.npy
            "emotion": str(emotions[trial - 1]),
            "condition": str(conditions[trial - 1]),
            "label": int(EMOTION_TO_CLASS_INDEX[emotions[trial - 1]]),
        }
        for row, trial in enumerate(speaking_trials)
    ]

    return {
        "subject_id": subject_id,
        "name": name,
        "eeg_file": eeg_out.name,
        "de_file": de_out.name,
        "mfcc_file": mfcc_out.name,
        "mel_file": mel_out.name,
        "n_eeg_trials": int(n_trials),
        "n_multimodal_samples": len(samples),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/EAV/EAV")
    parser.add_argument("--out-dir", default="data/processed/eav")
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--force", action="store_true", help="rebuild existing arrays")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        subject_dirs = list_subject_dirs(Path(args.data_dir), args.subjects)
    except EAVDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Preprocessing {len(subject_dirs)} subjects -> {out_dir}")
    print(
        f"  EEG:   {EAV_SAMPLE_RATE} Hz -> {EEG_TARGET_RATE} Hz, "
        f"band-pass {EEG_BANDPASS_LOW}-{EEG_BANDPASS_HIGH} Hz"
    )
    print(
        f"  Audio: {AUDIO_SAMPLE_RATE} Hz, {AUDIO_N_MFCC} MFCC, "
        f"n_fft={AUDIO_N_FFT}, hop={AUDIO_HOP}\n"
    )

    entries = []
    started = time.time()
    for i, subject_dir in enumerate(subject_dirs, 1):
        t0 = time.time()
        try:
            entry = process_subject(subject_dir, out_dir, args.force)
        except EAVDataError as exc:
            print(f"\nERROR in {subject_dir.name}: {exc}", file=sys.stderr)
            return 1
        entries.append(entry)
        print(
            f"  [{i:>2}/{len(subject_dirs)}] {entry['name']:<10} "
            f"{entry['n_eeg_trials']} trials, "
            f"{entry['n_multimodal_samples']} multimodal  "
            f"({time.time() - t0:.1f}s)"
        )

    total_samples = sum(e["n_multimodal_samples"] for e in entries)
    manifest = {
        "cache_version": CACHE_VERSION,
        "source": str(Path(args.data_dir)),
        "n_subjects": len(entries),
        "n_multimodal_samples": total_samples,
        "eeg": {
            "source_rate_hz": EAV_SAMPLE_RATE,
            "target_rate_hz": EEG_TARGET_RATE,
            "bandpass_hz": [EEG_BANDPASS_LOW, EEG_BANDPASS_HIGH],
            "decimation": EEG_DECIMATION,
        },
        "eeg_de": {
            "bands": [{"name": n, "low_hz": lo, "high_hz": hi} for n, lo, hi in EEG_BANDS],
            "window_seconds": DE_WINDOW_SECONDS,
            "euclidean_alignment": True,
            "note": (
                "Euclidean Alignment is unsupervised and computed per subject "
                "from that subject's own trials, so it is valid for held-out "
                "test subjects."
            ),
        },
        "audio": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE,
            "n_mfcc": AUDIO_N_MFCC,
            "n_fft": AUDIO_N_FFT,
            "hop_length": AUDIO_HOP,
        },
        "audio_mel": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE,
            "n_mels": MEL_N_MELS,
            "n_fft": MEL_N_FFT,
            "hop_length": MEL_HOP,
            "fmin_hz": MEL_FMIN,
            "scale": "decibels, top_db=80",
        },
        "emotion_to_class_index": EMOTION_TO_CLASS_INDEX,
        "subjects": entries,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone in {time.time() - started:.0f}s")
    print(f"  {len(entries)} subjects, {total_samples} multimodal samples")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
