"""Run the experiment matrix that substantiates (or refutes) the fusion claim.

Four runs, identical except for the variable under test:

===========================  ====================  ==================================
run                          purpose               what it answers
===========================  ====================  ==================================
multimodal / subject-indep   headline result       generalisation to unseen people
eeg / subject-indep          modality ablation     does EEG alone carry signal?
audio / subject-indep        modality ablation     does fusion beat audio alone?
multimodal / pooled-random   protocol ablation     how much did the old split inflate?
===========================  ====================  ==================================

The two modality ablations are what make a multimodal claim defensible: fusion is
only a contribution if it beats both single-modality models trained identically.
The protocol ablation quantifies the optimism of the original pooled split.

Run from the project root::

    python scripts/run_ablations.py
    python scripts/run_ablations.py --epochs 10          # quick pass
    python scripts/run_ablations.py --only multimodal_subject_independent
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

#: The baseline matrix: raw EEG time series and 13 MFCCs, 40 epochs.
BASELINE_RUNS = [
    {
        "name": "multimodal_subject_independent",
        "modality": "multimodal",
        "split": "subject_independent",
        "purpose": "Headline result: cross-modal fusion, generalising to unseen subjects",
    },
    {
        "name": "eeg_subject_independent",
        "modality": "eeg",
        "split": "subject_independent",
        "purpose": "Ablation: EEG alone, to show the modality carries emotion signal",
    },
    {
        "name": "audio_subject_independent",
        "modality": "audio",
        "split": "subject_independent",
        "purpose": "Ablation: audio alone, the bar fusion must clear",
    },
    {
        "name": "multimodal_pooled_random",
        "modality": "multimodal",
        "split": "pooled_random",
        "purpose": "Protocol ablation: reproduces the original subject-dependent split",
    },
]

#: The improved matrix. Each change responds to a measured failure of the
#: baseline rather than being a speculative addition:
#:
#: * EEG moves to Euclidean-aligned band-power features because the raw-signal
#:   encoder memorised its training subjects (99.75% train / 42.17% val).
#: * Audio moves to a 64-band log-mel spectrogram with SpecAugment because 13
#:   MFCCs on a 62K-parameter encoder were carrying the entire system.
#: * 100 epochs throughout, because the baseline audio model had not converged
#:   at 40 and the resulting budget mismatch produced a spurious fusion margin.
IMPROVED_RUNS = [
    {
        "name": "eeg_de_subject_independent",
        "modality": "eeg",
        "split": "subject_independent",
        "eeg_features": "de",
        "purpose": "EEG with Euclidean alignment + band-power features",
    },
    {
        "name": "audio_mel_subject_independent",
        "modality": "audio",
        "split": "subject_independent",
        "audio_features": "mel",
        "specaugment": True,
        "purpose": "Audio with log-mel spectrogram + SpecAugment",
    },
    {
        "name": "multimodal_improved_subject_independent",
        "modality": "multimodal",
        "split": "subject_independent",
        "eeg_features": "de",
        "audio_features": "mel",
        "specaugment": True,
        "purpose": "Fusion of both improved representations",
    },
]

RUN_SETS = {"baseline": BASELINE_RUNS, "improved": IMPROVED_RUNS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--set",
        choices=sorted(RUN_SETS),
        default="baseline",
        help="which experiment matrix to run",
    )
    parser.add_argument("--only", nargs="*", default=None, help="run only these names")
    args = parser.parse_args()

    all_runs = RUN_SETS[args.set]
    runs = all_runs if args.only is None else [r for r in all_runs if r["name"] in args.only]
    if not runs:
        print(f"no runs matched {args.only!r}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    results = []
    started = time.time()

    for i, run in enumerate(runs, 1):
        print(f"\n{'#' * 78}")
        print(f"# [{i}/{len(runs)}] {run['name']}")
        print(f"# {run['purpose']}")
        print(f"{'#' * 78}\n", flush=True)

        cmd = [
            sys.executable,
            "scripts/train_attention_fusion.py",
            "--modality", run["modality"],
            "--split-strategy", run["split"],
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--out-dir", str(out_dir),
            "--tag", run["name"],
            "--eeg-features", run.get("eeg_features", "raw"),
            "--audio-features", run.get("audio_features", "mfcc"),
        ]
        if run.get("specaugment"):
            cmd.append("--specaugment")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"FAILED: {run['name']} (exit {proc.returncode})", file=sys.stderr)
            return proc.returncode

        # Newest directory carrying this run's tag.
        candidates = sorted(out_dir.glob(f"{run['name']}_*"))
        if not candidates:
            print(f"no output directory for {run['name']}", file=sys.stderr)
            return 1
        payload = json.loads((candidates[-1] / "results.json").read_text(encoding="utf-8"))
        results.append(
            {
                "name": run["name"],
                "purpose": run["purpose"],
                "modality": run["modality"],
                "split": run["split"],
                "run_dir": candidates[-1].name,
                "eeg_features": run.get("eeg_features", "raw"),
                "audio_features": run.get("audio_features", "mfcc"),
                "specaugment": bool(run.get("specaugment")),
                "test_accuracy": payload["test"]["accuracy"],
                "test_uar": payload["test"]["uar"],
                "test_macro_f1": payload["test"]["macro_f1"],
                "best_val_accuracy": payload["best_val_accuracy"],
                "best_epoch": payload["best_epoch"],
                "final_train_accuracy": payload["history"][-1]["train_acc"],
                "per_class_recall": payload["test"]["per_class_recall"],
            }
        )

    summary_path = out_dir / f"ablation_summary_{args.set}.json"
    summary_path.write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "seed": args.seed,
                "duration_seconds": round(time.time() - started, 1),
                "runs": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'=' * 92}")
    print(f"ABLATION SUMMARY ({args.set})")
    print("=" * 92)
    print(f"{'run':<40} {'val':>8} {'test acc':>9} {'UAR':>8} {'train':>8} {'gap':>7}")
    for r in results:
        gap = r["final_train_accuracy"] - r["best_val_accuracy"]
        print(
            f"{r['name']:<40} {r['best_val_accuracy']:>8.4f} "
            f"{r['test_accuracy']:>9.4f} {r['test_uar']:>8.4f} "
            f"{r['final_train_accuracy']:>8.4f} {gap:>7.3f}"
        )
    print(f"\nchance = 0.2000")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
