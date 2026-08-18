"""Live demonstration: replay held-out subjects through the trained models.

The demonstration constraint for this project is that no EEG amplifier is
available at presentation time. That constraint does not prevent an honest
demonstration, because live capture from the presenter would be the *weaker*
option in any case:

* the presenter is not in the corpus, was recorded on different hardware, in a
  different room, speaking a different language from the EAV participants, so a
  wrong prediction would be uninterpretable -- domain shift, model error and
  labelling error are indistinguishable;
* there is no ground truth for the presenter's own affective state, so nothing
  can be scored.

Replaying subjects the model has never seen keeps ground truth available, so
every prediction can be marked right or wrong in front of the audience. Subjects
5, 19, 22, 26, 27, 30, 40 and 42 were held out of training entirely; see
``outputs/multimodal_improved_subject_independent_20260809_011134/results.json``.

The script runs the multimodal model and the audio-only model side by side on
the same trial, which demonstrates the thesis's central empirical finding
directly: the two agree on most trials, and where they disagree neither is
reliably right.

Usage
-----
    python scripts/demo_replay.py                  # 20 trials, step with Enter
    python scripts/demo_replay.py --auto --delay 2 # hands-free, for a recording
    python scripts/demo_replay.py --trials 8 --subjects 5 19
    python scripts/demo_replay.py --no-color       # plain text
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import EmotionPredictor
from src.late_fusion import LateFusionPredictor
from src.preprocessing.eav_dataset import EAVMultimodalDataset
from src.preprocessing.eav_labels import EMOTION_NAMES

#: Checkpoints written by scripts/train_attention_fusion.py. All were trained on
#: the same 28 training subjects and validated on the same 6, so their
#: predictions on the held-out 8 are directly comparable.
#:
#: The multimodal system is the **late fusion** of the two unimodal models --
#: the configuration that measured best (67.02% +/- 1.11 across three seeds of
#: cross-validation, against 62.77% +/- 1.52 for audio alone; paired difference
#: +4.25pp +/- 0.41, p = 0.0031). It averages the two output distributions and
#: has no fusion parameters. See docs/CHANGELOG.md.
EEG_CHECKPOINT = Path(
    "outputs/eeg_de_subject_independent_20260809_004512/model_best.pt"
)
AUDIO_CHECKPOINT = Path(
    "outputs/audio_mel_subject_independent_20260809_004702/model_best.pt"
)

#: Subjects excluded from training and validation for both checkpoints above.
HELD_OUT_SUBJECTS = (5, 19, 22, 26, 27, 30, 40, 42)

CHANCE_RATE = 1.0 / len(EMOTION_NAMES)

BAR_WIDTH = 32

#: Box-drawing and bar glyphs, with an ASCII fallback for consoles whose code
#: page cannot encode them (the Windows default is cp1252, which cannot).
GLYPHS_UNICODE = {
    "fill": "█",
    "empty": "·",
    "rule": "─",
    "heavy_rule": "═",
    "marker": "◀",
    "arrow": "→",
}
GLYPHS_ASCII = {
    "fill": "#",
    "empty": ".",
    "rule": "-",
    "heavy_rule": "=",
    "marker": "<--",
    "arrow": "->",
}


def select_glyphs() -> dict:
    """Use box-drawing glyphs only if the console can actually encode them."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(GLYPHS_UNICODE.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return GLYPHS_ASCII
    return GLYPHS_UNICODE


GLYPHS = GLYPHS_ASCII  # replaced in run_demo once stdout is configured


class Palette:
    """ANSI escapes, or empty strings when colour is disabled."""

    def __init__(self, enabled: bool):
        def code(value: str) -> str:
            return value if enabled else ""

        self.reset = code("\033[0m")
        self.bold = code("\033[1m")
        self.dim = code("\033[2m")
        self.green = code("\033[32m")
        self.red = code("\033[31m")
        self.yellow = code("\033[33m")
        self.blue = code("\033[34m")
        self.cyan = code("\033[36m")
        self.grey = code("\033[90m")


def configure_console() -> None:
    """Turn on ANSI escapes and, where possible, UTF-8 output.

    Windows consoles default to cp1252, which cannot encode the bar and
    box-drawing glyphs. Reconfiguring is best-effort: if it fails,
    :func:`select_glyphs` falls back to ASCII rather than crashing mid-demo.
    """
    if os.name == "nt":
        os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass


def probability_bar(probability: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(probability * width))
    return GLYPHS["fill"] * filled + GLYPHS["empty"] * (width - filled)


def format_distribution(
    probabilities: Dict[str, float],
    true_label: str,
    predicted: str,
    palette: Palette,
) -> List[str]:
    """One line per emotion, ordered as in EMOTION_NAMES, never re-sorted.

    Keeping the class order fixed across trials means the audience reads the
    same row position for the same emotion every time.
    """
    lines = []
    for name in EMOTION_NAMES:
        probability = probabilities[name]
        if name == true_label:
            colour, marker = palette.green, f"{GLYPHS['marker']} true"
        elif name == predicted:
            colour, marker = palette.red, f"{GLYPHS['marker']} predicted"
        else:
            colour, marker = palette.grey, ""
        lines.append(
            f"    {colour}{name:<10}{probability:5.1%} "
            f"{probability_bar(probability)}{palette.reset} "
            f"{palette.dim}{marker}{palette.reset}"
        )
    return lines


def verdict(correct: bool, palette: Palette) -> str:
    if correct:
        return f"{palette.green}{palette.bold}CORRECT{palette.reset}"
    return f"{palette.red}{palette.bold}WRONG{palette.reset}"


def load_predictors(eeg_path: Path, audio_path: Path) -> Dict[str, object]:
    """Load the audio model, and the late fusion that averages it with the EEG model.

    ``predictors["fusion"]`` must be a :class:`LateFusionPredictor`, not an
    ``EmotionPredictor`` built from the EEG checkpoint: the latter would silently
    ignore the audio argument and label an EEG-only prediction as multimodal.
    """
    for name, path in (("eeg", eeg_path), ("audio", audio_path)):
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found at {path}.\n"
                f"Train it with:  python scripts/run_ablations.py --set improved"
            )
    return {
        "fusion": LateFusionPredictor(str(eeg_path), str(audio_path)),
        "audio": EmotionPredictor(str(audio_path)),
    }


def build_dataset(subjects: List[int]) -> EAVMultimodalDataset:
    """Load held-out trials in the representation the checkpoints were trained on.

    ``normalize_audio`` is disabled here because :class:`EmotionPredictor`
    z-scores audio itself; normalising in both places would double-standardise
    the input and silently shift it away from the training distribution.
    """
    return EAVMultimodalDataset(
        subjects=subjects,
        eeg_features="de",
        audio_features="mel",
        normalize_audio=False,
    )


def run_demo(args: argparse.Namespace) -> int:
    global GLYPHS
    configure_console()
    GLYPHS = GLYPHS_ASCII if args.ascii else select_glyphs()
    palette = Palette(enabled=not args.no_color)

    subjects = args.subjects or list(HELD_OUT_SUBJECTS)
    unseen = set(subjects) - set(HELD_OUT_SUBJECTS)
    if unseen:
        print(
            f"{palette.yellow}WARNING: subjects {sorted(unseen)} were part of "
            f"training or validation for these checkpoints. Predictions on them "
            f"are not evidence of generalisation.{palette.reset}\n"
        )

    print(f"{palette.bold}Loading models...{palette.reset}")
    predictors = load_predictors(args.eeg_checkpoint, args.audio_checkpoint)
    dataset = build_dataset(subjects)

    info = predictors["fusion"].info()
    combined = info["eeg"]["parameters"] + info["audio"]["parameters"]
    audio_params = predictors["audio"].metadata["parameters"]

    print(
        f"  multimodal (late fusion) : {combined:,} parameters, "
        f"{info['fusion_parameters']} in the combiner ({info['rule']})\n"
        f"  audio only               : {audio_params:,} parameters\n"
        f"  held-out subjects        : {', '.join(str(s) for s in subjects)}\n"
        f"  trials available         : {len(dataset)}\n"
    )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(dataset))[: args.trials]

    tally = {"fusion": 0, "audio": 0, "agree": 0}
    n_shown = 0

    for position, index in enumerate(order, start=1):
        sample = dataset[int(index)]
        eeg = sample["eeg"].numpy()
        audio = sample["audio"].numpy()
        true_label = EMOTION_NAMES[sample["emotion"]]

        fusion_result = predictors["fusion"].predict(eeg_data=eeg, audio_data=audio)
        audio_result = predictors["audio"].predict(audio_data=audio)

        fusion_correct = fusion_result["emotion"] == true_label
        audio_correct = audio_result["emotion"] == true_label
        tally["fusion"] += int(fusion_correct)
        tally["audio"] += int(audio_correct)
        tally["agree"] += int(fusion_result["emotion"] == audio_result["emotion"])
        n_shown += 1

        print(f"{palette.blue}{GLYPHS['rule'] * 72}{palette.reset}")
        print(
            f"{palette.bold}Trial {position}/{len(order)}{palette.reset}   "
            f"{palette.dim}subject {sample['subject_id']}, "
            f"trial index {sample['trial_index']}   "
            f"EEG {eeg.shape}, audio {audio.shape}{palette.reset}"
        )
        print(f"  ground truth: {palette.bold}{true_label}{palette.reset}\n")

        print(
            f"  {palette.cyan}{palette.bold}LATE FUSION (EEG + audio){palette.reset}"
            f"  {GLYPHS['arrow']} {fusion_result['emotion']} "
            f"({fusion_result['confidence']:.1%})  {verdict(fusion_correct, palette)}"
        )
        for line in format_distribution(
            fusion_result["probabilities"], true_label, fusion_result["emotion"], palette
        ):
            print(line)

        print(
            f"\n  {palette.cyan}AUDIO ONLY{palette.reset}"
            f"                {GLYPHS['arrow']} {audio_result['emotion']} "
            f"({audio_result['confidence']:.1%})  {verdict(audio_correct, palette)}"
        )

        print(
            f"\n  {palette.dim}running: multimodal "
            f"{tally['fusion']}/{n_shown} ({tally['fusion'] / n_shown:.0%})   "
            f"audio {tally['audio']}/{n_shown} ({tally['audio'] / n_shown:.0%})   "
            f"models agree {tally['agree']}/{n_shown}{palette.reset}"
        )

        if position < len(order):
            if args.auto:
                time.sleep(args.delay)
            else:
                try:
                    input(f"\n{palette.dim}  [Enter] next trial{palette.reset}")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

    print(f"{palette.blue}{GLYPHS['heavy_rule'] * 72}{palette.reset}")
    print(f"{palette.bold}Session summary{palette.reset}  ({n_shown} held-out trials)\n")
    print(
        f"  multimodal (late fusion) : {tally['fusion']}/{n_shown} "
        f"= {tally['fusion'] / n_shown:.1%}"
    )
    print(
        f"  audio only               : {tally['audio']}/{n_shown} "
        f"= {tally['audio'] / n_shown:.1%}"
    )
    print(f"  chance                   : {CHANCE_RATE:.1%}")
    print(
        f"  models agreed on         : {tally['agree']}/{n_shown} trials\n"
    )
    print(
        f"{palette.dim}  A sample this small is illustrative only. The figures of "
        f"record come from\n  7-fold subject-wise cross-validation over all 42 "
        f"subjects, repeated across 3 seeds:\n"
        f"  late fusion 67.02% +/- 1.11, audio-only 62.77% +/- 1.52,\n"
        f"  paired difference +4.25pp +/- 0.41 (p = 0.0031). "
        f"See docs/CHANGELOG.md."
        f"{palette.reset}"
    )
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay held-out subjects through the trained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="number of trials to replay"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help=f"subject ids to draw from (default: the held-out {list(HELD_OUT_SUBJECTS)})",
    )
    parser.add_argument(
        "--auto", action="store_true", help="advance automatically instead of on Enter"
    )
    parser.add_argument(
        "--delay", type=float, default=2.5, help="seconds between trials with --auto"
    )
    parser.add_argument("--seed", type=int, default=7, help="trial shuffling seed")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    parser.add_argument(
        "--ascii", action="store_true", help="force ASCII glyphs instead of box drawing"
    )
    parser.add_argument("--eeg-checkpoint", type=Path, default=EEG_CHECKPOINT)
    parser.add_argument("--audio-checkpoint", type=Path, default=AUDIO_CHECKPOINT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_demo(parse_args()))
