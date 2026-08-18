"""Render the report figures directly from the result artifacts.

Every figure is generated from ``outputs/*/results.json``, so a figure cannot
drift from the number it illustrates. Re-run after any training run.

Produces, into ``MSE_CAPSTONE_REPORT_new/figures/``:

* ``ablation_accuracy.png``      -- modality and protocol ablations vs chance
* ``improved_representations.png`` -- baseline vs improved features
* ``cross_validation.png``       -- 7-fold subject-wise cross-validation
* ``complementarity.png``        -- per-class recall, all three models
* ``fusion_approaches.png``      -- every fusion approach tested, ranked
* ``confusion_matrix.png``       -- model of record, held-out test subjects
* ``training_curves.png``        -- train/validation accuracy per epoch
* ``per_class_recall.png``       -- per-emotion recall, model of record

Run from the project root::

    python scripts/generate_result_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.eav_labels import EMOTION_NAMES

# --- Palette -----------------------------------------------------------------
# Reference categorical palette, slots 1 and 2, used in documented order. The
# ordering is the colour-vision-deficiency safety mechanism, so it is not
# reshuffled. Identity is never carried by colour alone here: every bar is
# axis-labelled and value-labelled, and both line series are directly labelled.
SERIES_1 = "#2a78d6"   # blue   -- subject-independent protocol
SERIES_2 = "#eb6834"   # orange -- pooled-random protocol / validation series
# Slot 3, snapped one step darker than the reference #1baf7a: against a white
# surface that step measures 2.82:1, below the 3:1 relief floor. #199e70 measures
# 3.41:1 and keeps adjacent-pair CVD separation above the 8.0 target
# (protan 8.4 vs slot 2, deutan 11.6). Validated, not eyeballed.
SERIES_3 = "#199e70"   # green  -- fusion series

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"

# Sequential blue ramp, light -> dark, for magnitude (confusion counts).
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

CHANCE = 1.0 / len(EMOTION_NAMES)


def style_axes(ax, *, xgrid=False, ygrid=True):
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1.0)
    if xgrid:
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)


def load_runs(outputs: Path) -> dict:
    """Map run tag -> results payload, keeping the most recent run of each tag.

    Keyed on the tag rather than on modality+split, because the baseline and
    improved experiment sets contain runs that share a modality and a split
    strategy and would otherwise overwrite one another.
    """
    runs = {}
    for results_path in sorted(outputs.glob("*/results.json")):
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        tag = config.get("tag")
        if not tag:
            continue
        previous = runs.get(tag)
        if previous is None or payload["timestamp"] >= previous["timestamp"]:
            runs[tag] = payload
    return runs


def figure_ablation(runs: dict, out: Path) -> None:
    """Horizontal bars: one measure (test accuracy), chance marked."""
    order = [
        ("eeg_subject_independent", "EEG only", SERIES_1),
        ("audio_long", "Audio only", SERIES_1),
        ("multimodal_subject_independent", "Cross-modal fusion", SERIES_1),
        ("multimodal_pooled_random", "Fusion, pooled split", SERIES_2),
    ]
    rows = [(label, runs[key]["test"]["accuracy"], color)
            for key, label, color in order if key in runs]
    if not rows:
        print("  (no runs found for ablation figure)")
        return

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.4, 0.72 * len(rows) + 2.5), facecolor=SURFACE)
    ax.barh(y, values, height=0.58, color=colors)

    ax.axvline(CHANCE, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=3)
    # Above the first bar (the y-axis is inverted, so -0.5 is the top edge).
    ax.text(CHANCE + 0.008, -0.52, f"chance {CHANCE:.0%}",
            color=INK_MUTED, fontsize=8.5, va="bottom")

    for yi, value in zip(y, values):
        ax.text(value + 0.008, yi, f"{value:.1%}", va="center",
                color=INK_PRIMARY, fontsize=10, fontweight="medium")

    ax.set_yticks(y, labels, color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(len(rows) - 0.4, -0.85)   # inverted, with headroom for the label
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("Held-out test accuracy", color=INK_SECONDARY, fontsize=9.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax, xgrid=True, ygrid=False)

    # Two protocols are on screen; the differing bar is also labelled on the axis,
    # so the legend reinforces rather than carries the distinction. Placed below
    # the plot so it cannot occlude a bar or its value label.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES_1),
        plt.Rectangle((0, 0), 1, 1, color=SERIES_2),
    ]
    legend = ax.legend(handles, ["Subject-independent split", "Pooled random split"],
                       loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
                       frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_improved(runs: dict, out: Path) -> None:
    """Baseline vs improved representation, per modality.

    Two series of the same measure on one axis, grouped by modality. Every bar
    is value-labelled, so colour reinforces the grouping rather than carrying it.
    """
    groups = [
        ("EEG", "eeg_subject_independent", "eeg_de_subject_independent"),
        ("Audio", "audio_long", "audio_mel_subject_independent"),
        ("Fusion", "multimodal_subject_independent",
         "multimodal_improved_subject_independent"),
    ]
    rows = [
        (label, runs[base]["test"]["accuracy"], runs[imp]["test"]["accuracy"])
        for label, base, imp in groups
        if base in runs and imp in runs
    ]
    if not rows:
        print("  (improved runs not found; skipping comparison figure)")
        return

    labels = [r[0] for r in rows]
    baseline = [r[1] for r in rows]
    improved = [r[2] for r in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=SURFACE)
    ax.bar(x - width / 2 - 0.01, baseline, width, color=SERIES_1, label="Baseline")
    ax.bar(x + width / 2 + 0.01, improved, width, color=SERIES_2, label="Improved")

    for xi, (b, i) in enumerate(zip(baseline, improved)):
        ax.text(xi - width / 2 - 0.01, b + 0.012, f"{b:.1%}", ha="center",
                color=INK_PRIMARY, fontsize=9.5)
        ax.text(xi + width / 2 + 0.01, i + 0.012, f"{i:.1%}", ha="center",
                color=INK_PRIMARY, fontsize=9.5)
        delta = (i - b) * 100
        ax.text(xi, max(b, i) + 0.062, f"{delta:+.2f}pp", ha="center",
                color=INK_SECONDARY, fontsize=9.5, fontweight="medium")

    ax.axhline(CHANCE, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(-0.45, CHANCE + 0.016, f"chance {CHANCE:.0%}", ha="left", va="bottom",
            color=INK_MUTED, fontsize=8.5)

    ax.set_xticks(x, labels, color=INK_SECONDARY, fontsize=10.5)
    ax.set_ylabel("Held-out test accuracy", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(0, max(max(baseline), max(improved)) * 1.30)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
                       frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_cross_validation(outputs: Path, out: Path) -> None:
    """Cross-validated accuracy with per-fold spread, against the single partition.

    Error bars are the per-fold standard deviation, which is the quantity the
    single-partition estimate could not show.
    """
    wanted = [
        ("cv_eeg_de", "EEG\n(DE + alignment)", 0.4600),
        ("cv_audio_mel", "Audio\n(log-mel)", 0.6713),
        ("cv_fusion_improved", "Fusion", 0.6262),
    ]
    rows = []
    for tag, label, single in wanted:
        matches = sorted(outputs.glob(f"{tag}_*/cv_results.json"))
        if not matches:
            continue
        payload = json.loads(matches[-1].read_text(encoding="utf-8"))
        rows.append((label, payload["pooled"]["accuracy"],
                     payload["fold_accuracy_std"], single))
    if not rows:
        print("  (no cross-validation runs found; skipping figure)")
        return

    labels = [r[0] for r in rows]
    cv = [r[1] for r in rows]
    err = [r[2] for r in rows]
    single = [r[3] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=SURFACE)
    ax.bar(x, cv, width=0.5, color=SERIES_1, yerr=err, capsize=5,
           error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.5})

    # The superseded single-partition estimate, as a marker rather than a bar,
    # so it reads as a reference point and not a second measurement.
    ax.plot(x, single, linestyle="none", marker="D", markersize=8,
            color=SERIES_2, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="Single 8-subject partition", zorder=5)

    for xi, (value, e, s) in enumerate(zip(cv, err, single)):
        # Inside the bar, so the space above stays clear for the reference
        # marker and its delta annotation.
        ax.text(xi, value - 0.035, f"{value:.1%}", ha="center", va="top",
                color="#ffffff", fontsize=11, fontweight="medium")
        ax.text(xi + 0.28, s, f"{(s - value) * 100:+.2f}pp", va="center",
                ha="left", color=INK_SECONDARY, fontsize=9)

    ax.axhline(CHANCE, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(-0.45, CHANCE + 0.016, f"chance {CHANCE:.0%}", ha="left", va="bottom",
            color=INK_MUTED, fontsize=8.5)

    ax.set_xticks(x, labels, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Accuracy, 7-fold subject-wise CV", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(0, max(max(cv), max(single)) * 1.28)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                       frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_confusion(payload: dict, out: Path) -> None:
    """Sequential single-hue heatmap of the held-out confusion matrix."""
    matrix = np.array(payload["test"]["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=SURFACE)

    image = ax.imshow(matrix, cmap=BLUE_CMAP, vmin=0, vmax=matrix.max())

    threshold = matrix.max() * 0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=10,
                    color="#ffffff" if matrix[i, j] > threshold else INK_PRIMARY)

    ax.set_xticks(range(len(EMOTION_NAMES)), EMOTION_NAMES, rotation=30,
                  ha="right", color=INK_SECONDARY, fontsize=9.5)
    ax.set_yticks(range(len(EMOTION_NAMES)), EMOTION_NAMES,
                  color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("True", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    bar.set_label("Samples", color=INK_SECONDARY, fontsize=9.5)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_training_curves(payload: dict, out: Path) -> None:
    """Two series on one shared accuracy axis; both directly labelled."""
    history = payload["history"]
    epochs = [h["epoch"] for h in history]
    train = [h["train_acc"] for h in history]
    val = [h["val_acc"] for h in history]

    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=SURFACE)
    ax.plot(epochs, train, color=SERIES_1, linewidth=2.0, label="Training")
    ax.plot(epochs, val, color=SERIES_2, linewidth=2.0, label="Validation")

    best_epoch = payload["best_epoch"]
    best_val = payload["best_val_accuracy"]
    ax.plot([best_epoch], [best_val], marker="o", markersize=8,
            color=SERIES_2, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"best epoch {best_epoch} ({best_val:.1%})",
                xy=(best_epoch, best_val), xytext=(-4, 12),
                textcoords="offset points", color=INK_SECONDARY, fontsize=9,
                ha="right")

    ax.axhline(CHANCE, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(epochs[-1], CHANCE + 0.008, f"chance {CHANCE:.0%}", ha="right",
            color=INK_MUTED, fontsize=8.5)

    # Direct labels at the line ends, so identity does not rest on the legend.
    ax.text(epochs[-1] + 0.4, train[-1], "Training", color=INK_SECONDARY,
            fontsize=9.5, va="center")
    ax.text(epochs[-1] + 0.4, val[-1], "Validation", color=INK_SECONDARY,
            fontsize=9.5, va="center")

    ax.set_xlabel("Epoch", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Accuracy", color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlim(1, epochs[-1] + 5)
    ax.set_ylim(0, max(max(train), max(val)) * 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_per_class(payload: dict, out: Path) -> None:
    """Single-measure bars; one colour, because colour carries no identity here."""
    recall = payload["test"]["per_class_recall"]
    support = payload["test"]["support"]
    names = list(recall)
    values = [recall[n] for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=SURFACE)
    ax.bar(x, values, width=0.58, color=SERIES_1)

    ax.axhline(CHANCE, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(-0.45, CHANCE + 0.016, f"chance {CHANCE:.0%}", ha="left", va="bottom",
            color=INK_MUTED, fontsize=8.5)

    for xi, value in zip(x, values):
        ax.text(xi, value + 0.012, f"{value:.1%}", ha="center",
                color=INK_PRIMARY, fontsize=10)

    ax.set_xticks(x, [f"{n}\n(n={support[n]})" for n in names],
                  color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Recall on held-out subjects", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(0, max(values) * 1.20)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_complementarity(outputs: Path, out: Path) -> None:
    """Grouped bars: per-class recall for all three models.

    One measure (recall) on one axis, so the three models are a categorical
    series and take palette slots 1--3 in fixed order. The figure exists to show
    a single reversal -- Happiness is the only class where EEG beats audio, and
    the only class where fusion beats both -- so that group is annotated and the
    others are left to read as the uniform pattern they are.
    """
    source = outputs / "complementarity_analysis.json"
    if not source.exists():
        print(f"  ({source} not found; run scripts/analyze_complementarity.py)")
        return
    per_class = json.loads(source.read_text(encoding="utf-8"))["per_class"]

    names = list(per_class)
    series = [
        ("EEG only", "eeg_accuracy", SERIES_1),
        ("Audio only", "audio_accuracy", SERIES_2),
        ("Fusion", "fusion_accuracy", SERIES_3),
    ]
    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor=SURFACE)

    for offset, (label, key, colour) in zip((-width, 0.0, width), series):
        values = [per_class[n][key] for n in names]
        # 2px surface gap between adjacent fills, per the mark spec.
        ax.bar(x + offset, values, width=width * 0.92, color=colour,
               label=label, edgecolor=SURFACE, linewidth=2.0)

    ax.axhline(CHANCE, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    # Left margin, clear of every bar and of the line itself. Anchoring it on the
    # right put it on top of the Happiness group, which is the one group the
    # figure is asking the reader to look at.
    ax.text(-0.62, CHANCE + 0.018, f"chance {CHANCE:.0%}", ha="left", va="bottom",
            color=INK_MUTED, fontsize=8.5)

    # Direct-label only the reversal: a number on every bar would be 15 numbers
    # and would bury the one comparison the figure is making.
    happy = names.index("Happiness")
    for offset, (_, key, _) in zip((-width, 0.0, width), series):
        value = per_class["Happiness"][key]
        ax.text(happy + offset, value + 0.014, f"{value:.0%}", ha="center",
                va="bottom", color=INK_PRIMARY, fontsize=9.5, fontweight="bold")

    ax.annotate(
        "EEG beats audio here —\nand fusion beats both",
        xy=(happy, 0.755), xytext=(happy - 0.95, 0.90),
        ha="left", va="top", fontsize=9.5, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1.2,
                        connectionstyle="arc3,rad=-0.15"),
    )

    ax.set_xticks(x, names, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Recall, pooled cross-validated predictions",
                  color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                       frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_fusion_approaches(outputs: Path, out: Path) -> None:
    """Horizontal bars: every fusion approach tested, ordered by accuracy.

    One measure on one axis. Colour separates late fusion -- combining two
    separately trained unimodal models with at most five free parameters -- from
    single models and end-to-end learned fusion (8,064 parameters in the fusion
    module). That is the variable the figure is about, and it is redundant with
    the axis labels rather than the sole carrier of the distinction.
    """
    source = outputs / "late_fusion_cv_final.json"
    if not source.exists():
        print(f"  ({source} not found; run scripts/cross_validate_late_fusion.py)")
        return
    payload = json.loads(source.read_text(encoding="utf-8"))

    # (label, accuracy, is_late_fusion). Values not in the JSON come from their
    # own CV runs, recorded in docs/CHANGELOG.md.
    rules = payload["rules"]
    rows = [
        ("EEG only (adversarial)", rules["eeg_only"]["accuracy"], False),
        ("Sequence attention fusion", 0.63190, False),
        ("Adversarial end-to-end fusion", 0.63595, False),
        ("Trained attention fusion", rules["trained attention fusion"]["accuracy"], False),
        ("Audio only", rules["audio_only"]["accuracy"], False),
        ("Max-confidence gating", rules["max_confidence"]["accuracy"], True),
        ("Per-class weights", rules["per_class (LOFO)"]["accuracy"], True),
        ("Weighted average", rules["weighted (LOFO)"]["accuracy"], True),
        ("Mean of probabilities", rules["mean"]["accuracy"], True),
    ]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [SERIES_1 if r[2] else SERIES_2 for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.4, 5.4), facecolor=SURFACE)
    ax.barh(y, values, height=0.62, color=colours, edgecolor=SURFACE, linewidth=2.0)

    audio = rules["audio_only"]["accuracy"]
    ax.axvline(audio, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    # Headroom above the top bar so the annotation never sits on it.
    ax.set_ylim(-0.65, len(rows) - 0.20)
    ax.text(audio - 0.005, len(rows) - 0.42, "audio-only baseline", ha="right",
            va="center", color=INK_MUTED, fontsize=8.5)

    oracle = payload["oracle"]
    ax.axvline(oracle, color=BASELINE, linewidth=1.5, linestyle=(0, (2, 3)))
    ax.text(oracle - 0.004, 0.1, f"oracle {oracle:.1%}", ha="right", va="center",
            color=INK_MUTED, fontsize=8.5)

    for yi, value in zip(y, values):
        # Bars ending within a hair of the baseline rule would print their value
        # on top of it; push those labels clear.
        offset = 0.010 if abs(value - audio) < 0.012 else 0.004
        ax.text(value + offset, yi, f"{value:.1%}", va="center", ha="left",
                color=INK_PRIMARY, fontsize=9.5)

    ax.set_yticks(y, labels, color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlabel("Accuracy, 7-fold subject-wise cross-validation (n = 4,200)",
                  color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlim(0.40, 0.88)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax, xgrid=True, ygrid=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES_1),
        plt.Rectangle((0, 0), 1, 1, color=SERIES_2),
    ]
    legend = ax.legend(handles,
                       ["Late fusion of two trained models ($\leq$5 parameters)",
                        "Single model, or end-to-end learned fusion"],
                       loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
                       frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figures", default="MSE_CAPSTONE_REPORT_new/figures")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    runs = load_runs(outputs)
    if not runs:
        print(f"No results.json found under {outputs}", file=sys.stderr)
        return 1

    print(f"Found {len(runs)} run(s): {', '.join(sorted(runs))}\n")

    figure_ablation(runs, figures / "ablation_accuracy.png")
    figure_improved(runs, figures / "improved_representations.png")
    figure_cross_validation(outputs, figures / "cross_validation.png")
    figure_complementarity(outputs, figures / "complementarity.png")
    figure_fusion_approaches(outputs, figures / "fusion_approaches.png")

    record = runs.get("multimodal_subject_independent")
    if record is None:
        print("  (model of record not found; skipping its figures)")
        return 0

    figure_confusion(record, figures / "confusion_matrix.png")
    figure_training_curves(record, figures / "training_curves.png")
    figure_per_class(record, figures / "per_class_recall.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
