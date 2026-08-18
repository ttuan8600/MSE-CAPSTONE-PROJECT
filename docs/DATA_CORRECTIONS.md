# Data pipeline corrections

**Status:** applied 2026-08-08. All results predating this date are withdrawn.

This document records four defects found in the EAV data pipeline, the evidence
for each, and what was changed. Three of them invalidated every experimental
result the project had produced; the fourth made the deployed API non-functional.

Every figure quoted here is reproducible:

```bash
python scripts/audit_eav_alignment.py --json outputs/eav_alignment_audit.json
python scripts/verify_data_fix.py     --json outputs/data_fix_verification.json
pytest tests/ -q
```

---

## Summary

| # | Defect | Effect | Status |
| --- | --- | --- | --- |
| 1 | EEG array read on the wrong axis | Every EEG tensor was a single 2 ms time-point, reshaped so the trial axis posed as time | Fixed |
| 2 | One EEG tensor shared by 100 samples | The EEG stream carried zero label information; all "fusion" results were audio-only | Fixed |
| 3 | Two incompatible split implementations, neither subject-independent | 69% of the reported test set had been trained on; every subject appeared on both sides | Fixed |
| 4 | Inference model did not match the trained architecture | The REST API could never load the model of record | Fixed |
| 5 | Load failures silently replaced with random noise | Proportion of noise in any result was unknowable | Fixed |

---

## 1. The EEG array was read on the wrong axis

### What the data actually is

Each subject ships `subject<N>_eeg.mat` containing one variable — named `seg`
for 11 subjects and `seg1` for the other 31 — of shape:

```
(10000, 30, 200)
 time    ch  trials
```

10,000 samples at 500 Hz is a 20-second trial. There are 30 EEG channels and 200
trials per subject (100 *Listening*, 100 *Speaking*).

### What the loader did

`src/preprocessing/data_loader.py` documented the array as
`(n_segments, n_channels, time_steps)` and read:

```python
eeg_data = eeg_raw[0, :, :]      # believed: "the first segment"
```

Against the real layout this selects **time index 0** — one 2-millisecond
sample — producing a `(30, 200)` matrix whose second axis is *trials, not time*.
It was then truncated to 28 channels and fed to a 1-D CNN as though it were a
200-step time series.

### Evidence

`tests/test_eav_io.py::test_returns_trial_major_orientation` pins the correct
orientation element-wise. The audit confirms the shape is identical across all
42 subjects:

```
seg shapes observed:
  (200, 30, 10000): 42 subject(s)      # after transposition to trial-major
```

### Fix

`src/preprocessing/eav_io.py::load_subject_segments` transposes to
`(trials, channels, time)` and validates the axis order, raising if axis 1 is not
30 channels or if axis 0 is implausibly short for a time axis. Nothing guesses.

---

## 2. One hundred samples shared one EEG tensor

### What the loader did

`_find_matched_samples` paired each subject's single `.mat` file with **every**
`.wav` file in that subject's `Audio/` directory:

```python
for audio_file in sorted(audio_files):
    matched.append({'eeg': eeg_file, 'audio': audio_file, ...})
```

Each subject has one EEG file and 100 audio files, so this produced 100 samples
per subject. Because `_load_eeg` is deterministic and always returned
`seg[0, :, :]`, all 100 received a **byte-identical** EEG tensor — while carrying
five different emotion labels.

### Consequence

The EEG branch received constant input with varying targets. It could not
contribute label information even in principle. **Every result the project
reported as "multimodal fusion" was in substance an audio-only result**, and the
cross-modal attention module was attending between audio features and a
per-subject constant.

### Evidence

From `scripts/verify_data_fix.py`:

```
 subject  samples   original  corrected
       1      100          1        100
       2      100          1        100
       3      100          1        100
       4      100          1        100
       5      100          1        100

Across 5 subjects (500 samples): 5 distinct EEG tensors originally,
                                 500 after the fix.
EEG tensor shape: [28, 200] (original) -> [30, 2500] (corrected)
```

### Fix

The three-digit prefix of every media filename is the **1-based trial index**
into the EEG trial axis — `002_Trial_02_Speaking_Neutral_Aud.wav` is trial 2.
Each audio clip is now paired with its own EEG trial through that index.

This was verified, not assumed. `scripts/audit_eav_alignment.py` cross-checks the
emotion and condition parsed from every filename against the dataset's own label
matrix across all subjects:

```
filename/label cross-check: 12600 files, 0 mismatches
subjects with errors:   0
subjects with warnings: 0
```

`scripts/preprocess_eav.py` re-asserts the same agreement while building the
cache, so a future data refresh cannot silently reintroduce a misalignment.

---

## 3. Labels now come from the dataset's ground truth

The original pipeline derived emotion labels by substring-matching audio
filenames, defaulting anything unmatched to `Neutral`. The dataset ships explicit
per-trial ground truth in `subject<N>_eeg_label.mat`: a `(10, 200)` one-hot
matrix over five emotions × two conditions, decoded as `emotion = row // 2`,
`condition = row % 2`.

The row semantics were derived empirically by cross-referencing against filenames
and are recorded in `src/preprocessing/eav_labels.py`.

**In this instance the filename-derived labels were in fact correct** — both
methods yield 840 samples per class:

```
filename-derived (original): {Anger: 840, Calmness: 840, Happiness: 840, Neutral: 840, Sadness: 840}
ground-truth label matrix   : {Neutral: 840, Anger: 840, Calmness: 840, Sadness: 840, Happiness: 840}
```

The pipeline now reads the label matrix regardless, because a silent default to
`Neutral` on an unparsed filename is not a failure mode worth keeping. Ambiguous
label columns raise `LabelDecodeError`.

> **Note.** The classes are exactly balanced. Statements in the archived
> documentation attributing poor per-class results to class imbalance — and the
> hand-tuned focal-loss weights `[1.0, 1.0, 1.5, 1.5, 1.0]` motivated by it — were
> addressing a problem that does not exist in this dataset. The real cause of the
> collapsed per-class scores was defect 2.

---

## 4. Splitting: two RNGs, and no subject independence

### Defect 4a — the RNG mismatch

Two scripts split the data with the same seed value but different random number
generators, which produce unrelated permutations:

| Script | Method |
| --- | --- |
| `train_attention_fusion.py:87` | `np.random.seed(42)` + `np.random.permutation` |
| `evaluate_finetuned_model.py:155` | `torch.randperm(generator=manual_seed(42))` |
| `finetune_attention_fusion.py:241` | `torch.randperm(generator=manual_seed(42))` |

`evaluate_finetuned_model.py` carried the comment *"same split as training"*. It
was not. Reproduced exactly by
`tests/test_splits.py::test_reproduces_original_rng_contamination`:

```
NumPy-split test set vs PyTorch-split test set: 108/630 shared (17.14%)
PyTorch-split 'test' samples that were in NumPy-split training: 435/630 (69.05%)
```

### Defect 4b — subject-dependent splitting

Both implementations permuted **pooled samples**, so every one of the 42 subjects
appeared in train, validation and test:

```
pooled random split      : 42 of 42 test subjects also appear in training
subject-independent split:  0 of  8 test subjects also appear in training
```

For EEG this matters more than for most modalities: the subject-specific
component of the signal is large, and a model can score well on a pooled split by
recognising the person rather than the emotion. A pooled score does not support
the claim an affective-computing system needs to make — that it works on someone
it has not seen.

### Fix

`src/data/splits.py` is now the single split implementation, used by every
script. `subject_independent_split` is the default and holds out whole subjects.
`pooled_random_split` is retained **only** so the inflated figure can be
reproduced deliberately and reported alongside the honest one.
`SplitResult.__post_init__` raises if any index appears in two partitions.

`scripts/evaluate_model.py` no longer derives a split at all: it reads the split
recorded in the training run's `results.json` and refuses to run if the
reconstruction disagrees with what was recorded.

---

## 5. Silent noise substitution

`_load_eeg` returned `np.random.randn(28, 200)` on **any** exception while
keeping the real emotion label; audio failures became zero tensors. Corrupt
inputs were indistinguishable from valid ones, so the fraction of pure noise in
any reported result is unknowable after the fact.

All such fallbacks are removed. `EAVDataError` is raised instead, and
`load_subject_segments` additionally rejects non-finite values. The preprocessing
cache is built once, up front, so failures surface at build time rather than
silently mid-epoch.

---

## 6. The API could not load the model

`src/inference.py` defined `EmotionRecognitionModel` as a placeholder — two
encoders plus `Linear(256, 128)` — and called `load_state_dict` on it with a
checkpoint written by the cross-modal attention architecture. The parameter names
could never match.

`create_app` caught the exception and continued with `predictor = None`, so every
prediction endpoint returned 503. The `/model-info` endpoint meanwhile reported a
hardcoded `'accuracy': '78.57%'` and `'eeg_channels': 28` regardless of what was
loaded.

**Fix.** `EmotionRecognitionModel` is now the real architecture, reconstructed
from metadata stored in the checkpoint. Both the current unified checkpoint
format and the legacy four-module format load, the latter flagged with a warning.
`/model-info` reports what is actually loaded. Startup fails loudly on a bad
checkpoint unless `allow_missing_model=True`. Missing audio is a 400 rather than
a zero tensor, which previously yielded a confident-looking prediction from half
a model. `tests/test_inference.py` round-trips a real checkpoint through the
predictor.

---

## What the corrected dataset looks like

| | Original | Corrected |
| --- | --- | --- |
| Multimodal samples | 4,200 | 4,200 |
| Distinct EEG recordings | **42** (one per subject) | **4,200** |
| EEG tensor | `(28, 200)`, one 2 ms time-point | `(30, 2500)`, a 20 s trial at 125 Hz |
| EEG channels used | 28 of 30 | 30 of 30 |
| Label source | audio filename substring | `label.mat` ground truth |
| Class balance | 840 per class | 840 per class |
| Split | pooled, two conflicting RNGs | subject-independent, one implementation |
| Load failure | random noise, real label | `EAVDataError` |

The sample count is unchanged; what changed is that the samples now contain the
data they claimed to.

Preprocessing (`scripts/preprocess_eav.py`, ~5 minutes for all 42 subjects)
band-passes EEG to 0.5–45 Hz and decimates 500 Hz → 125 Hz, and converts audio to
13 MFCCs at 16 kHz. The cache is ~2.8 GB and holds all 200 trials per subject, so
the 8,400 EEG trials are available for EEG-only work even though only the 4,200
*Speaking* trials have matched audio.

---

## Relationship to previously reported numbers

**Every accuracy figure produced before 2026-08-08 is withdrawn**, including the
78.57% previously described in `docs/CHANGELOG.md` as "the only headline accuracy
in this project supported by an uncontaminated held-out evaluation." That
statement addressed defect 4a only. It remains true that 78.57% was measured on a
split free of the RNG mismatch — but that split was still subject-dependent
(defect 4b), and the EEG stream feeding it was degenerate (defects 1 and 2).

The corrected results are recorded in `docs/CHANGELOG.md` under *Corrected
results*.
