# MSE-CAPSTONE-PROJECT

Multimodal emotion recognition from EEG and speech, using cross-modal attention
fusion. Capstone project for the FPT Master of Software Engineering.

> **⚠️ Results before 2026-08-08 are withdrawn.** An audit found four defects in
> the data pipeline, the most serious being that all 100 samples from a subject
> shared one identical EEG tensor — so every "multimodal" result was in substance
> an audio-only result. See **[docs/DATA_CORRECTIONS.md](docs/DATA_CORRECTIONS.md)**.
> Do not cite 78.57%, 82.06%, or 84.44%.

## 🚀 Quick Start

Requires **Python 3.11+**. Full instructions in **[docs/SETUP.md](docs/SETUP.md)**.

```bash
git clone https://github.com/ttuan8600/MSE-CAPSTONE-PROJECT.git
cd MSE-CAPSTONE-PROJECT

python -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .                    # required so `import src...` resolves

pytest                              # expect 74 passed
```

> **A fresh clone contains code only.** The datasets (~47 GB) and model
> checkpoints are excluded from git and must be transferred separately — see
> [docs/SETUP.md](docs/SETUP.md#3-datasets-not-in-git). The test suite passes
> without them.

## 📁 Data

- **EAV** — 42 subjects × 200 trials of synchronised EEG (30 ch @ 500 Hz, 20 s),
  audio and video. 100 trials per subject are *Speaking* and carry matched audio,
  giving **4,200 multimodal samples**, exactly balanced across 5 emotions.
  Ground-truth labels come from the corpus's own per-trial label matrix.
- **FACED** — 123 subject files, available for EEG pre-training (not used in the
  reported results).

Training reads a preprocessed cache, not the raw files. Build it once:

```bash
python scripts/audit_eav_alignment.py   # verify the corpus first
python scripts/preprocess_eav.py        # ~5 min, writes ~2.8 GB
```

## 🎯 Results

**[docs/CHANGELOG.md](docs/CHANGELOG.md) is the single source of truth.** Every
number is traced to an artifact under `outputs/`.

All results are **subject-independent**: 8 of 42 subjects held out entirely, so
the test score measures generalisation to people the model has never seen. Chance
is 20.0%.

Headline figures come from **7-fold subject-wise cross-validation** — every one of
the 42 subjects held out exactly once, giving 4,200 pooled held-out predictions.

| Approach | Fusion params | CV accuracy | vs audio | p |
| --- | --- | --- | --- | --- |
| EEG only (band power) | — | 45.32% ± 0.92 | −17.45pp | <0.0001 |
| Sequence attention fusion | 8,064 | 63.19% | — | ❌ |
| Adversarial end-to-end fusion | 8,064 | 63.60% | — | ❌ |
| Trained attention fusion | 8,064 | 64.12% | −0.36pp | 0.7238 ❌ |
| Audio only (log-mel + SpecAugment) | — | 64.48% | — | |
| **Late fusion: mean of probabilities** | **0** | **67.02% ± 1.11** | **+4.25pp ± 0.41** | **0.0031** ✅ |
| *Oracle (unattainable)* | — | *81.69%* | *+17.21pp* | |

**The best model averages two independently trained unimodal models and has no
fusion parameters at all.** It beats audio alone by **+4.25pp ± 0.41** across
three seeds (paired, p = 0.0031), is positive on every seed, and wins on 32 of
42 subjects. Reproduce with `scripts/cross_validate_late_fusion.py`, then
`scripts/aggregate_seeds.py`.

> Accuracies here are means over 3 seeds. Single-seed figures for this system
> ranged 66.12--68.26%; a 69.55% figure briefly reported on 2026-08-09 was seed
> 42 alone and is withdrawn. Quote the paired *difference* (±0.41) rather than
> the absolute accuracy (±1.11) — pairing cancels the shared fold draw.

**The fewer parameters the combiner has, the better it transfers.** Compared
against the audio model from its own run — the only valid comparison, since the
runs' audio baselines differ by more than the effect — the equal mean is the best
rule under both EEG encoders (+5.19pp, +5.07pp) and the 8,064-parameter attention
module is the worst under both (+1.83pp, −0.36pp). The ordering of the 1- and
5-parameter rules reverses between runs and is not claimed. When the failure mode
is transfer to unseen people, a combiner with parameters fits the *training
subjects'* modality preferences. See
[docs/CHANGELOG.md](docs/CHANGELOG.md#the-ordering-is-the-finding).

> ⚠️ **Two independent noise estimates put the floor near 1pp.** Three runs with
> identical seed and config gave 63.41% / 62.29% / 64.48%, differing only in CPU
> thread scheduling; one EEG configuration varies ±0.92pp across 4 seeds. Pin
> `OMP_NUM_THREADS` and set `OMP_DYNAMIC=FALSE`, and **treat any margin under
> ~1pp as unmeasurable** — that includes the +0.71pp once claimed for attention
> fusion and the +0.64pp once claimed for adversarial EEG training.

### The headline finding

**EEG contributes +4.25pp ± 0.41 (p = 0.0031) — but only when combined by
averaging, not by a learned fusion module.** The architecture was the problem,
not the modality.

The section below records the earlier negative result, which still holds *of the
trained attention architecture* (−0.36pp, p = 0.7238) and which motivated the
diagnosis that led to the working configuration above.

**Adding EEG to audio via cross-modal attention produces no measurable change in
*mean* accuracy — and not because EEG is redundant.**

| Comparison | Δ | 95% CI | p |
| --- | --- | --- | --- |
| Audio vs EEG | +17.50pp | [+11.52, +22.98] | < 0.0001 ✅ |
| Fusion vs EEG | +18.21pp | [+14.76, +21.41] | < 0.0001 ✅ |
| **Fusion vs Audio** | **+0.71pp** | **[−4.02, +5.74]** | **0.4617** ❌ |

Per subject it is a coin flip: audio wins 23 of 42, fusion 18, one tie. This held
across every protocol tested — matched-budget single partition (p = 1.00) and
cross-validation (p = 0.46).

### Why the flat mean is misleading

`python scripts/analyze_complementarity.py` conditions on audio's errors, which
separates "EEG is redundant" from "EEG is complementary but unexploited":

| Quantity | Value |
| --- | --- |
| Trials audio gets wrong | 1,537 of 4,200 |
| **EEG correct on exactly those trials** | **47.50%** (95% CI [45.01, 49.99]) |
| Chance | 20.0% |
| **Oracle ceiling** (either model correct) | **80.79%** — +17.38pp over audio |

**EEG is 2.4× chance precisely where audio fails.** The current fusion recovers
791 of audio's errors and breaks 761 of its correct answers — net **+30 trials
out of 4,200**, which *is* the +0.71pp. It is a near-1:1 trade, not an absence of
signal.

Per class the trade is far from uniform:

| Emotion | EEG | Audio | Fusion | Fusion − Audio |
| --- | --- | --- | --- | --- |
| Neutral | 49.4% | 70.1% | 65.5% | −4.6pp |
| Anger | 58.8% | 74.0% | 68.7% | −5.3pp |
| Calmness | 29.0% | 56.8% | 52.7% | −4.1pp |
| Sadness | 29.3% | 71.4% | 64.2% | −7.2pp |
| **Happiness** | **63.0%** | **44.6%** | **69.5%** | **+24.9pp** ✅ |

**Happiness is the case multimodal fusion was proposed for.** It is the one
emotion where EEG beats audio — and there fusion beats *both* unimodal models by
24.9 points. The neural channel compensates for the weakest vocal cue in the
corpus. On the other four classes audio is already strong and fusion costs 4–7
points, which averages the aggregate back to zero.

### Why single partitions were not enough

| Model | Single 8-subject partition | 7-fold CV | Error |
| --- | --- | --- | --- |
| Audio | 67.13% | 63.40% | **−3.73pp** |
| Fusion | 62.62% | 64.12% | +1.50pp |

A single partition overstated audio by nearly 4 points and understated fusion,
reversing the ranking. Per-subject accuracy spans **25% to 81%**, which is why.
Quote the cross-validated numbers.

What the ablations do establish:

- **EEG carries real emotion signal** — 35.75% vs 20% chance. The old pipeline was
  structurally incapable of showing this, because its EEG stream was constant.
- **Audio dominates** — +19.00pp over EEG alone (p < 0.0001).
- **The bottleneck is cross-subject transfer, not fusion capacity.** The EEG model
  scores 6.42pp *lower* on unseen subjects than on validation subjects; audio
  shows the reverse. The EEG latent varies more by person than by emotion.
- **Evaluation protocol dominates architecture.** The identical model scores
  **68.25%** under a pooled random split (all 42 subjects in every partition) —
  a **+12.75pp** inflation, roughly seventeen times the fusion margin above.
  This is why the old 78.57% is not comparable to the 55.50% reported here.

### Reproduce everything

```bash
python scripts/run_ablations.py --set baseline   # original representations, ~30 min
python scripts/run_ablations.py --set improved   # log-mel + band-power, ~50 min
python scripts/cross_validate.py --modality audio --audio-features mel --specaugment
python scripts/compare_cv.py outputs/cv_*        # McNemar + by-subject bootstrap
python scripts/generate_result_figures.py        # figures, from artifacts only
```

## 🏗️ Model of record

**Audio-only**, 510,917 parameters, 1.95 MB, CPU-only inference.

| Component | Params | Shape |
| --- | --- | --- |
| Log-mel encoder (3 strided blocks) | 444,352 | `(64, 1313)` → 128-d |
| Emotion classifier | 66,565 | 128-d → 5 |

Fusion measured 64.12% against audio's 63.40% — not significant (p = 0.46) —
while being **3.4× less stable across folds**, 66% larger, and requiring a
30-channel EEG cap at inference. When two models are statistically tied, the
simpler and cheaper one wins.

## 🎬 Demonstration

No EEG amplifier is needed to demonstrate the multimodal system. Replay subjects
the model has never seen:

```bash
python scripts/demo_replay.py                    # step through 20 trials with Enter
python scripts/demo_replay.py --auto --delay 2   # hands-free, for a screen recording
```

Each trial shows the ground-truth emotion, the multimodal model's prediction with
its full probability distribution, and the audio-only model's prediction beside
it — so the central finding is visible live: the two mostly agree, and where they
disagree neither is reliably right.

Subjects 5, 19, 22, 26, 27, 30, 40 and 42 were excluded from training and
validation entirely. Replaying all 800 of their trials through this script
reproduces the recorded test accuracies exactly (62.63% multimodal, 67.13%
audio), confirming the demo path and the evaluation path are the same code.

> **Live microphone capture is deliberately not offered.** EAV participants are
> Korean speakers reading scripted prompts under studio conditions; a presenter
> speaking English into a laptop microphone is a different distribution, and
> there is no ground truth for the presenter's own emotional state. A wrong
> prediction would be uninterpretable rather than informative.

## 🔌 Serving the model

```bash
pip install -r requirements_api.txt
python app.py --model outputs/model_of_record.pt --port 5000
```

```python
import requests, numpy as np
r = requests.post("http://localhost:5000/predict", json={
    "audio": np.random.randn(64, 1313).tolist(),   # 64-band log-mel, 16 kHz, 16 ms hop
})
print(r.json())   # {"emotion": ..., "confidence": ..., "probabilities": {...}}
```

Every modality the loaded checkpoint needs is mandatory — a missing one returns
HTTP 400 rather than silently substituting zeros. Query `/model-info` for the
input contract of whatever checkpoint is loaded.

## 📚 Documentation

- **[docs/DATA_CORRECTIONS.md](docs/DATA_CORRECTIONS.md)** — what was wrong, the
  evidence, and what changed. Read this before citing any older number.
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — results of record and experiment history
- [docs/SETUP.md](docs/SETUP.md) — environment, datasets, preprocessing
- [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) — serving the model
- [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) — REST reference
- [scripts/legacy/README.md](scripts/legacy/README.md) — superseded experiment
  scripts, retained for provenance only

Superseded status documents are in [docs/archive/](docs/archive/). They contain
the withdrawn figures and should not be cited.

## 🔬 Scope

**Not implemented:** GAN augmentation (descoped — see the report methodology
chapter), video modality, ICA artefact removal. Earlier documentation claimed
some of these; none were implemented at the time they were claimed.
