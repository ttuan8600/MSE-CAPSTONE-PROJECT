# Project Changelog & Results of Record

**Single source of truth** for what the current model is and what it actually scores.
Every number here is traced to a result artifact under `outputs/`. Where a
frequently-quoted number is *not* reproducible from an artifact, it is listed in
[Disputed and superseded numbers](#disputed-and-superseded-numbers) with the reason.

Last verified: 2026-08-08

---

## Current model of record

| | |
| --- | --- |
| **Architecture** | Cross-Modal Attention Fusion (4-head), EEG + audio MFCC |
| **Checkpoint** | `outputs/attention_fusion_model_best.pt` |
| **Held-out test accuracy** | **78.57%** |
| **Validation accuracy** | 75.87% (best epoch 36 of 40) |
| **Loss** | Focal Loss (gamma = 2.0) |
| **Training script** | `scripts/train_attention_fusion.py` |
| **Artifact** | `outputs/attention_fusion_20260401_182606/results.json` |
| **Dataset** | EAV, 4,200 samples, 70/15/15 split (2,940 / 630 / 630) |

Per-class accuracy on the held-out test set:

| Emotion | Accuracy |
| --- | --- |
| Sadness | 88.46% |
| Anger | 84.62% |
| Calmness | 79.13% |
| Neutral | 72.73% |
| Happiness | 67.48% |

**78.57% is the only headline accuracy in this project that is supported by an
uncontaminated held-out evaluation.** Use it.

---

## Known measurement issue: train/test contamination

The fine-tuning and comparison stages of 2026-04-05 evaluated on a split that
overlaps the training data. Numbers derived from them are invalid.

**Cause.** Two scripts split the dataset with the same seed value but different
random number generators, which produce unrelated permutations:

| Script | Split method |
| --- | --- |
| `scripts/train_attention_fusion.py:87` | `np.random.seed(42)` + `np.random.permutation` (NumPy RNG) |
| `scripts/evaluate_finetuned_model.py:155` | `random_split(generator=torch.Generator().manual_seed(42))` (PyTorch RNG) |
| `scripts/finetune_attention_fusion.py:241` | `random_split(generator=torch.Generator().manual_seed(42))` (PyTorch RNG) |

`evaluate_finetuned_model.py` carries the comment "same split as training". It is not.

**Measured impact** (reproduce with the snippet in [Verifying the contamination](#verifying-the-contamination)):

- The evaluation script's 630-sample "test set" shares only **108 samples (17.1%)**
  with the training script's true test set.
- **435 of its 630 samples (69.0%)** were in the model's training set.
- This inflates the reported figure by roughly 6 percentage points
  (78.57% clean, 84.92% contaminated, same checkpoint).

**Consequence for the fine-tuned model.** Fine-tuning trained on the PyTorch split,
which includes **432 of the 630 samples (68.6%)** of the baseline's held-out test set.
No uncontaminated evaluation set therefore remains for
`attention_fusion_finetuned_best.pt`. Its true generalization accuracy is **unmeasured**,
not 82.06% and not 84.44%.

**Note.** Even on the contaminated split, fine-tuning did not help: the comparison
artifact records baseline 84.92% vs fine-tuned 84.44%, an improvement of
**-0.48pp**, with `"status": "UNCHANGED"`. The claim of a +3.49pp gain from
fine-tuning is not supported by any artifact in this repository.

**To fix**, make both scripts derive the split from one shared, seeded helper, retrain
or re-evaluate, and re-measure. Until then, report 78.57%.

---

## Verified results

Every row traced to an artifact. Ordered chronologically.

| Date | Run | Test acc | Val acc | Artifact |
| --- | --- | --- | --- | --- |
| 2026-02-28 | EEG-only vs EEG+Audio comparison | n/a | n/a | `comparison_20260228_230047.json` — **no data produced** (0 batches) |
| 2026-03-12 | Optimized training (gated) | n/a | n/a | `optimized_training_20260312_231327/metrics.json` — **failed run** (NaN loss, 0.0 acc) |
| 2026-03-16 | Comparison rerun | n/a | n/a | `comparison_20260316_031730.json` — **no data produced** (0 batches) |
| 2026-03-22 | Gated fusion (CNN encoder) | 52.22% | 52.70% | `finetuned_final_20260322_132618/results.json` |
| 2026-03-25 | LSTM encoder + gated fusion | 49.21% | 54.13% | `lstm_enhanced_20260325_095128/results.json` |
| 2026-03-29 | CNN encoder + Focal Loss | 63.02% | 65.87% | `focal_loss_20260329_073014/results.json` |
| **2026-04-01** | **Cross-modal attention fusion** | **78.57%** | **75.87%** | `attention_fusion_20260401_182606/results.json` |
| 2026-04-01 | Ensemble (0.7 attention + 0.3 focal) | 73.70% | — | `ensemble_results/ensemble_metrics.json` |
| 2026-04-05 | Fine-tuning + SpecAugment/EEG jitter | **invalid** | — | no `results.json`; see contamination section |

Headline progression, using only clean measurements:
**52.22% → 63.02% → 78.57%** (+26.35pp from gated-fusion baseline to attention fusion).

The ensemble scored **below** the attention model alone (73.70% vs 78.57%) and was
correctly not deployed.

---

## Disputed and superseded numbers

| Number | Where it appears | Status |
| --- | --- | --- |
| **78.57%** | 16 documents | ✅ **Correct.** Clean held-out test accuracy. |
| **82.06%** | README, DEPLOYMENT_CHANGELOG, DEPLOYMENT_GUIDE, FINETUNING_RESULTS_SUMMARY, MODEL_PERFORMANCE_COMPARISON, SUBMISSION_STATUS | ❌ **Not a measurement.** Hardcoded literal at `scripts/deploy_finetuned_model.py:69`. Labelled "validation" in `evaluate_finetuned_model.py:190`. No artifact contains it. |
| **84.44%** | SUBMISSION_STATUS | ❌ **Contaminated.** Fine-tuned model on the leaked split (69% training data). |
| **84.92%** | comparison artifact only | ❌ **Contaminated.** Baseline on the leaked split. Same checkpoint scores 78.57% clean. |
| **+3.49pp** | README, deployment docs | ❌ **Unsupported.** Derived from 78.57 → 82.06, a clean number minus a hardcoded one. The one real comparison shows **-0.48pp**. |
| **96.85% UAR** | `context.md` | ⚠️ **Aspirational target** from the original proposal, never approached. Best achieved is 78.57%. |
| **52.22%** | various | ✅ Correct — gated-fusion baseline, not a CNN-only baseline. |

---

## Chronological changelog

### 2026-08-08 — Documentation consolidation
Consolidated ~43 overlapping root status documents into this file. Superseded
documents moved to `docs/archive/`. Audited every accuracy claim against
`outputs/` artifacts; found and documented the train/test contamination above.

### 2026-08-07 — Repository hygiene
Purged 191 MB of model checkpoints and 18 private report documents from git
history (`.git`: 181 MB → 1.2 MB). Files retained locally, untracked. Removed two
dead root test scripts. Fixed dead README links.

### 2026-04-05 — Fine-tuning (results invalid)
Fine-tuned with SpecAugment and EEG jitter (sigma=0.01) at lr=1e-4. Deployed
`attention_fusion_finetuned_best.pt` to production on the basis of a claimed
+3.49pp gain. That gain is not supported by any artifact; the comparison that ran
20 minutes after deployment recorded -0.48pp on a contaminated split.
**Recommendation:** treat `attention_fusion_model_best.pt` (78.57%) as the model of
record until a clean re-evaluation exists.

### 2026-04-04 — REST API
Flask API added (`app.py`): `/predict`, `/batch-predict`, `/health`, `/emotions`,
`/model-info`. **Known defect:** `src/inference.py` loads a placeholder
concat-and-linear model, not the attention-fusion architecture, so its
`load_state_dict` cannot consume the production checkpoint. The API does not
currently serve the model of record.

### 2026-04-01 — Cross-modal attention fusion (breakthrough)
4-head multi-head attention with bidirectional EEG↔audio fusion, ~920K parameters.
**78.57% test accuracy**, +15.55pp over the focal-loss CNN. Current model of record.
An ensemble with the focal-loss CNN scored lower (73.70%) and was not adopted.

### 2026-03-29 — Focal loss
Focal Loss (gamma=2.0) on the CNN encoder: 63.02%, +10.80pp over gated fusion.

### 2026-03-25 — LSTM encoder
LSTM variant: 49.21%, below the CNN gated baseline. Not pursued.

### 2026-03-22 — Gated fusion baseline
CNN encoder with gated fusion, 20 epochs: 52.22%. Calmness collapsed to 15.25%,
motivating the move to focal loss.

### 2026-03-10 → 03-16 — Training optimization (largely failed)
Multiple pre-train and optimization runs. The 2026-03-12 optimized run produced NaN
losses and 0.0 accuracy; the 02-28 and 03-16 comparison runs processed 0 batches.
Any figures quoted from these runs in archived documents are not real.

### 2026-02-25 → 02-28 — Baseline and fusion scaffolding
EEG encoder (4-layer 1D-CNN), audio encoder (MFCC CNN), three fusion modes
(concat / cross-attention / gated), 5-class emotion classifier. Test suite established.

---

## Verifying the contamination

```python
import numpy as np, torch
N, tr, va = 4200, 2940, 630
np.random.seed(42); npperm = np.random.permutation(N)
train_np, test_np = set(npperm[:tr].tolist()), set(npperm[tr+va:].tolist())
tperm = torch.randperm(N, generator=torch.Generator().manual_seed(42)).numpy()
test_torch = set(tperm[tr+va:].tolist())

print(len(test_np & test_torch))   # 108  -> the two "test sets" barely overlap
print(len(test_torch & train_np))  # 435  -> 69% of the eval set was trained on
```

---

## Open gaps

1. **GAN augmentation never implemented.** `src/gan/` contains only an empty
   `__init__.py`, though `pyproject.toml` and `context.md` describe the project as
   GAN-based (research objective RO1). No reported result uses GAN augmentation.
2. **API serves the wrong architecture** (see 2026-04-04 above).
3. **Silent data-quality masking.** `EAVMultimodalDataset._load_eeg` returns
   `np.random.randn(28, 200)` on any `.mat` load failure, keeping the real label
   (`src/preprocessing/data_loader.py:366-370`). Audio failures become zero tensors.
   The fraction of noise samples in reported results is unknown.
4. **Video modality is a no-op.** `_load_video_stub` always returns `None`.
5. **No dependency pinning and no CI**, so none of the above is reproducible by
   version or guarded against regression.

---

## Superseded documentation

Documents in `docs/archive/` are retained for provenance. **They contain the
disputed numbers above and should not be cited.** This file supersedes them.
