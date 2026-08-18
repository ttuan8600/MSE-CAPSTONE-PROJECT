# Project Changelog & Results of Record

**Single source of truth** for what the current model is and what it actually scores.
Every number here is traced to a result artifact under `outputs/`.

Last verified: 2026-08-10

> **All results produced before 2026-08-08 are withdrawn.** Four defects in the
> data pipeline — documented in [DATA_CORRECTIONS.md](DATA_CORRECTIONS.md) — meant
> that every "multimodal" experiment was in substance an audio-only experiment,
> measured on a partition sharing all 42 subjects with training. This includes
> the 78.57% figure that a previous revision of this file described as the only
> trustworthy number in the project.

---

## Fusion that works (2026-08-09, revised 2026-08-10 after multi-seed repetition)

**Headline: 67.02% ± 1.11 under 7-fold subject-wise cross-validation repeated
across 3 seeds, +4.25pp over audio alone (paired, p = 0.0031).** Previous best
was 64.12%.

The winning configuration is a **zero-parameter average** of two independently
trained unimodal models. Artifacts: `outputs/multiseed_summary.json`,
`outputs/late_fusion_cv_final.json`, `scripts/cross_validate_late_fusion.py`,
`scripts/aggregate_seeds.py`.

> ⚠️ **A single-seed figure of 69.55% was reported here on 2026-08-09 and is
> withdrawn.** It was seed 42, the best of the three seeds subsequently run. The
> same document had, one day earlier, recorded that single-run figures are not
> trustworthy; the multi-seed repetition below was run because that lesson
> applies to this project's own new results as much as to its inherited ones.
> See [multi-seed repetition](#multi-seed-repetition-2026-08-10).

### Every approach tested

All figures pooled over 4,200 held-out predictions, 42 subjects. Paired McNemar,
bootstrap resampled by subject, compared against the same audio run.

| Approach | Params (fusion) | CV accuracy | vs audio | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| EEG only (adversarial) | — | 46.55% | −17.93pp | [−23.14, −12.36] | <0.0001 |
| Sequence attention fusion | 8,064 | 63.19% | — | — | ❌ |
| Adversarial end-to-end fusion | 8,064 | 63.60% | — | — | ❌ |
| Trained attention fusion | 8,064 | 64.12% | −0.36pp | [−5.26, +4.74] | 0.7238 ❌ |
| Audio only | — | 64.48% | — | | |
| Max-confidence gating | 0 | 67.71% | +3.24pp | [+1.31, +5.29] | <0.0001 ✅ |
| Mean, standard EEG | 0 | 68.26%† | +3.79pp† | — | ✅ |
| Per-class weights (LOFO) | 5 | 68.90% | +4.43pp | [+2.17, +6.71] | <0.0001 ✅ |
| Weighted (LOFO) | 1 | 69.31% | +4.83pp | [+2.67, +7.10] | <0.0001 ✅ |
| Mean, adversarial EEG | 0 | 69.55%† | +5.07pp† | [+2.81, +7.40] | <0.0001 ✅ |
| **Mean of probabilities (recommended)** | **0** | **67.02% ± 1.11** | **+4.25pp ± 0.41** | — | **0.0031** ✅ |
| *Oracle (unattainable)* | — | *81.69%* | *+17.21pp* | | |

Fusion is better on **32 of 42 subjects**, worse on 9, tied on 1 — a broad
effect, not a few outliers.

† Single-seed (seed 42) figures, retained for the per-rule ranking, which was
computed on one seed. Absolute accuracies should be read from the multi-seed
table below; the *ordering* of the rules is what this table establishes.

### The ordering is the finding

**Read this before quoting the table above across rows.** The `†` rows come from
different runs, and the two runs' audio baselines differ by 2.19pp at the same
configuration — inside the noise floor. Absolute accuracies are therefore *not*
comparable across those rows. The valid comparison is each rule's paired
advantage over the audio model from its **own** run:

| Rule | Fusion params | Standard EEG | Adversarial EEG |
| --- | --- | --- | --- |
| **Mean of probabilities** | **0** | **+5.19pp** | **+5.07pp** |
| Weighted average (LOFO) | 1 | +4.26pp | +4.83pp |
| Per-class weights (LOFO) | 5 | +4.69pp | +4.43pp |
| Max-confidence gating | 0 (hard switch) | +2.76pp | +3.24pp |
| Trained attention fusion | 8,064 | +1.83pp | −0.36pp |

Artifacts: `outputs/late_fusion_cv.json` (standard),
`outputs/late_fusion_cv_final.json` (adversarial).

**What replicates across both encoders, and is the finding:**

> The equal mean is the **best** rule under both. The 8,064-parameter attention
> module is the **worst** under both, and every combiner with ≤5 free parameters
> beats it under both.

**What does not replicate, and must not be claimed:** the ordering of the
1-parameter and 5-parameter rules reverses between encoders (+4.26 vs +4.69, then
+4.83 vs +4.43). They sit less than half a point apart, inside the noise floor.
The endpoints of the ordering are established; the middle is not.

When the failure mode is transfer to unseen people, a combiner with parameters
fits the *training subjects'* modality preferences and does not carry them
across. Direct evidence, independent of any ranking: on the single partition a
weight fitted on the validation subjects scored 69.00% there and **64.88%** on
test — below the untuned mean. One free parameter overfitted six subjects.

### Subject-adversarial training: no measurable benefit

⚠️ **Retracted.** This section previously claimed +0.64pp solo and +1.29pp inside
the fusion, from seed 42 alone. Neither survives repetition:

| Claim | Seed 42 only | Across seeds | p | Verdict |
| --- | --- | --- | --- | --- |
| Adversarial EEG, solo | +0.64pp | **+0.18pp ± 0.82** (n=4) | 0.6950 | ❌ |
| Adversarial EEG, inside fusion | +1.29pp | **+0.92pp ± 0.61** (n=3) | 0.1211 | ❌ |

Per-seed, solo: $-0.91$, $+0.02$, $+0.95$, $+0.65$pp — the sign flips. Seed 42
was the favourable draw.

Applied **end-to-end to the fusion model** it is worse still (63.60% vs 64.12%),
though that too is a single run and inside the noise band.

**The subject-adversarial encoder is therefore not part of the recommended
system.** It remains implemented and documented because the negative result is
informative: making the EEG representation subject-invariant, at least by
gradient reversal at this scale, is not what unlocks the fusion gain. The gain
comes from the combination rule.

### Multi-seed repetition (2026-08-10)

Every configuration had been trained once, at seed 42, so each difference was
confounded with initialisation variance. Configurations were repeated across
seeds; `scripts/aggregate_seeds.py` aggregates them.

Changing the seed changes **both** the fold assignment and the initialisation,
because `subject_kfold` takes the same seed. The spread below is therefore the
variability of the whole procedure — the quantity that answers "would this
margin survive re-running the experiment?" Comparisons *within* a seed remain
exactly paired, since audio and EEG at seed $s$ share seed $s$'s folds.

| Configuration | Mean ± sd | Per-seed |
| --- | --- | --- |
| Audio only | 62.77% ± 1.52 | 61.55, 62.29, 64.48 |
| **Late fusion (standard EEG)** | **67.02% ± 1.11** | 66.12, 66.69, 68.26 |
| Late fusion (adversarial EEG) | 67.94% ± 1.61 | 66.33, 67.95, 69.55 |
| EEG only (standard) | 45.32% ± 0.92 | (n=4) |
| EEG only (adversarial) | 45.49% ± 0.79 | (n=4) |

| Paired difference | Result | p | Verdict |
| --- | --- | --- | --- |
| **Late fusion (std) − audio** | **+4.25pp ± 0.41** | **0.0031** | ✅ |
| Late fusion (adv) − audio | +5.17pp ± 0.45 | 0.0025 | ✅ |
| Adversarial gain in fusion | +0.92pp ± 0.61 | 0.1211 | ❌ |
| Adversarial gain solo | +0.18pp ± 0.82 | 0.6950 | ❌ |

**The late-fusion result survives**: positive on every seed, worst case
$+3.79$pp. Note the paired difference is far more stable ($\pm 0.41$) than the
individual accuracies ($\pm 1.5$), because pairing cancels the shared fold draw
— which is why differences, not absolute accuracies, are the quantity to quote.

Because the adversarial gain is not significant, the **recommended system is the
simpler one**: mean late fusion with the standard band-power EEG encoder, at
67.02% ± 1.11. It also has the tightest fold spread of the three.

Not repeated: the trained attention and sequence-fusion configurations, at
2--3.4 h per run. Both are null results already, and a null does not become more
null with more seeds; nothing in the conclusions rests on their precision.

### The noise floor, from two independent estimates

| Source | Spread |
| --- | --- |
| CPU thread scheduling, identical seed | 2.2pp range across 3 runs |
| Seed variation, one EEG configuration | ±0.92pp (n=4) |

**Treat any margin under roughly 1 percentage point on this hardware as
unmeasurable.** Both the originally claimed +0.71pp attention-fusion advantage
and the +0.64pp adversarial advantage fall inside it. The +4.25pp late-fusion
gain does not.

### This supersedes the earlier "fusion does not help" finding

The retracted claim was that adding EEG produces no measurable change. That
remains true **of the trained attention architecture** (−0.36pp, p = 0.7238).
It is false of the modality: the same EEG stream, combined by averaging, is
worth +4.25pp ± 0.41 across seeds (p = 0.0031). The complementarity analysis predicted this and was correct.

### ⚠️ CPU-threading nondeterminism, quantified

Three cross-validation runs, identical seed and configuration, differing only in
CPU thread scheduling:

| Run | Threads | Pooled accuracy |
| --- | --- | --- |
| original | 8, uncontended | 63.41% |
| probability capture | 8, contended | 62.29% |
| pinned | 6, pinned + `OMP_DYNAMIC=FALSE` | 64.48% |

**A 2.2-point spread from thread scheduling alone.** Training is bit-identical
at a *fixed* pinned thread count, and differs across thread counts because
parallel reduction order changes floating-point summation.

Consequences:

* Any margin below ~1pp on this hardware is not reproducible. The originally
  reported +0.71pp attention-fusion advantage and the +0.64pp adversarial
  advantage were both inside that band; the +4.25pp late-fusion gain is 3.9x
  outside it.
* **Pin `OMP_NUM_THREADS` and set `OMP_DYNAMIC=FALSE` on every run.**
* Paired comparisons computed on a *single* run's predictions are unaffected,
  which is why the significance tests above are computed that way.

### Single-partition checkpoints disagree, as expected

The deployable checkpoints trained on the standard 28/6/8 split give, on those 8
subjects: audio 67.13%, late fusion with standard EEG 69.00% (+1.88pp), late
fusion with the adversarial EEG checkpoint 67.00% (−0.12pp). That adversarial
checkpoint early-stopped at epoch 10 of 70 on a 6-subject validation signal and
is a weak instance.

The cross-validated figures describe the **procedure** over 42 subjects; a single
8-subject partition carries several points of uncertainty in either direction.
This is the same lesson recorded above, and no checkpoint was selected on test
performance.

## Current model of record

| | |
| --- | --- |
| **Architecture** | Audio-only: 64-band log-mel + 3-block 1-D CNN |
| **Parameters** | 510,917 (1.95 MB) |
| **Checkpoint** | `outputs/model_of_record.pt` |
| **Evaluation** | 7-fold subject-wise CV — every one of 42 subjects held out once |
| **Cross-validated accuracy** | **63.40%** (chance 20.0%) |
| **UAR** | **63.40%** |
| **Per-fold stability** | ±1.78pp |
| **Input** | audio only — `(64, 1313)` log-mel, 16 kHz, 16 ms hop |
| **Loss** | Focal Loss (gamma = 2.0), inverse-frequency alpha |
| **Regularisation** | SpecAugment (train only), dropout 0.3 |
| **Training script** | `scripts/train_attention_fusion.py --modality audio --audio-features mel --specaugment` |
| **Artifacts** | `outputs/cv_audio_mel_20260809_032321/cv_results.json`, `outputs/audio_mel_subject_independent_20260809_004702/` |
| **Dataset** | EAV, 4,200 samples, 42 subjects |

**No EEG is required at inference.** Fusion with EEG was measured at 64.12%,
which is not significantly different (p = 0.46), while being 3.4× less stable
across folds and requiring a 30-channel EEG cap. See
[Model of record](#model-of-record-audio-only) below.

---

## Corrected results

All runs share the same encoders, classifier, loss, optimiser, schedule, seed
(42) and subject-independent partition. Only the stated variable differs.

| Run | Modality | Split | Test acc | UAR | Macro F1 | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| EEG only | EEG | subject-independent | 35.75% | 35.75% | 34.64% | `eeg_subject_independent_20260808_151928/` |
| Audio only, 40 epochs | audio | subject-independent | 54.75% | 54.75% | 54.65% | `audio_subject_independent_20260808_152701/` |
| **Audio only, 100 epochs** | audio | subject-independent | **55.50%** | **55.50%** | 56.05% | `audio_long_20260809_001833/` |
| **Fusion** | EEG+audio | subject-independent | **55.50%** | **55.50%** | **56.15%** | `multimodal_subject_independent_20260808_150930/` |
| Fusion, pooled split | EEG+audio | pooled random | 68.25% | 68.43% | 68.10% | `multimodal_pooled_random_20260808_153016/` |

The last row is a **protocol ablation, not a result**. It uses the original
subject-dependent split, in which all 42 subjects appear in every partition. The
gap of **+12.75pp** over the same model evaluated subject-independently is the
size of the shortcut that pooled splitting permits — larger than any
architectural difference measured in this project.

Summary artifact: `outputs/ablation_summary.json`.

### Statistical testing

Paired McNemar exact test on the shared 800-sample test partition, with
percentile bootstrap confidence intervals (10,000 resamples, seed 12345).
Artifact: `outputs/significance.json`.

| Comparison | Difference | 95% CI | p | Significant |
| --- | --- | --- | --- | --- |
| Fusion vs EEG only | +19.75pp | [+16.00, +23.50] | < 0.0001 | ✅ yes |
| Audio only (40ep) vs EEG only | +19.00pp | [+14.12, +24.00] | < 0.0001 | ✅ yes |
| Fusion vs Audio only (40ep) | +0.75pp | [−3.63, +5.00] | 0.78 | ❌ no |
| **Fusion vs Audio only (100ep)** | **0.00pp** | **[−4.37, +4.38]** | **1.0000** | ❌ **no** |

Artifact for the last row: `outputs/significance_audio_long.json`.

### What this means

1. **EEG carries recoverable emotion information** — 35.75% against 20% chance.
   The previous pipeline could not have shown this: its EEG stream was constant
   within each subject.
2. **Audio dominates** — 55.50% alone (100 epochs), ~+20pp over EEG.
3. **Cross-modal attention fusion does not beat audio alone.** Under matched
   training budgets the difference is **exactly 0.00pp, p = 1.0000**. The +0.75pp
   margin measured at 40 epochs was an artifact of the audio model not having
   converged; extending it to 100 epochs eliminated the margin precisely. The
   previously claimed +15.55pp fusion gain was likewise an artifact.
4. **The bottleneck is cross-subject transfer.** EEG validation accuracy (42.17%)
   exceeds its test accuracy (35.75%) by 6.42pp; audio shows the opposite
   ordering. The EEG latent varies more by person than by emotion.
5. **Evaluation protocol matters more than architecture here.** Pooled splitting
   inflates the same model by 12.75pp — about seventeen times the fusion margin
   that was found insignificant.

### Caveats

- **Single seed.** Each configuration was trained once at seed 42. Differences
  between configurations are confounded with initialisation variance.
- **Convergence — tested and resolved.** The audio-only model was still improving
  at epoch 40. Retrained to 100 epochs it converged at epoch 62 and matched
  fusion exactly. The fusion model itself was not given a 100-epoch budget; it
  had peaked at epoch 21 of 40 and declined after, so a longer schedule is
  unlikely to help it, but for strict symmetry it should be run.
- **Eight test subjects.** Leave-one-subject-out validation is implemented
  (`src/data/splits.py`) but was not run; it is the highest-value next experiment.

---

## Improved representations (2026-08-09)

Two representation changes, each aimed at a failure the baseline ablations
measured. 100 epochs throughout. Artifact: `outputs/ablation_summary_improved.json`.

| Run | Params | Val | Test | UAR | Train | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| EEG, DE + Euclidean alignment | 190,897 | 50.83% | 46.00% | 46.00% | 74.28% | 23.4pp |
| **Audio, log-mel + SpecAugment** | 510,917 | 60.67% | **67.13%** | **67.12%** | 80.50% | 19.8pp |
| Fusion of both | 850,417 | **69.50%** | 62.62% | 62.62% | 90.73% | 21.2pp |

Against the baseline representations:

| Modality | Baseline test | Improved test | Δ |
| --- | --- | --- | --- |
| EEG | 35.75% | 46.00% | **+10.25pp** |
| Audio | 55.50% | 67.13% | **+11.63pp** |
| Fusion | 55.50% | 62.62% | +7.12pp |

Both interventions worked, and both reduced overfitting: the EEG gap fell from
57.6pp to 23.4pp, the audio gap from 31.3pp to 19.8pp.

### Significance on the improved runs

Artifact: `outputs/significance_improved.json`.

| Comparison | Difference | 95% CI | p | Significant |
| --- | --- | --- | --- | --- |
| Audio-mel vs EEG-DE | +21.12pp | [+16.13, +26.13] | < 0.0001 | ✅ yes |
| EEG-DE vs Fusion | −16.62pp | [−21.00, −12.25] | < 0.0001 | ✅ yes |
| **Audio-mel vs Fusion** | **+4.50pp** | **[+0.75, +8.37]** | **0.0237** | ✅ **yes** |

**Adding EEG now makes the model significantly worse on held-out subjects.** With
a genuinely informative EEG stream (46.00% vs 20% chance), fusion still fails to
help — and this time it measurably hurts.

### 7-fold subject-wise cross-validation (definitive)

Every subject takes a turn in a test fold, giving **4,200 pooled held-out
predictions across all 42 subjects** rather than 800 across eight. This resolves
the validation/test reversal described below. Artifacts:
`outputs/cv_*/cv_results.json`, `outputs/cv_comparison.json`.

| Model | CV pooled | UAR | Per-fold sd | Single partition | Δ |
| --- | --- | --- | --- | --- | --- |
| EEG-DE | 45.90% | 45.90% | ±2.31pp | 46.00% | −0.10pp |
| **Audio-mel** | **63.40%** | 63.40% | **±1.78pp** | 67.13% | **−3.73pp** |
| Fusion | **64.12%** | 64.12% | ±6.10pp | 62.62% | +1.50pp |

**The single-partition figures were unreliable in both directions** — audio was
overstated by 3.73pp, fusion understated by 1.50pp. Neither error was visible
without cross-validation.

#### Paired comparison on pooled predictions

McNemar exact test; bootstrap resampled **by subject** (samples within a subject
are not independent — per-subject accuracy spans ~50pp).

| Comparison | Δ | 95% CI | p | Per-subject wins | Significant |
| --- | --- | --- | --- | --- | --- |
| Audio-mel vs EEG-DE | +17.50pp | [+11.52, +22.98] | < 0.0001 | 35 – 7 | ✅ yes |
| Fusion vs EEG-DE | +18.21pp | [+14.76, +21.41] | < 0.0001 | 40 – 2 | ✅ yes |
| **Fusion vs Audio-mel** | **+0.71pp** | **[−4.02, +5.74]** | **0.4617** | **18 – 23** | ❌ **no** |

**Final answer on fusion: audio-only and fusion are statistically
indistinguishable.** Per subject it is a coin flip — audio wins 23 of 42, fusion
18, one tie.

⚠️ **This does not mean the EEG stream is redundant.** See
[complementarity](#eeg-is-complementary-not-redundant-2026-08-09) — a flat mean
hides a near-1:1 trade, not an absence of information.

This also retracts the intermediate claim that fusion was *significantly worse*
(−4.50pp, p = 0.0237). That result came from a single partition and does not
survive cross-validation either. The correct statement across every protocol
tested is that **adding EEG to audio produces no measurable change in accuracy**.

#### Model of record: audio-only

`outputs/model_of_record.pt` is the **audio-mel** model. The two candidates are
equal in accuracy, so the choice is made on the remaining criteria, none of which
involve the test set:

| | Audio-mel | Fusion |
| --- | --- | --- |
| CV accuracy | 63.40% | 64.12% (n.s.) |
| Per-fold stability | **±1.78pp** | ±6.10pp |
| Parameters | **510,917** | 850,417 |
| Input required | audio only | audio **+ 30-channel EEG** |

Fusion is **3.4× more variable across folds**, 66% larger, and requires an EEG
cap at inference — for no measurable accuracy gain. Selecting the simpler, more
stable, cheaper model when the two are statistically tied is parsimony, not
test-set selection.

The cited 63.40% is the accuracy of the **procedure** under cross-validation, not
a re-measurement of this particular checkpoint; the checkpoint itself was trained
on the standard 28-subject split.

#### Which number to report

"Model of record" is a **deployment** choice, not a claim about which number the
thesis reports. The two questions are separate and have different answers:

| Question | Answer |
| --- | --- |
| What accuracy does the proposed multimodal system achieve? | **64.12%** |
| Does the EEG stream contribute significantly to that? | **No** (+0.71pp, p = 0.46) |
| What should a practitioner deploy? | **Audio-only** — same accuracy, a third of the variance, no EEG cap |

All three are simultaneously true and none is in tension with the others. The
proposed system is the fusion architecture and 64.12% is its measured
cross-validated accuracy; the negative result concerns the *attribution* of that
accuracy to the EEG pathway, not the system's performance. Reporting 64.12% as
the system's accuracy while reporting the ablation as a finding is the standard
and honest presentation.

### EEG is complementary, not redundant (2026-08-09)

Artifact: `outputs/complementarity_analysis.json`, from
`scripts/analyze_complementarity.py`.

A flat mean difference admits two very different explanations, and they call for
opposite conclusions:

* **redundant** — EEG knows only what audio knows, so no fusion design can help;
* **complementary but unexploited** — EEG is informative where audio fails, gains
  and losses cancel, and a better fusion design *would* help.

Conditioning on audio's errors separates them.

| Quantity | Value |
| --- | --- |
| Trials audio gets wrong | 1,537 of 4,200 |
| **EEG correct on those trials** | **730 = 47.50%** (95% CI [45.01, 49.99]) |
| Chance | 20.0% |

**EEG is 47.50% accurate precisely where audio fails — 2.4× chance, with the
confidence interval nowhere near it.** The streams are complementary. The
information exists; the architecture is what fails to keep it.

*(730 is the same discordant cell as McNemar's `c` in the audio-vs-EEG
comparison above, and reconstructs EEG's 45.90% exactly: 28.52% both-correct +
730/4200.)*

#### Oracle upper bound

| | |
| --- | --- |
| At least one model correct | **80.79%** ← ceiling for any combiner of these two |
| Both correct | 28.52% |
| Neither correct | 19.21% |
| **Headroom over audio alone** | **+17.38pp** |

#### What the current fusion actually does

| | Trials |
| --- | --- |
| Audio errors it recovers | 791 (51.5% of them) |
| Audio-correct answers it breaks | 761 |
| **Net** | **+30 = +0.71pp** |

It captures 74.7% of the trials where EEG knew the answer and audio did not — the
routing largely works — but pays for them almost 1:1 elsewhere. **The +0.71pp is
a near-cancelling trade, not an absence of signal.**

#### Per class — the trade is not uniform

| Emotion | EEG | Audio | Fusion | Fusion − Audio |
| --- | --- | --- | --- | --- |
| Neutral | 49.4% | 70.1% | 65.5% | −4.6pp |
| Anger | 58.8% | 74.0% | 68.7% | −5.3pp |
| Calmness | 29.0% | 56.8% | 52.7% | −4.1pp |
| Sadness | 29.3% | 71.4% | 64.2% | −7.2pp |
| **Happiness** | **63.0%** | **44.6%** | **69.5%** | **+24.9pp** ✅ |

**Happiness is the case the thesis was built for.** It is the one emotion where
EEG *beats* audio (63.0% vs 44.6%) — and there fusion beats both unimodal models,
by 24.9 points over audio. Vocal happiness is the weakest acoustic cue in this
corpus; the neural channel compensates, and the attention mechanism exploits it.

On the other four classes audio is already strong and fusion is dragged down 4–7
points. Averaged over five classes this nets to approximately zero, which is why
the aggregate looked like "EEG adds nothing".

This is consistent with — though it does not prove — the expression-masking
rationale for multimodal affect sensing: where the vocal channel under-expresses
an emotion, the neural channel still carries it. It cannot be proven on EAV
because the corpus labels the *elicited* emotion of a scripted conversational
scenario, so "the speaker under-expressed" and "this class is acoustically
harder" are not separable here.

#### Consequence for future work

The binding constraint is a fusion mechanism that cannot decide *per trial* which
modality to trust. Concretely motivated next steps, in order:

1. **Confidence-gated fusion** — weight modalities per sample by predictive
   entropy rather than with a single learned gate. The oracle says 80.79% is
   available; a usable gate would capture part of it.
2. **Per-class fusion weights** — the trade is class-dependent and currently
   uniform.
3. **Fuse before pooling** — both encoders pool globally over time, so the
   attention module never sees temporal structure.

### Demonstration without EEG hardware

`scripts/demo_replay.py` replays held-out subjects (5, 19, 22, 26, 27, 30, 40,
42) through both the fusion and audio-only checkpoints side by side, with ground
truth shown per trial. Replaying all 800 held-out trials through it reproduces
the recorded test accuracies exactly (62.63% / 67.13%), which verifies that the
demonstration path and the evaluation path are the same code — in particular that
audio is z-scored once rather than twice.

Live microphone capture is deliberately not offered: EAV participants are Korean
speakers reading scripted prompts in studio conditions, so a presenter speaking
into a laptop microphone is off-distribution and has no ground-truth label. A
wrong prediction under those conditions would be uninterpretable.

### ⚠️ Validation and test disagreed on a single partition

| Model | Val (6 subjects) | Test (8 subjects) |
| --- | --- | --- |
| Audio-mel | 60.67% | **67.13%** |
| Fusion | **69.50%** | 62.62% |

The ranking **flips**. Fusion wins on validation by 8.83pp; audio-only wins on
test by 4.50pp. Model selection may legitimately use only validation, which
selects fusion — yet fusion is the worse model on the held-out subjects.

The most defensible reading is that the EEG stream carries information specific
to the validation subjects (6, 8, 21, 29, 33, 39) that does not transfer to the
test subjects (5, 19, 22, 26, 27, 30, 40, 42). Fusion's higher training accuracy
(90.73% vs 80.50%) is consistent with EEG contributing capacity rather than
signal.

**Consequence:** with 6 validation and 8 test subjects, between-subject variance
is large enough that neither figure ranks these two models reliably. Subject-wise
k-fold cross-validation is required before naming either as the model of record.
This is now the top open gap.

---

## Withdrawn results

Every figure below was produced with the defective pipeline. None is comparable
to the corrected results, and none should be cited.

| Figure | Previously described as | Why withdrawn |
| --- | --- | --- |
| 52.22% | Gated-fusion / CNN baseline | Degenerate EEG stream; pooled split |
| 49.21% | LSTM encoder variant | Degenerate EEG stream; pooled split |
| 63.02% | Focal-loss concatenation fusion | Degenerate EEG stream; pooled split |
| 73.70% | Ensemble | Degenerate EEG stream; pooled split |
| **78.57%** | "Model of record" | Degenerate EEG stream; pooled split |
| 82.06% | Fine-tuned validation accuracy | **Never a measurement** — hardcoded literal at `deploy_finetuned_model.py:69` |
| 84.44% | Final fine-tuned accuracy | Evaluated on a set that was 69% training data |
| 84.92% | Baseline on the same split | Evaluated on a set that was 69% training data |
| +15.55pp | Gain from attention fusion | Both terms measured with a degenerate EEG stream |
| +3.49pp | Gain from fine-tuning | A measured value minus a hardcoded one |
| +26.35pp | Cumulative gain | Sum of the above |
| 96.85% UAR | Proposal target | Never approached; best is 55.50% |

The fine-tuning stage is not reproduced in the corrected pipeline. It trained on
a partition containing 68.6% of the baseline's test set, and even on that
contaminated split the recorded comparison was **−0.48pp**.

---

## Chronological changelog

### 2026-08-08 — Pipeline corrections and re-measurement

Audited the data pipeline and found four defects, all of which produced plausible
results rather than errors. Full account in
[DATA_CORRECTIONS.md](DATA_CORRECTIONS.md).

1. **EEG read on the wrong axis.** The array is `(time, channels, trials)`;
   the loader treated it as `(segments, channels, time)` and read `seg[0,:,:]`,
   extracting a single 2 ms time-point and reinterpreting the trial axis as time.
2. **One EEG tensor shared by 100 samples.** Each subject's single `.mat` file
   was joined to all 100 of that subject's `.wav` files, so 100 samples carrying
   five different labels received a byte-identical EEG tensor. The EEG stream
   could not contribute label information; every "fusion" result was audio-only.
3. **Split defects.** Three scripts implemented the split independently, two of
   them with a different RNG than the third (69% contamination), and none was
   subject-independent (all 42 subjects in all partitions).
4. **Silent noise substitution.** EEG load failures returned
   `np.random.randn(28, 200)` with the real label attached; audio failures became
   zeros.

Also fixed: the REST API could never load the model of record (placeholder
architecture in `src/inference.py`), and `/model-info` returned hardcoded values.

**Changes.** New `src/preprocessing/eav_io.py`, `eav_labels.py`, `eav_dataset.py`;
new `src/data/splits.py` as the single split implementation; preprocessing cache
via `scripts/preprocess_eav.py`; verification via `scripts/audit_eav_alignment.py`
and `scripts/verify_data_fix.py`; rewritten `src/inference.py`; test suite grown
from 9 to 74 tests; 25 superseded scripts moved to `scripts/legacy/`.

**Verification.** Corpus audit: 42 subjects, uniform `(200, 30, 10000)`, 12,600
media files cross-checked against ground-truth labels with **0 mismatches**.

### 2026-08-08 — Documentation consolidation

Consolidated ~43 overlapping root status documents into this file. Superseded
documents moved to `docs/archive/`.

### 2026-08-07 — Repository hygiene

Purged 191 MB of model checkpoints and 18 private report documents from git
history (`.git`: 181 MB → 1.2 MB).

### 2026-02-25 → 2026-04-05 — Original experiment series (all withdrawn)

Baseline scaffolding, gated fusion, LSTM variant, focal loss, cross-modal
attention fusion, ensemble, and fine-tuning. All used the defective pipeline
described above. Retained in `docs/archive/` and `scripts/legacy/` for
provenance. See the withdrawn-results table.

---

## Open gaps

1. **Subject-wise cross-validation not run — now blocking.** The improved runs
   rank audio-only and fusion in *opposite* orders on validation and test
   (see above), so neither partition can settle which model is better. Every
   subject needs a turn in the test set. Leave-one-subject-out is implemented in
   `src/data/splits.py` (42 folds, ~17 h CPU at 100 epochs); 7-fold subject-wise
   CV is the practical compromise at ~3 h.
2. **Single seed per configuration.** Multi-seed repetition needed before
   treating any small margin as real.
3. **Video modality unused.** Present for all 8,400 trials. Likely the most
   promising route to a materially better system.
4. **FACED pre-training unused.** 123 subjects available; the model of record is
   trained from scratch on EAV.
5. **Subject normalisation partly addressed.** Euclidean Alignment is implemented
   and improved EEG-only by 10.25pp, but the EEG stream still fails to transfer
   well enough to help fusion. Adversarial subject-invariance and Riemannian
   alignment remain untried.
6. **ICA artefact removal not implemented.** Was claimed in earlier
   documentation; never existed.
7. **Model of record unresolved.** The 55.50% fusion checkpoint remains
   `outputs/model_of_record.pt` pending gap 1. On current evidence the audio-only
   log-mel model (67.13% test, 510,917 params, no EEG hardware required) is the
   stronger candidate, but promoting it on the basis of a test-set comparison
   would be exactly the selection error this project spent its audit removing.
