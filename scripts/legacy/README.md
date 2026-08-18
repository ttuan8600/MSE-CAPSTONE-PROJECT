# Superseded experiment scripts

**Nothing in this directory runs against the current code, and no number it
produced should be cited.** These scripts are retained for provenance: they are
the code that generated the results recorded in `docs/archive/`, and they are the
evidence for the defects described in [`docs/DATA_CORRECTIONS.md`](../../docs/DATA_CORRECTIONS.md).

## Why they were retired

Every script here calls `EAVMultimodalDataset(eav_data_dir=...)`, the
pre-correction constructor. That data loader:

1. read `seg[0, :, :]` from a `(time, channels, trials)` array, extracting a
   single 2 ms time-point and reinterpreting the trial axis as time;
2. paired each subject's one EEG file with all 100 of that subject's audio
   files, so 100 samples with five different labels shared one identical EEG
   tensor;
3. substituted `np.random.randn(28, 200)` for any EEG that failed to load, and
   zeros for any audio, while keeping the real label.

Consequently every "multimodal fusion" result these scripts produced was in
substance an audio-only result, and the EEG contribution reported in the archived
documents does not exist.

Several of them also split the data with `torch.randperm` while
`train_attention_fusion.py` split it with `np.random.permutation`, both seeded
`42`. Those are unrelated permutations, so `evaluate_finetuned_model.py`'s
comment "same split as training" was false: 69% of its test set had been trained
on.

## Replacement

| Retired | Use instead |
| --- | --- |
| `train_focal_loss.py`, `train_optimized.py`, `train_advanced.py`, `train_final.py`, `train.py`, `train_lstm_*.py`, `train_cnn_baseline_for_ensemble.py` | `scripts/train_attention_fusion.py --modality ... --split-strategy ...` |
| `compare_modalities.py`, `compare_encoders.py` | `scripts/run_ablations.py` |
| `evaluate_best_model.py`, `evaluate_finetuned_model.py`, `quick_eval.py` | `scripts/evaluate_model.py` |
| `finetune_attention_fusion.py` | not replaced — see below |
| `analyze_class_balance.py`, `check_class_balance.py`, `diagnose_data_quality.py` | `scripts/audit_eav_alignment.py`, `scripts/verify_data_fix.py` |
| `run_ensemble.py`, `evaluate_ensemble*.py`, `ensemble_simulation.py`, `test_ensemble_weights.py` | not replaced — the ensemble scored below the single model and was never adopted |
| `deploy_finetuned_model.py` | not replaced — it hardcoded the accuracy literal `82.06` |
| `generate_thesis_figures.py`, `generate_report_visualizations.py` | `scripts/generate_result_figures.py` |
| `validate_report_integration.py`, `check_ai_content.py` | `scripts/check_report.py` |

## A specific hazard

`generate_thesis_figures.py` hardcodes `82.06`, `84.44` and `78.57` as literals
and writes `training_curves.png` into the report's figures directory. Running it
would silently overwrite a correct figure with a withdrawn one. Its outputs have
been quarantined in `MSE_CAPSTONE_REPORT_new/figures/superseded/`.

The replacement, `scripts/generate_result_figures.py`, reads every value from
`outputs/*/results.json`, so a figure cannot drift from the number it illustrates.

The fine-tuning stage is not reproduced. It trained on a split that included 69%
of the baseline's test set, and even on that contaminated split it measured
−0.48 pp. There is no evidence it helped, and the corrected pipeline trains the
model in one stage.
