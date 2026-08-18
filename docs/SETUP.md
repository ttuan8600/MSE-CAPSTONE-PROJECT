# Setup Guide

Getting a working environment from a fresh `git clone`, on a machine that has
never seen this project.

**Read this first:** the repository contains code only. The datasets (~47 GB) and
the trained model checkpoints are **not** in git and must be transferred
separately. See [Datasets](#3-datasets-not-in-git) and
[Model checkpoints](#4-model-checkpoints-not-in-git). Without them, the test
suite still passes and the code imports cleanly, but you cannot train or run
inference.

---

## 1. Prerequisites

| | Requirement |
| --- | --- |
| Python | **3.11 or newer** (3.13 is what the pinned versions were validated on) |
| Git | any recent version |
| Disk | ~2 GB for the environment, plus ~47 GB if you copy the datasets |
| GPU | not required — the project runs and was evaluated on CPU |

Check your Python:

```bash
python --version     # must print 3.11.x or newer
```

Python 3.10 or older will fail: the pinned `numpy`, `pandas`, `scipy`, and
`scikit-learn` all require >= 3.11.

---

## 2. Environment

```bash
git clone https://github.com/ttuan8600/MSE-CAPSTONE-PROJECT.git
cd MSE-CAPSTONE-PROJECT

python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS / Git Bash
source .venv/bin/activate     # Git Bash on Windows: source .venv/Scripts/activate
```

Install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                      # required — makes `import src...` resolve
```

Optional extras:

```bash
pip install -r requirements-dev.txt   # pytest, black, flake8, jupyter
pip install -r requirements_api.txt   # flask, flask-cors, requests (for app.py)
```

### Verify

```bash
pytest
```

Expect **74 passed**. This works with no datasets present — the tests use
synthetic tensors and temporary files.

```bash
python -c "import torch, librosa, sklearn; from src.models.attention_fusion import CrossModalAttentionFusion; print('imports OK')"
```

### CUDA (optional)

`requirements.txt` installs CPU wheels, which is what this project runs on.
Inference is CPU-only by design and takes ~2-5 ms per sample. If you want CUDA,
install torch from the PyTorch index *instead of* the pinned CPU wheels:

```bash
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Do not mix: install one or the other, not both.

---

## 3. Datasets (not in git)

`data/` is gitignored (~47 GB). A fresh clone has an empty `data/` tree.
Two datasets are used, and the loaders expect these **exact** paths.

### FACED — EEG pre-training

```
data/raw/Processed_data/Processed_data/
├── sub000.pkl
├── sub001.pkl
└── ...            (123 files)
```

Each pickle holds an array of shape `(28 channels, 32 trials, 7500 timesteps)`.
Consumed by `FAEDDataset` in `src/preprocessing/data_loader.py`.

### EAV — multimodal fine-tuning and evaluation

```
data/raw/EAV/EAV/
├── subject1/
│   ├── Audio/     002_Trial_02_Speaking_Neutral_Aud.wav, ...
│   ├── EEG/       *.mat
│   └── Video/     *.mp4
├── subject2/
└── ...            (42 subjects)
```

Each subject directory holds one `subject<N>_eeg.mat` (a `(10000, 30, 200)`
array in `(time, channels, trials)` order), one `subject<N>_eeg_label.mat` (the
`(10, 200)` one-hot ground-truth matrix), 100 `.wav` files and 200 `.mp4` files.

Emotion labels come from the `.mat` label matrix. The three-digit filename prefix
(`002_Trial_02_...`) is the 1-based trial index that pairs each audio clip with
its own EEG trial.

### Transferring

The datasets are large and not publicly redistributable from this repo. Copy
`data/` wholesale from the original machine (external drive, `robocopy`, `rsync`)
and preserve the directory names exactly — the paths above are hardcoded as
defaults in the training scripts.

Verify placement and structural integrity:

```bash
python scripts/audit_eav_alignment.py
```

Expect `42 subjects`, `(200, 30, 10000)` for every subject, and
`12600 files, 0 mismatches`. Any error here means the copy is incomplete or the
nesting is wrong — fix it before preprocessing.

---

## 3b. Build the preprocessing cache (required before training)

Training and evaluation read a preprocessed cache rather than the raw `.mat` and
`.wav` files. Build it once:

```bash
python scripts/preprocess_eav.py
```

Takes about 5 minutes for all 42 subjects and writes ~2.8 GB to
`data/processed/eav/`. It band-passes EEG to 0.5–45 Hz, decimates 500 Hz → 125 Hz,
and extracts 13 MFCCs at 16 kHz.

Verify:

```bash
python -c "
from src.preprocessing.eav_dataset import EAVMultimodalDataset
print(EAVMultimodalDataset().describe())
"
```

Expect 4,200 samples, 42 subjects, EEG `(30, 2500)`, audio `(13, 2101)`, and 840
samples per class.

Optionally confirm the pipeline corrections are in force:

```bash
python scripts/verify_data_fix.py
```

---

## 4. Model checkpoints (not in git)

`outputs/` and all `*.pt` files are gitignored and were deliberately purged from
git history (they were 191 MB and made the repository unclonable in practice).
**A fresh clone has no trained model.**

To run inference you need:

| File | Size | What it is |
| --- | --- | --- |
| `outputs/model_of_record.pt` | 3.7 MB | **Model of record** — see [CHANGELOG.md](CHANGELOG.md) for its measured accuracy |

Checkpoints written before 2026-08-08 (`attention_fusion_model_best.pt`,
`attention_fusion_finetuned_best.pt`, `focal_loss_model_best.pt`,
`lstm_model_best.pt`) were trained on the defective pipeline described in
[DATA_CORRECTIONS.md](DATA_CORRECTIONS.md). They still load — the inference
wrapper recognises the legacy format and flags it — but their predictions are not
meaningful and their reported accuracies are withdrawn.

### Getting them

**Option A — copy from the original machine (fastest).** Create `outputs/` in the
clone and copy the `.pt` files in. Only the checkpoints are needed; the
per-epoch subdirectories are not.

**Option B — publish once as a GitHub Release** (recommended if you will clone
repeatedly). Releases hold binaries without bloating git history:

```bash
gh release create v2.0 outputs/model_of_record.pt \
    --title "Model of record (subject-independent)" \
    --notes "Cross-modal attention fusion checkpoint. See docs/CHANGELOG.md."
```

Then on any machine:

```bash
mkdir -p outputs
gh release download v2.0 --dir outputs
```

**Option C — retrain.** Requires the EAV dataset and the preprocessing cache.
About 10 minutes per run on CPU:

```bash
python scripts/train_attention_fusion.py          # the model of record
python scripts/run_ablations.py                   # all four reported runs (~40 min)
```

---

## 5. Running things

```bash
# Tests (no data needed)
pytest

# Verify the raw data before anything else (needs EAV)
python scripts/audit_eav_alignment.py

# Build the preprocessing cache, once (needs EAV)
python scripts/preprocess_eav.py

# Train the model of record
python scripts/train_attention_fusion.py

# Reproduce every reported result
python scripts/run_ablations.py

# Evaluate a finished run on its own recorded split
python scripts/evaluate_model.py outputs/multimodal_subject_independent_*/

# REST API (needs the checkpoint + requirements_api.txt)
python app.py --model outputs/model_of_record.pt --port 5000
```

The API expects EEG shaped `(30, 2500)` at 125 Hz and MFCCs shaped `(13, 2101)`.
Both modalities are required; omitting one returns HTTP 400 rather than silently
substituting zeros.

---

## 6. Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
You skipped `pip install -e .`, or you are not in the activated venv. Re-run
both. Scripts import as `from src.models...`, which requires the editable install.

**`ModuleNotFoundError: No module named 'librosa'`**
`librosa` was historically missing from `requirements.txt` while being imported
by the EAV loader. Re-install from the current `requirements.txt`.

**`invalid pyproject.toml config: project.authors[0]` during `pip install -e .`**
You are on a clone from before the packaging fix. Pull the latest `main`.

**`ERROR: Invalid requirement: """API Dependencies..."""`**
Same cause — the old `requirements_api.txt` began with a Python docstring, which
pip cannot parse. Pull the latest `main`.

**`FileNotFoundError: Model not found: outputs/...pt`**
Checkpoints are not in git. See [section 4](#4-model-checkpoints-not-in-git).

**`FileNotFoundError: No FACED .pkl files found` / EAV dataset length is 0**
Datasets are not in git, or the directory nesting is wrong. The doubled folder
names (`Processed_data/Processed_data`, `EAV/EAV`) are intentional — see
[section 3](#3-datasets-not-in-git).

**`EAVCacheMissing: EAV cache not found at data/processed/eav/manifest.json`**
You have not built the preprocessing cache. Run `python scripts/preprocess_eav.py`
— see [section 3b](#3b-build-the-preprocessing-cache-required-before-training).

**`EAVCacheMissing: ... is version 1, expected 2`**
The cache predates a preprocessing change. Rebuild with
`python scripts/preprocess_eav.py --force`.

**`EAVDataError: no EEG variable found`**
The `.mat` file uses a variable name other than `seg` or `seg1`. This is fatal by
design — the previous loader substituted random noise here, silently poisoning
training data. Check the file is not truncated.

**`UnicodeEncodeError: 'charmap' codec can't encode character`**
Windows console using cp1252 against a script printing Unicode. Force UTF-8:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

**`torch.cuda.is_available()` returns False**
Expected with the pinned CPU wheels. The project does not require CUDA. To
enable it, see [CUDA](#cuda-optional).

**Python version errors from numpy/pandas during install**
You are on Python 3.10 or older. This project needs 3.11+.

**Accuracy looks suspiciously high**
Check the split strategy recorded in the run's `results.json`. A
`pooled_random` run places every subject in every partition and scores far above
a `subject_independent` run on the same data; only the latter measures
generalisation to a new person. Historically the training and evaluation scripts
also seeded *different* random number generators, putting ~69% of the training
set into the "test" set. Both issues are documented in
[DATA_CORRECTIONS.md](DATA_CORRECTIONS.md).

**Old checkpoints load but predict nonsense**
Checkpoints from before 2026-08-08 were trained on mis-indexed EEG. The inference
wrapper flags them (`metadata["format"] == "legacy"`). Retrain instead.
