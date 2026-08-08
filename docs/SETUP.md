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

Expect **9 passed**. This works with no datasets present — the tests use
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

Consumed by `EAVMultimodalDataset`. Emotion labels are parsed **from the audio
filenames** (the `Neutral` / `Anger` / `Calmness` / `Sadness` / `Happiness`
token), not from the `.mat` label files — so the Audio folder must be present
even if you only care about EEG.

### Transferring

The datasets are large and not publicly redistributable from this repo. Copy
`data/` wholesale from the original machine (external drive, `robocopy`, `rsync`)
and preserve the directory names exactly — the paths above are hardcoded as
defaults in the training scripts.

Verify placement:

```bash
python -c "
from src.preprocessing.data_loader import EAVMultimodalDataset
d = EAVMultimodalDataset('data/raw/EAV/EAV', load_audio=True)
print('EAV samples:', len(d))          # expect 4200
"
```

---

## 4. Model checkpoints (not in git)

`outputs/` and all `*.pt` files are gitignored and were deliberately purged from
git history (they were 191 MB and made the repository unclonable in practice).
**A fresh clone has no trained model.**

To run inference you need at minimum:

| File | Size | What it is |
| --- | --- | --- |
| `outputs/attention_fusion_model_best.pt` | 3.7 MB | **Model of record**, 78.57% test accuracy |

Optionally, for comparison work:

| File | Size |
| --- | --- |
| `outputs/focal_loss_model_best.pt` | 3.1 MB |
| `outputs/lstm_model_best.pt` | 3.8 MB |
| `outputs/attention_fusion_finetuned_best.pt` | 3.7 MB |

### Getting them

**Option A — copy from the original machine (fastest).** Create `outputs/` in the
clone and copy the `.pt` files in. Only the checkpoints are needed; the
per-epoch subdirectories are not.

**Option B — publish once as a GitHub Release** (recommended if you will clone
repeatedly). Releases hold binaries without bloating git history:

```bash
gh release create v1.0 outputs/attention_fusion_model_best.pt \
    --title "Model of record (78.57%)" \
    --notes "Cross-modal attention fusion checkpoint. See docs/CHANGELOG.md."
```

Then on any machine:

```bash
mkdir -p outputs
gh release download v1.0 --dir outputs
```

**Option C — retrain.** Requires the EAV dataset. CPU training takes several
hours:

```bash
python scripts/train_attention_fusion.py
```

---

## 5. Running things

```bash
# Tests (no data needed)
pytest

# Train the model of record (needs EAV)
python scripts/train_attention_fusion.py

# REST API (needs the checkpoint + requirements_api.txt)
python app.py --model outputs/attention_fusion_model_best.pt --port 5000
```

> **Known defect:** `src/inference.py` builds a placeholder concat-and-linear
> model rather than the attention fusion architecture, so `app.py` cannot load
> `attention_fusion_model_best.pt` — `load_state_dict` fails on mismatched keys.
> The API does not currently serve the model of record. Tracked in
> [CHANGELOG.md](CHANGELOG.md#open-gaps).

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

**Training accuracy looks far better than 78.57%**
Do not trust it without checking the split. The training and evaluation scripts
historically seeded *different* random number generators, which put ~69% of the
training set into the "test" set. See
[CHANGELOG.md](CHANGELOG.md#known-measurement-issue-traintest-contamination).
