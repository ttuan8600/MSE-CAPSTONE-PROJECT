# Documentation

| Document | Purpose |
| --- | --- |
| [DATA_CORRECTIONS.md](DATA_CORRECTIONS.md) | **Read first.** The four pipeline defects found on 2026-08-08, the evidence, and what changed. Explains why older numbers are withdrawn. |
| [SETUP.md](SETUP.md) | **Start here on a new machine.** Environment, datasets, preprocessing cache, troubleshooting. |
| [CHANGELOG.md](CHANGELOG.md) | **Results of record.** Current model, verified accuracy, ablations, significance tests, open gaps. |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Loading the model, inference, deployment scenarios, rollback. |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API endpoint reference for `app.py`. |

## Which accuracy is correct?

**55.50%** — cross-modal attention fusion, on eight subjects held out entirely
from training. Chance is 20.0%.

Equally important: **fusion does not significantly beat audio alone** (54.75%,
p = 0.78). See [CHANGELOG.md](CHANGELOG.md#corrected-results).

Everything reported before 2026-08-08 — **78.57%, 82.06%, 84.44%**, and the gains
derived from them — is **withdrawn**. Those experiments ran on a pipeline in which
all 100 samples from a subject shared one identical EEG tensor, evaluated on a
split where every subject appeared in training and test simultaneously. See
[DATA_CORRECTIONS.md](DATA_CORRECTIONS.md).

## archive/

Superseded status reports and report drafts, kept for provenance only.
**Do not cite or compile them.**

- [archive/superseded_reports/](archive/superseded_reports/) — earlier LaTeX
  thesis drafts. The current report is `MSE_CAPSTONE_REPORT_new/`.

Some files in `archive/` are local-only and deliberately excluded from version
control. They exist in a working copy but are not published to the repository.

## Not in this folder

- `README.md` at the repository root — project overview and quick start.
- `MSE_CAPSTONE_REPORT_new/` — the current LaTeX thesis source. Verify it with
  `python scripts/check_report.py` before building.
- `scripts/legacy/README.md` — why 25 experiment scripts were retired.
