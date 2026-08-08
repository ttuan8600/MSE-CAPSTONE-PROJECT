# Documentation

| Document | Purpose |
| --- | --- |
| [CHANGELOG.md](CHANGELOG.md) | **Results of record.** Current model, verified accuracy, experiment history, known measurement issues, open gaps. Start here. |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Loading the model, inference, deployment scenarios, rollback. |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API endpoint reference for `app.py`. |

## Which accuracy is correct?

**78.57%** — the cross-modal attention fusion model on a clean held-out test set.

Documents in [archive/](archive/) quote **82.06%**, **84.44%**, and a **+3.49pp**
fine-tuning gain. None of those are supported by a result artifact; see
[CHANGELOG.md](CHANGELOG.md#disputed-and-superseded-numbers) for the audit.

## archive/

Superseded status reports, kept for provenance only. **Do not cite them.** They
predate the accuracy audit and contain the disputed figures above.

Some files in `archive/` are local-only and deliberately excluded from version
control (thesis drafts, baseline and training notes). They exist in a working
copy but are not published to the repository.

## Not in this folder

- `README.md` at the repository root — project overview and quick start.
- `MSE-CAPSTONE-REPORT/` and `MSE_CAPSTONE_REPORT_new/` — LaTeX thesis sources,
  which keep their own documentation alongside the chapters.
