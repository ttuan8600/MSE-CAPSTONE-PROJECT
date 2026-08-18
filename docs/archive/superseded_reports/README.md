# Superseded report drafts

**Do not compile, submit, or cite anything in this directory.**

These are earlier drafts of the capstone report. Every accuracy figure in them
is withdrawn — see [DATA_CORRECTIONS.md](../../DATA_CORRECTIONS.md) for why.

| Path | What it is |
| --- | --- |
| `MSE-CAPSTONE-REPORT/` | Earlier chaptered draft (April 2026) |
| `PROJECT_REPORT.tex` | Single-file draft |
| `PROJECT_REPORT_FINAL.tex` | Single-file draft, later revision |

## The current report

**[`MSE_CAPSTONE_REPORT_new/`](../../../MSE_CAPSTONE_REPORT_new/)** — build from
`main.tex`. Verify it before building:

```bash
python scripts/check_report.py
```

## What changed

The drafts here claim 78.57% / 82.06% / 84.44% accuracy and a +15.55pp gain from
cross-modal attention fusion. All of it is withdrawn. The pipeline that produced
those numbers fed every sample from a subject an identical EEG tensor, so the
"multimodal" models were audio-only in substance, and evaluation used a split in
which all 42 subjects appeared in training, validation and test simultaneously.

The current report measures **55.50%** on eight held-out subjects and reports
that fusion does **not** significantly beat audio alone (p = 0.78). It also
contains a chapter documenting the defects, which the drafts here do not.
