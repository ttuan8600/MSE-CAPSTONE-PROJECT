"""Structural sanity check for the LaTeX report.

Catches the errors that a missing LaTeX toolchain would otherwise let through to
compile time: unresolved cross-references, unbalanced environments, missing
figure files, and any reappearance of a withdrawn accuracy figure outside a
context that marks it as withdrawn.

Run from the project root::

    python scripts/check_report.py
    python scripts/check_report.py --report-dir MSE_CAPSTONE_REPORT_new
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|autoref|cref)\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite\{([^}]*)\}")
BEGIN_RE = re.compile(r"\\begin\{([^}]*)\}")
END_RE = re.compile(r"\\end\{([^}]*)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.MULTILINE)

#: Figures withdrawn on 2026-08-08. Any occurrence must sit near wording that
#: marks it as withdrawn, otherwise it reads as a live claim.
WITHDRAWN = ["78.57", "82.06", "84.44", "84.92", "63.02", "52.22", "49.21", "73.70"]
WITHDRAWN_CONTEXT = (
    "withdraw", "not a measurement", "superseded", "earlier draft", "previously",
    "original", "artifact", "contaminated", "hardcoded", "no longer", "defect",
    "not used as a comparison", "disposition", "cannot", "smaller number",
    "degenerate", "pooled split", "corrections",
)

#: Lines either side of a hit to search for a withdrawal marker.
CONTEXT_WINDOW = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="MSE_CAPSTONE_REPORT_new")
    args = parser.parse_args()

    root = Path(args.report_dir)
    if not root.is_dir():
        print(f"ERROR: {root} not found", file=sys.stderr)
        return 2

    files = sorted((root / "chapters").glob("*.tex")) + [root / "main.tex"]
    files = [f for f in files if f.exists()]

    labels: set[str] = set()
    refs: list[tuple[str, str]] = []
    cites: list[tuple[str, str]] = []
    problems: list[str] = []

    for path in files:
        text = path.read_text(encoding="utf-8")

        duplicates = [k for k, v in Counter(LABEL_RE.findall(text)).items() if v > 1]
        for dup in duplicates:
            problems.append(f"{path.name}: duplicate label '{dup}'")
        labels |= set(LABEL_RE.findall(text))

        refs += [(path.name, r) for r in REF_RE.findall(text)]
        for group in CITE_RE.findall(text):
            cites += [(path.name, key.strip()) for key in group.split(",")]

        opens, closes = Counter(BEGIN_RE.findall(text)), Counter(END_RE.findall(text))
        for env in sorted(set(opens) | set(closes)):
            if opens[env] != closes[env]:
                problems.append(
                    f"{path.name}: environment '{env}' begin={opens[env]} end={closes[env]}"
                )

        stripped = re.sub(r"\\[{}]", "", text)
        if stripped.count("{") != stripped.count("}"):
            problems.append(
                f"{path.name}: brace mismatch "
                f"{{={stripped.count('{')} }}={stripped.count('}')}"
            )

        for graphic in GRAPHIC_RE.findall(text):
            candidate = root / graphic
            alt = root / "figures" / Path(graphic).name
            if not candidate.exists() and not alt.exists():
                problems.append(f"{path.name}: missing figure '{graphic}'")

        # A withdrawal marker may sit a few lines away (table headers, wrapped
        # prose), so check a window rather than the single line.
        lines = text.splitlines()
        for line_no, line in enumerate(lines, 1):
            for figure in WITHDRAWN:
                if figure not in line:
                    continue
                lo, hi = max(0, line_no - 1 - CONTEXT_WINDOW), line_no + CONTEXT_WINDOW
                window = " ".join(lines[lo:hi]).lower()
                if not any(c in window for c in WITHDRAWN_CONTEXT):
                    problems.append(
                        f"{path.name}:{line_no}: withdrawn figure {figure}% "
                        f"without withdrawal context"
                    )
                break

    for name, ref in refs:
        if ref not in labels:
            problems.append(f"{name}: unresolved \\ref{{{ref}}}")

    bib = root / "references.bib"
    if bib.exists():
        keys = set(BIB_ENTRY_RE.findall(bib.read_text(encoding="utf-8", errors="ignore")))
        for name, key in cites:
            if key and key not in keys:
                problems.append(f"{name}: citation '{key}' not in references.bib")

    print(f"files checked : {len(files)}")
    print(f"labels        : {len(labels)}")
    print(f"refs          : {len(refs)}")
    print(f"citations     : {len(cites)}")
    print(f"problems      : {len(problems)}\n")

    for problem in problems:
        print(f"  {problem}")

    if not problems:
        print("  none")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
