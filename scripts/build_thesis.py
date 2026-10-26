#!/usr/bin/env python3
"""Build the dissertation PDF and check it actually compiled.

XeLaTeX needs two passes for the table of contents and cross-references to settle.
The build fails loudly rather than leaving a stale PDF in place, because a
missing table of contents is exactly the kind of defect that gets found after
printing.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis"
MAIN = "main"
OUT = THESIS / "build"


def run(engine: str) -> int:
    return subprocess.run(
        [engine, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={OUT}", MAIN],
        cwd=THESIS, capture_output=True, text=True, errors="replace",
    ).returncode


def main() -> int:
    OUT.mkdir(exist_ok=True)
    engine = "xelatex"
    if shutil.which(engine) is None:
        print(f"ERROR: {engine} not on PATH", file=sys.stderr)
        return 2

    for attempt in (1, 2):
        code = run(engine)
        log = (OUT / f"{MAIN}.log").read_text(encoding="utf-8", errors="replace")
        if code != 0:
            print(f"FAILED on pass {attempt}", file=sys.stderr)
            for line in log.splitlines():
                if line.startswith("!") or "Undefined control sequence" in line:
                    print("  " + line, file=sys.stderr)
            # show the offending context
            m = re.search(r"\n(.*\n){0,3}.*\n!.*", log)
            if m:
                print(m.group(0)[-600:], file=sys.stderr)
            return 1

    log = (OUT / f"{MAIN}.log").read_text(encoding="utf-8", errors="replace")
    pdf = OUT / f"{MAIN}.pdf"
    if not pdf.exists():
        print("ERROR: no PDF produced", file=sys.stderr)
        return 1

    # MiKTeX wraps this line across two lines, so search the whole log.
    m = re.search(r"\((\d+) pages", log)
    pages = int(m.group(1)) if m else -1

    warnings = [ln for ln in log.splitlines()
                if ln.startswith("LaTeX Warning") and "undefined" in ln.lower()]

    print(f"built {pdf.relative_to(ROOT)}  ({pages} pages, {pdf.stat().st_size/1024:.0f} KB)")
    if warnings:
        print(f"{len(warnings)} unresolved reference(s):")
        for w in warnings[:10]:
            print("  " + w)
        return 1
    print("no unresolved references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
