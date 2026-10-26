
#!/usr/bin/env python3
"""Build the Phase III / Mid-Term submission PDF."""
from __future__ import annotations
import re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for attempt in (1, 2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={OUT}", "main"],
                           cwd=HERE, capture_output=True, text=True, errors="replace")
        log = (OUT / "main.log").read_text(encoding="utf-8", errors="replace") if (OUT/"main.log").exists() else ""
        if r.returncode != 0:
            print(f"FAILED on pass {attempt}", file=sys.stderr)
            for line in log.splitlines():
                if line.startswith("!"):
                    print("  " + line, file=sys.stderr)
            return 1
    pdf = OUT / "main.pdf"
    if not pdf.exists():
        print("ERROR: no PDF", file=sys.stderr); return 1
    m = re.search(r"\((\d+) pages", log)
    warn = [l for l in log.splitlines() if l.startswith("LaTeX Warning") and "undefined" in l.lower()]
    print(f"built {pdf}  ({m.group(1) if m else '?'} pages, {pdf.stat().st_size/1024:.0f} KB)")
    if warn:
        print(f"{len(warn)} unresolved reference(s)"); return 1
    print("no unresolved references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
