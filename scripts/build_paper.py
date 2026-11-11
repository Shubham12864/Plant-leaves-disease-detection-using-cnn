#!/usr/bin/env python3
"""Build the IEEE conference research paper PDF."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
TEX_FILE = PAPER_DIR / "paper.tex"
PDF_FILE = PAPER_DIR / "paper.pdf"


def main() -> int:
    if not TEX_FILE.exists():
        print(f"Error: {TEX_FILE} does not exist.", file=sys.stderr)
        return 1

    print("Compiling IEEE Research Paper using pdflatex...")
    for pass_num in (1, 2):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
            cwd=PAPER_DIR,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if res.returncode != 0:
            print(f"Failed on pass {pass_num}!", file=sys.stderr)
            for line in res.stdout.splitlines():
                if line.startswith("!"):
                    print("  " + line, file=sys.stderr)
            return 1

    if not PDF_FILE.exists():
        print("Error: paper.pdf was not produced.", file=sys.stderr)
        return 1

    print(f"✅ Successfully compiled {PDF_FILE} ({PDF_FILE.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
