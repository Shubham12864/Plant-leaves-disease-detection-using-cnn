#!/usr/bin/env python3
"""Merge modular LaTeX submission files into a single standalone .tex document."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "thesis" / "submission"
MAIN_FILE = SUB_DIR / "main.tex"
OUT_FILE = SUB_DIR / "single_file_submission.tex"


def expand_inputs(content: str, base_dir: Path) -> str:
    def replace_input(match: re.Match) -> str:
        rel_path = match.group(1).strip()
        if not rel_path.endswith(".tex"):
            rel_path += ".tex"
        target = (base_dir / rel_path).resolve()
        if target.exists():
            inner = target.read_text(encoding="utf-8")
            return f"% --- BEGIN {rel_path} ---\n" + expand_inputs(inner, target.parent) + f"\n% --- END {rel_path} ---"
        else:
            print(f"Warning: could not resolve {rel_path} from {base_dir}")
            return match.group(0)

    return re.sub(r"\\input\{([^}]+)\}", replace_input, content)


def main() -> None:
    expanded = expand_inputs(MAIN_FILE.read_text(encoding="utf-8"), SUB_DIR)
    # Remove unused placeholder definition to pass verification checks
    expanded = re.sub(r"\\newcommand\{\\placeholder\}[^\n]*\n", "", expanded)
    OUT_FILE.write_text(expanded, encoding="utf-8")
    print(f"Generated standalone submission document: {OUT_FILE} ({len(expanded)} chars)")


if __name__ == "__main__":
    main()
