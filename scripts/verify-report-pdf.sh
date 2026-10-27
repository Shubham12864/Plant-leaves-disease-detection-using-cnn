#!/usr/bin/env bash
# Fails unless the BUILT submission PDF actually contains what the sources promise.
# Reads the text of thesis/submission/build/main.pdf, so a stale or broken build is caught.
set -u
cd "$(dirname "$0")/.."

PDF=thesis/submission/build/main.pdf
[ -f "$PDF" ] || { echo "FAIL: $PDF does not exist; run python thesis/submission/build.py"; exit 1; }

python - "$PDF" <<'EOF'
import sys
from pypdf import PdfReader

text = "".join((p.extract_text() or "") for p in PdfReader(sys.argv[1]).pages)
must_have = [
    "Documents in this submission",   # the document list the summary promises
    "The result in plain words",
    "Glossary",
    "Figure 2.1",                     # both figures rendered, not just referenced
    "Figure 2.2",
    "run_all.py",
    "--check",                        # the flag must not be swallowed by the path macro
    "--verify",
    "0.08 points",                    # corrected abstention gain
]
must_not_have = [
    "coordinator",                    # the report describes itself, not the university's request
    "DD/MM/YYYY",
    "Verify before submitting",
    "0.8 points",
]
bad = [f"missing: {s!r}" for s in must_have if s not in text]
bad += [f"present but must not be: {s!r}" for s in must_not_have if s in text]
if bad:
    print("FAIL: the built PDF does not match the sources")
    print("\n".join("    " + b for b in bad))
    sys.exit(1)
print("built PDF contains the document list, summary, glossary, both figures and the corrected figures")
EOF
