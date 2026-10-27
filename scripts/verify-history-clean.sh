#!/usr/bin/env bash
# Fails if ANY commit reachable from any ref contains AI-assistant memory or tool
# folders, local notes, built PDFs, LaTeX build output, or the Phase 2 binaries.
# Complements verify_repo_clean.sh, which only checks the current tree.
set -u
cd "$(dirname "$0")/.."

pattern='^\.cocreate/|^\.claude/|(^|/)(MEMORY|CLAUDE|AGENTS|WALLET_INDEX|DECISIONS|ARCHITECTURE)\.md$|^Literature-Survey-Evidence-Register\.md$|^PHASE-II-ROADMAP\.md$|\.pdf$|^thesis/submission/build/|^reports/phase2_baseline/(phase2_baseline_artifacts\.zip|plantvillage_smallcnn_best\.pt)$'

bad=$(git log --all --name-only --format= | grep -E "$pattern" | sort -u || true)
if [ -n "$bad" ]; then
  echo "Found in git history:"
  echo "$bad" | sed -E 's#^(\.cocreate|\.claude)/.*#\1/ (folder)#' | sort -u
  exit 1
fi
echo "history is clean: no memory, tool, PDF or build files in any commit"
