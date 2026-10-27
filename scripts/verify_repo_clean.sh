#!/usr/bin/env bash
# Fails if git tracks files that do not belong in the repository:
# built documents (PDF/Word), LaTeX build products, model weights and archives,
# logs, caches, local AI-tool folders, and AI-assistant memory/notes files.
set -u
cd "$(dirname "$0")/.."

status=0

bad=$(git ls-files | grep -iE \
  '\.(pdf|docx?|zip|7z|tar|gz|pt|pth|onnx|ckpt|aux|log|out|toc|synctex\.gz|pyc)$|(^|/)__pycache__/|(^|/)\.pytest_cache/|(^|/)\.venv|^\.claude/|^\.cocreate/|^thesis/(submission/)?build/' \
  || true)
if [ -n "$bad" ]; then
  echo "Tracked files that should not be in the repository:"
  echo "$bad"
  status=1
fi

memory=$(git ls-files | grep -E '(^|/)(MEMORY|CLAUDE|AGENTS|GEMINI|WALLET_INDEX|DECISIONS)\.md$' || true)
if [ -n "$memory" ]; then
  echo "AI-assistant memory or notes files are tracked:"
  echo "$memory"
  status=1
fi

extra_root_md=$(git ls-files | grep -E '^[^/]+\.md$' | grep -vxE 'README\.md' || true)
if [ -n "$extra_root_md" ]; then
  echo "Only README.md may be a tracked markdown file at the repository root (put the rest under docs/):"
  echo "$extra_root_md"
  status=1
fi

notes=$(git grep -liE 'obsidian|wallet_index' -- . ':!.gitignore' ':!scripts/verify_repo_clean.sh' ':!scripts/verify-history-clean.sh' || true)
if [ -n "$notes" ]; then
  echo "Tracked files still carry note-vault (Obsidian) markup:"
  echo "$notes"
  status=1
fi

[ "$status" -eq 0 ] && echo "repository is clean: no tracked build products, weights, logs, tool folders or AI memory files"
exit "$status"
