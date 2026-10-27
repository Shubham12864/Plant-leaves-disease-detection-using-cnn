#!/usr/bin/env bash
# Fails if git tracks files that do not belong in the repository:
# built documents (PDF/Word), LaTeX build products, model weights and archives,
# logs, caches, and local AI-tool folders.
set -u
cd "$(dirname "$0")/.."

bad=$(git ls-files | grep -iE \
  '\.(pdf|docx?|zip|7z|tar|gz|pt|pth|onnx|ckpt|aux|log|out|toc|synctex\.gz|pyc)$|(^|/)__pycache__/|(^|/)\.pytest_cache/|(^|/)\.venv|^\.claude/|^\.cocreate/|^thesis/(submission/)?build/' \
  || true)

if [ -n "$bad" ]; then
  echo "Tracked files that should not be in the repository:"
  echo "$bad"
  exit 1
fi
echo "repository is clean: no tracked build products, weights, logs or tool folders"
