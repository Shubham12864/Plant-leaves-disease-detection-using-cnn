#!/usr/bin/env bash
# Fails if README.md links to a repository path that git does not track,
# so the published README never points at a missing file.
set -u
cd "$(dirname "$0")/.."

status=0
for target in $(grep -oE '\]\(([^)#]+)' README.md | sed 's/](//' | grep -vE '^(https?:|mailto:)' | sort -u); do
  if [ -z "$(git ls-files -- "$target")" ]; then
    echo "README links to a path that is not tracked: $target"
    status=1
  fi
done
[ "$status" -eq 0 ] && echo "all README links resolve to tracked files"
exit "$status"
