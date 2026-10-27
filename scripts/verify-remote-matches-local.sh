#!/usr/bin/env bash
# Fails unless origin/main points at exactly the same commit as local HEAD.
set -u
cd "$(dirname "$0")/.."

git fetch origin --quiet || { echo "could not reach origin"; exit 2; }
local_sha=$(git rev-parse HEAD)
remote_sha=$(git rev-parse origin/main)
if [ "$local_sha" != "$remote_sha" ]; then
  echo "local  HEAD:        $local_sha"
  echo "remote origin/main: $remote_sha"
  exit 1
fi
echo "origin/main matches local HEAD ($local_sha)"
