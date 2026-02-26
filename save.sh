#!/usr/bin/env bash
# Usage:  ./save.sh "commit message"
#         ./save.sh              (uses a timestamped default message)
set -euo pipefail

MSG="${1:-"WIP snapshot $(date '+%Y-%m-%d %H:%M')"}"

# Stage only source-controlled directories/files — never raw data or outputs.
git add \
  .gitignore \
  .githooks/ \
  .vscode/ \
  README.md \
  save.sh \
  code/ \
  docs/ \
  reports/tables/

# Check if there is anything new to commit.
if git diff --cached --quiet; then
  echo "Nothing to commit — working tree is clean."
  exit 0
fi

git commit -m "$MSG"
git push origin main
echo "Saved and pushed: $MSG"
