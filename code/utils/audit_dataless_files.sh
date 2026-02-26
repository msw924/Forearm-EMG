#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"

printf "Scanning for dataless (cloud-only) files under: %s\n" "$root"
find "$root" -type f -flags +dataless
